"""Grant management inside the existing Telegram adapter, never an agent tool.

Views are intentionally ephemeral. Restart, expiry, refresh and replacement
invalidate their nonces. No human decision is recovered or replayed on startup.
"""

import asyncio
import html
import re
import secrets
import time
from dataclasses import dataclass, field

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from gateway import permission_grants as grants
from gateway.permission_bridge import BridgeError

VIEW_TTL = 300
MAX_VIEWS = 128
UNAVAILABLE = "Grant inspection unavailable. Use /grants to try again."
STALE = "This grant view expired or changed. Open /grants again."


@dataclass(eq=False, repr=False)
class View:
    owner: grants.GrantOwner
    thread: str | None
    generation: str
    deadline: float
    message_id: str = ""
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    after: int = 0
    previous: tuple = ()
    rows: tuple = ()
    more: bool = False
    inspected: grants.Grant | None = None

    @property
    def key(self):
        return self.owner.user, self.owner.chat


class GrantUI:
    def __init__(self, adapter):
        self.adapter = adapter
        self.views = {}
        self.generations = {}

    def prune(self):
        now = time.monotonic()
        for nonce, view in list(self.views.items()):
            if view.deadline <= now:
                self.views.pop(nonce, None)
        active = {view.key for view in self.views.values()}
        for key, (_, deadline) in list(self.generations.items()):
            if key not in active and deadline <= now:
                self.generations.pop(key, None)

    def current(self, view):
        return (view.deadline > time.monotonic()
                and self.generations.get(view.key, (None,))[0] == view.generation)

    @staticmethod
    def button(view, label, action):
        return InlineKeyboardButton(label, callback_data=f"hg:{view.nonce}:{action}")

    async def send(self, view, text, buttons, query=None):
        if not self.current(view):
            return
        markup = InlineKeyboardMarkup(buttons) if buttons else None
        try:
            if query is None:
                kwargs = {"message_thread_id": int(view.thread)} if view.thread else {}
                message = await self.adapter._bot.send_message(
                    chat_id=int(view.owner.chat), text=text, parse_mode="HTML",
                    reply_markup=markup, **kwargs)
                view.message_id = str(message.message_id)
            else:
                await query.edit_message_text(text=text, parse_mode="HTML", reply_markup=markup)
            # Delivery uncertainty never leaves a usable callback. A late send
            # may be visible, but only the current generation receives a nonce.
            if buttons and self.current(view) and view.message_id:
                self.views[view.nonce] = view
        except Exception:
            pass  # Telegram errors may contain sensitive response data.

    async def page(self, view, query=None):
        rows = await asyncio.to_thread(grants.list_grants, view.owner, view.after)
        if not self.current(view):
            return
        view.rows, view.more = rows[:grants.PAGE_SIZE], len(rows) > grants.PAGE_SIZE
        view.inspected = None
        text = "Standing grants\n" + ("Select a grant to inspect its full scope.\n" if rows else "No more grants.\n")
        if rows and not view.more:
            text += "End of recorded grants at this time.\n"
        buttons = []
        for index, grant in enumerate(view.rows):
            text += f"\n{index + 1}. {grant.grant_id}\n{grant.status}\n"
            buttons.append([self.button(view, f"Inspect {index + 1}", f"i{index}")])
        navigation = []
        if view.previous:
            navigation.append(self.button(view, "Previous", "p"))
        if view.more:
            navigation.append(self.button(view, "Next", "n"))
        if navigation:
            buttons.append(navigation)
        buttons.append([self.button(view, "Refresh", "f")])
        await self.send(view, "<pre>" + html.escape(text) + "</pre>", buttons, query)

    async def command(self, message):
        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        user_id, chat_id = str(getattr(user, "id", "")), str(getattr(chat, "id", ""))
        thread = getattr(message, "message_thread_id", None)
        # Intake prefilters permit pairing requests. Human grant management
        # requires the same strict authorization as a native button callback.
        if (not user or getattr(user, "is_bot", False)
                or not self.adapter._is_callback_user_authorized(
                    user_id, chat_id=chat_id, chat_type=getattr(chat, "type", None),
                    thread_id=str(thread) if thread is not None else None,
                    user_name=getattr(user, "first_name", None))):
            return
        self.prune()
        key = user_id, chat_id
        for nonce, old in list(self.views.items()):
            if old.key == key:
                self.views.pop(nonce, None)
        generation, deadline = secrets.token_urlsafe(24), time.monotonic() + VIEW_TTL
        if key not in self.generations and len(self.generations) >= MAX_VIEWS:
            return
        self.generations[key] = generation, deadline
        try:
            owner = grants.configured_owner(user_id, chat_id)
        except Exception:
            # No endpoint or credential-derived text is exposed on errors.
            try:
                await message.reply_text(UNAVAILABLE)
            except Exception:
                pass
            return
        view = View(owner, str(thread) if thread is not None else None, generation, deadline)
        try:
            await self.page(view)
        except Exception:
            await self.send(view, UNAVAILABLE, [])

    async def callback(self, query):
        self.prune()
        parts = query.data.split(":")
        view = self.views.get(parts[1]) if len(parts) == 3 else None
        message = getattr(query, "message", None)
        user = getattr(query, "from_user", None)
        thread = getattr(message, "message_thread_id", None)
        if (view is None or not self.current(view)
                or getattr(user, "is_bot", False)
                or str(getattr(user, "id", "")) != view.owner.user
                or str(getattr(message, "chat_id", "")) != view.owner.chat
                or str(getattr(message, "message_id", "")) != view.message_id
                or (str(thread) if thread is not None else None) != view.thread):
            await self.answer(query, STALE)
            return
        action = parts[2]
        index = int(action[1:]) if re.fullmatch(r"i[0-4]", action) else None
        valid = (action == "f" or (action == "b" and view.inspected is not None)
                 or (action == "p" and view.previous and view.inspected is None)
                 or (action == "n" and view.more and view.inspected is None)
                 or (index is not None and index < len(view.rows) and view.inspected is None)
                 or (action == "r" and view.inspected is not None and not view.inspected.revoked))
        if not valid:
            await self.answer(query, STALE)
            return
        # A single-use nonce is claimed before any await, including Telegram
        # ACK. New views always get a new nonce, even on a read-only refresh.
        self.views.pop(view.nonce, None)
        view.nonce = secrets.token_urlsafe(24)
        await self.answer(query, "Checking grant service…")
        try:
            if grants.configured_owner(view.owner.user, view.owner.chat) != view.owner:
                raise BridgeError()
            if index is not None:
                view.inspected = await asyncio.to_thread(grants.refresh_grant, view.owner, view.rows[index])
                text = grants.grant_prompt(view.inspected)
                buttons = [[self.button(view, "Back to grants", "b")]]
                if not view.inspected.revoked:
                    buttons.insert(0, [self.button(view, "Revoke future use", "r")])
                await self.send(view, text, buttons, query)
            elif action == "r":
                current = await asyncio.to_thread(grants.refresh_grant, view.owner, view.inspected)
                if (not self.current(view)
                        or grants.configured_owner(view.owner.user, view.owner.chat) != view.owner):
                    return
                if current.revoked:
                    result = "already_revoked"
                else:
                    result = await asyncio.to_thread(grants.revoke_grant, view.owner, current)
                label = {
                    "revoked": "Grant revoked for future use.",
                    "already_revoked": "Grant was already revoked for future use.",
                    "revoked_after_uncertain_response": "The service now reports this grant revoked; the original response was lost.",
                    "rejected": "Revocation rejected by the service. Refresh to inspect current state.",
                    "unconfirmed": "Revocation is unconfirmed. No automatic retry. Refresh to inspect current state.",
                }[result]
                label += " Execution already authorized cannot be recalled."
                view.inspected = None
                await self.send(view, label, [[self.button(view, "Refresh grants", "f")]], query)
            else:
                if action == "p":
                    view.after, view.previous = view.previous[-1], view.previous[:-1]
                elif action == "n":
                    view.previous = (view.previous + (view.after,))[-128:]
                    view.after = view.rows[-1].cursor
                await self.page(view, query)
        except Exception:
            await self.send(view, UNAVAILABLE, [], query)

    @staticmethod
    async def answer(query, text):
        try:
            await query.answer(text=text)
        except Exception:
            pass


def native_grants(adapter):
    ui = getattr(adapter, "_native_grants_ui", None)
    if ui is None:
        ui = adapter._native_grants_ui = GrantUI(adapter)
    return ui


async def handle_grants_command(adapter, message):
    text = message.text.strip()
    if not re.fullmatch(r"/grants(?:@[A-Za-z0-9_]+)?(?:\s.*)?", text, flags=re.DOTALL):
        return False
    await native_grants(adapter).command(message)
    return True
