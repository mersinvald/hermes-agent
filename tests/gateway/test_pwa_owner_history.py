"""Actual private HTTP/native SQLite owner-history sweep behavior."""

import asyncio
import json

import pytest

from tests.gateway.test_pwa_http import native_row, service
from tests.gateway.test_native_commands import JournalAgent, command, started
from tests.gateway.test_native_events import provider_loop

pytestmark = pytest.mark.asyncio


async def sweep(client, kind, *, limit=2, **query):
    pages, cursor = [], None
    for _ in range(1000):
        params = {"limit": str(limit), **query}
        if cursor:
            params["cursor"] = cursor
        response = await client.get("/v1/pwa/" + kind, params=params)
        assert response.status == 200, await response.text()
        page = await response.json()
        pages.append(page)
        total = sum(
            len(group["matches" if kind == "search" else "messages"])
            + (int(group["include_title"]) if kind == "search" else 0)
            for group in page["groups"]
        )
        assert total <= limit and len(page["groups"]) <= 32
        cursor = page["next_cursor"]
        if not cursor:
            assert page["checkpoint"] and page["coverage"]["state"] in {
                "complete",
                "partial",
            }
            return pages
    pytest.fail("history sweep did not terminate")


async def test_real_native_retained_compression_branches_and_foreign_exclusion(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, root, source = native[2], native[4].session_id, native[5]
        db.append_message(root, "user", "Retained Straße needle", timestamp=1)
        db.append_message(root, "system", "SYSTEM needle")
        db.append_message(root, "session_meta", "METADATA needle")
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET active=0,compacted=1 WHERE session_id=? AND role='user'",
                (root,),
            )
        )
        db.publish_compression_child(
            parent_session_id=root,
            child_session_id="tip",
            source=source.platform.value,
            messages=[{"role": "user", "content": "compressed handoff needle"}],
            require_compression_lease=False,
        )
        native_row(
            db,
            source,
            "branch",
            parent_session_id=root,
            model_config={"_branched_from": root},
        )
        db.append_message("branch", "user", "branch needle")
        native_row(
            db,
            source,
            "delegate",
            parent_session_id=root,
            model_config={"_delegate_from": root},
        )
        db.append_message("delegate", "user", "DELEGATE needle")
        db.create_session(
            "foreign",
            source.platform.value,
            user_id="foreign",
            chat_id=source.chat_id,
            chat_type=source.chat_type,
        )
        db.append_message("foreign", "user", "FOREIGN needle")
        pages = await sweep(client, "search", q="needle", limit=2)
        hits = [
            (
                group["conversation"]["conversation_id"],
                hit["native_session_id"],
                hit["snippet"]["text"],
            )
            for page in pages
            for group in page["groups"]
            for hit in group["matches"]
        ]
        assert {(r, segment) for r, segment, _ in hits} == {
            (root, root),
            (root, "tip"),
            ("branch", "branch"),
        }
        assert any("Retained Straße" in text for _, _, text in hits)
        encoded = json.dumps(pages)
        assert all(
            secret not in encoded
            for secret in (
                "SYSTEM needle",
                "METADATA needle",
                "DELEGATE needle",
                "FOREIGN needle",
            )
        )
        assert pages[-1]["coverage"]["state"] == "complete"
        synced = await sweep(client, "sync", limit=1)
        messages = [
            message
            for page in synced
            for group in page["groups"]
            for message in group["messages"]
        ]
        assert len(messages) == 3 and len({row["message_id"] for row in messages}) == 3


async def test_fixed_append_cutoff_retry_checkpoint_and_equal_size_rewrite(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, root = native[2], native[4].session_id
        for index in range(110):
            db.append_message(root, "user", f"retained row {index:03}")
        params = {"limit": "10", "conversation_id": root}
        first = await (await client.get("/v1/pwa/sync", params=params)).json()
        cursor = first["next_cursor"]
        db.append_message(root, "user", "new append outside cutoff")
        second = await (
            await client.get("/v1/pwa/sync", params={**params, "cursor": cursor})
        ).json()
        repeated = await (
            await client.get("/v1/pwa/sync", params={**params, "cursor": cursor})
        ).json()
        assert second == repeated
        pages = [first, second]
        while pages[-1]["next_cursor"]:
            response = await client.get(
                "/v1/pwa/sync", params={**params, "cursor": pages[-1]["next_cursor"]}
            )
            assert response.status == 200, await response.text()
            pages.append(await response.json())
            assert len(pages) <= 20
        messages = [m for p in pages for g in p["groups"] for m in g["messages"]]
        assert len(messages) == 110
        assert "outside cutoff" not in json.dumps(pages)
        probe = await (
            await client.get(
                "/v1/pwa/sync", params={"checkpoint": pages[-1]["checkpoint"]}
            )
        ).json()
        assert probe["state"] == "refresh_required"
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET content='modified row 050' WHERE content='retained row 050'"
            )
        )
        assert (
            await client.get("/v1/pwa/sync", params={**params, "cursor": cursor})
        ).status == 409


async def test_partial_legacy_and_omitted_content_coverage_survives_empty_pages(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, root, source = native[2], native[4].session_id, native[5]
        db.create_session("ambiguous", source.platform.value)
        db.append_message(root, "user", [{"type": "image", "url": "synthetic-only"}])
        pages = await sweep(client, "search", q="not present", limit=1)
        assert all(not g["matches"] for page in pages for g in page["groups"])
        assert pages[-1]["coverage"] == {
            "state": "partial",
            "scope": "retained_display_text",
            "reasons": ["legacy_ownership_unproven", "unsupported_content"],
        }
        synced = await sweep(client, "sync", limit=1)
        omitted = [m for page in synced for g in page["groups"] for m in g["messages"]]
        assert len(omitted) == 1 and omitted[0]["content_state"] == "omitted"
        assert synced[-1]["coverage"]["state"] == "partial"


async def test_structured_nul_prefix_cannot_bypass_sql_history_byte_limits(monkeypatch, tmp_path):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, root = native[2], native[4].session_id
        db.append_message(root, "user", [{"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + "A" * 1000000}}],
            display_metadata={"private": "📷" * 16385})
        with db._read_ctx() as conn:
            stored = conn.execute("SELECT id,length(content),length(CAST(content AS BLOB)) FROM messages WHERE session_id=?",
                                  (root,)).fetchone()
        assert stored[1] == 0 and stored[2] > 1000000
        row = db.native_pwa_history_rows(root, 0, stored[0], 1)[0]
        assert row["content"] is None and row["content_length"] == stored[2]
        assert row["display_metadata"] is None and row["image_display"] is None
        assert len(json.dumps(row).encode()) < 1024
        response = await client.get(f"/v1/pwa/conversations/{root}/history")
        assert response.status == 200, await response.text()
        history = await response.json()
        pages = await sweep(client, "sync", limit=1)
        messages = history["messages"] + [m for p in pages for g in p["groups"] for m in g["messages"]]
        assert len(messages) == 2
        assert all(m["content"] is None and m["omission_reason"] == "oversized" for m in messages)
        assert "data:image" not in json.dumps([history, pages])


async def wait_idle(ingress, root):
    for _ in range(1000):
        if ingress._owner(root) is None:
            return
        await asyncio.sleep(0.01)
    pytest.fail("synthetic native execution did not finish")


async def test_actual_incremental_writer_start_and_completion_do_not_starve_captured_sweep(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (server, client, native):
        monkeypatch.setattr("agent.conversation_loop.run_conversation", provider_loop)
        db, root = native[2], native[4].session_id
        # Establish any native origin/model metadata through the real path.
        assert (
            await client.post(
                "/v1/pwa/commands", json=command(root, text="warmup", id="warmup")
            )
        ).status == 200
        await started()
        JournalAgent.gates[0].set()
        await wait_idle(server.ingress, root)
        for index in range(25):
            db.append_message(root, "user", f"retained {index}")
        params = {"limit": "5", "conversation_id": root}
        first = await (await client.get("/v1/pwa/sync", params=params)).json()
        assert first["next_cursor"]
        assert (
            await client.post(
                "/v1/pwa/commands",
                json=command(root, text="new running input", id="during-sweep"),
            )
        ).status == 200
        await started()
        response = await client.get(
            "/v1/pwa/sync", params={**params, "cursor": first["next_cursor"]}
        )
        assert response.status == 200, await response.text()
        second = await response.json()
        JournalAgent.gates[1].set()
        await wait_idle(server.ingress, root)
        pages = [first, second]
        for _ in range(20):
            if not pages[-1]["next_cursor"]:
                break
            response = await client.get(
                "/v1/pwa/sync", params={**params, "cursor": pages[-1]["next_cursor"]}
            )
            assert response.status == 200, await response.text()
            pages.append(await response.json())
        assert pages[-1]["next_cursor"] is None
        content = [
            m["content"] for p in pages for g in p["groups"] for m in g["messages"]
        ]
        assert content.count("warmup") == 1 and content.count("synthetic done") == 1
        assert "new running input" not in content
        assert len([text for text in content if text.startswith("retained ")]) == 25


async def test_empty_candidate_pages_group_bound_and_foreign_history(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        db, source = native[2], native[5]
        # More than one candidate budget, no public counts or foreign payloads.
        for index in range(505):
            db.create_session(
                f"000-foreign-{index:03}",
                source.platform.value,
                user_id="foreign",
                chat_id=source.chat_id,
                chat_type=source.chat_type,
            )
        for index in range(40):
            native_row(db, source, f"z-owned-{index:03}")
            db.append_message(f"z-owned-{index:03}", "user", "visible needle")
        pages = await sweep(client, "sync", limit=100)
        assert pages[0]["groups"] == [] and pages[0]["next_cursor"]
        assert any(len(page["groups"]) == 32 for page in pages)
        assert sum(len(g["messages"]) for p in pages for g in p["groups"]) == 40
        assert pages[-1]["coverage"]["state"] == "complete"
        assert "000-foreign" not in json.dumps(pages)
        search = await sweep(client, "search", limit=50, q="needle")
        assert sum(len(g["matches"]) for p in search for g in p["groups"]) == 40


async def test_cursor_query_restart_foreign_principal_and_current_source_revocation(
    monkeypatch, tmp_path
):
    from gateway.conversation_control import Principal
    from tests.gateway.test_pwa_http import headers

    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        for index in range(3):
            native[2].append_message(root, "user", f"needle {index}")
        params = {"limit": "1", "conversation_id": root, "q": "needle"}
        first = await (await client.get("/v1/pwa/search", params=params)).json()
        query = {**params, "cursor": first["next_cursor"]}
        assert (
            await client.get("/v1/pwa/search", params={**query, "q": "different"})
        ).status == 409
        foreign = await client.get(
            "/v1/pwa/search",
            params=query,
            headers=headers(Principal("https://issuer.invalid", "foreign")),
        )
        assert foreign.status == 403 and "needle" not in await foreign.text()
        server.history_scan.handles.close()
        assert (await client.get("/v1/pwa/search", params=query)).status == 409
        original = server.ingress.native_conversation

        async def revoked(*args):
            result = await original(*args)
            server.runner._is_user_authorized_for_source = lambda source: False
            return result

        monkeypatch.setattr(server.ingress, "native_conversation", revoked)
        response = await client.get("/v1/pwa/search", params=params)
        assert response.status == 404 and "needle" not in await response.text()


async def test_database_failure_never_becomes_empty_complete_page(
    monkeypatch, tmp_path
):
    async with service(monkeypatch, tmp_path) as (_, client, native):
        root = native[4].session_id
        native[2].append_message(root, "user", "private needle")

        def unavailable(*args):
            raise RuntimeError("synthetic PRIVATE SQL failure")

        monkeypatch.setattr(native[2], "native_pwa_history_rows", unavailable)
        response = await client.get("/v1/pwa/sync", params={"conversation_id": root})
        assert response.status == 503
        assert "PRIVATE" not in await response.text()


async def test_byte_budget_omits_large_rows_and_never_skips_later_match(
    monkeypatch, tmp_path
):
    from dataclasses import replace

    async with service(monkeypatch, tmp_path) as (server, client, native):
        server.config = replace(server.config, max_response_bytes=32768)
        db, root = native[2], native[4].session_id
        db.append_message(root, "user", "large needle " + "x" * 20000)
        db.append_message(root, "user", "later needle")
        pages = await sweep(client, "sync", limit=10, conversation_id=root)
        messages = [m for p in pages for g in p["groups"] for m in g["messages"]]
        assert len(messages) == 2
        assert messages[0]["content_state"] == "omitted"
        assert messages[1]["content"] == "later needle"
        assert pages[-1]["coverage"]["state"] == "partial"
        assert pages[-1]["coverage"]["reasons"] == ["oversized_content"]
        for page in pages:
            assert len(json.dumps(page).encode()) < 32768
        matches = await sweep(
            client, "search", limit=2, conversation_id=root, q="needle"
        )
        assert sum(len(g["matches"]) for p in matches for g in p["groups"]) == 2


async def test_scan_redacts_before_matching_and_timeout_is_explicit(
    monkeypatch, tmp_path
):
    from dataclasses import replace
    from tests.gateway.test_pwa_http import TOKEN

    async with service(monkeypatch, tmp_path) as (server, client, native):
        root = native[4].session_id
        native[2].append_message(root, "user", "visible " + TOKEN)
        pages = await sweep(client, "search", q=TOKEN, conversation_id=root)
        assert all(not g["matches"] for p in pages for g in p["groups"])
        visible = await sweep(client, "search", q="visible", conversation_id=root)
        assert TOKEN not in json.dumps(visible)
        assert sum(len(g["matches"]) for p in visible for g in p["groups"]) == 1
        server.config = replace(server.config, request_timeout=1)

        async def stalled(*args):
            await asyncio.Event().wait()

        monkeypatch.setattr(server.ingress, "native_conversation", stalled)
        response = await client.get("/v1/pwa/sync", params={"conversation_id": root})
        assert response.status == 503
        assert "visible" not in await response.text()
