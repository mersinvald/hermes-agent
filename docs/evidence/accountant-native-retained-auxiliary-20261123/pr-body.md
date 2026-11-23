A retained native completion can now supply the synchronous title consumer under an independent operator title selection.
The actual `generate_title()` and decorated `call_llm()` use the existing adapter, shared runtime, and one retained dispatcher.

The adapter keeps complete source, transformation, request, response, and output records with original parent capabilities and object identities.
Main and auxiliary sends remain separate, including failed sends.
The title model cannot replace the main model or its captured selection.

The first supported pair is `title` / `title_generation` with invented `synthetic_native` nonstreaming `chat_completions`.
Empty snippets refuse before dispatch.
A semantic `None` after a response requires complete output proof.
Other consumers remain unsupported.

Validation passed 198 focused cases in 538.6 runner seconds, with zero failures and retries.
Exact pins are in `docs/evidence/accountant-native-retained-auxiliary-20261123/`.
The tests use the accepted local 0.2.0 wheel, cached locked dependencies, blocked network callbacks, and zero retries.
The evidence preserves failed and overlapping intermediate runs.
Root owns final acceptance and merge.
Full P10/P06 and production qualification remain open.
