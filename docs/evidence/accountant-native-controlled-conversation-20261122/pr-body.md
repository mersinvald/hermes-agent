Controlled native adapters previously refused both complete conversation entrypoints and could not construct an inert real agent.
This change adds an optional keyword-only `controlled_main` constructor branch and one shared private implementation for both existing entrypoints.

The constructed agent uses accepted admission, the actual prologue/caller/dispatcher, and public shared derivation for complete native results.
It retains original response/result identities and earlier handles, preserves cancellation, and closes before generic cleanup.
Existing public adapter signatures, `NativeMainInvocation`, generic guards, manifests, and accepted tests remain unchanged.

Validation: 100 focused cases passed in 49.3 seconds with zero retries and blocked network callbacks.
They include 90 real-construction integration cases and ten existing compatibility cases.
The exact installed wheel and unchanged engine 0.1.0 pass one API/import smoke.
No new F/E9 diagnostics appear, and STE scores do not increase.

Source/API freeze: `62d5eaf0033343eb66b749360274bb5246d6d1da`.
Evidence: `docs/evidence/accountant-native-controlled-conversation-20261122/`.
Earlier failed and overlapping scopes remain documented.
The coordinator retains acceptance and merge ownership.

Full P10/P06, real providers, egress, durable restoration, private processing, source/financial qualification, and deployment remain open.
No CI success claim follows from zero configured checks.
