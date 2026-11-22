# Native policy admission and refusal

The native component refuses sensitive processing before preprocessing or provider dispatch.
Runtime/test source: `3b4fe67e8e708532fbe78ee3a088d26353a64d11`.
Base: `089fadba6cfeacc5af9ebffcec1e6f72ae9eed28`.
Root approved the runtime scope and supplied the accepted standalone shared wheel.

Read [implementation.md](implementation.md) for actual native-function coverage, explicit refusals, and remaining acceptance gates.
[checks.json](checks.json) records final and historical validation scopes.
[runtime-api.json](runtime-api.json) records native adapter signatures and source hashes.

Read [design.md](design.md) for the concrete component and shared-package ownership proposal.
Read [source-map.md](source-map.md) for actual native call sites and current gaps.
[allowlist.json](allowlist.json) names 19 proposed native runtime files and eight new test files.
[validation-plan.md](validation-plan.md) separates native-function checks from controlled service simulations.
[source-pins.json](source-pins.json) records 40 exact tracked source objects.
[design-checks.json](design-checks.json) records design validation and retained tool failures.

The current shared contract accepts only synthetic qualifications.
The proposed component therefore denies sensitive real-provider dispatch before preprocessing.
Ordinary authenticated bindings require an explicit assignment when the operator enables policy.
Missing context cannot authorize a policy-enabled execution.
The accepted package handoff uses a hashed wheel in the task-owned validation environment.
Native dependency manifests remain unchanged.

P10.1 results retain their recorded sources and scopes.
The five PostgreSQL-dependent facade skips remain unverified.
The initial design stage ran no runtime tests.
Implementation tests use invented content and fake provider ports.
No real model, production service, egress probe, or deployment action ran.
Serialized authentication, durable restoration, private-base differences, actual egress, full P10, and financial activation remain separate gates.
