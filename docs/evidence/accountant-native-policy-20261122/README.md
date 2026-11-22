# Native policy design gate

This evidence proposes native shared content-policy integration from source `089fadba6cfeacc5af9ebffcec1e6f72ae9eed28`.
Only this evidence directory changes in the initial stage.
Runtime implementation awaits the coordinator decision.

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
The package extraction requires a separate worktree and artifact decision from root.

P10.1 results retain their recorded sources and scopes.
The five PostgreSQL-dependent facade skips remain unverified.
This stage runs no runtime tests, model calls, network probes, or deployment actions.
Serialized authentication, durable restoration, private-base differences, actual egress, full P10, and financial activation remain separate gates.
