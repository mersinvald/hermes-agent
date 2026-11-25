# Compression expiry after shared result retention

The selected test passed on frozen source commit `ae6cae19f0734f2f31a917700a130e2131f6411a`.
The original native error occurred after shared response retention and before native decode.
Shared state stayed healthy and idle. Native state became invalid and denied.

The qualifier retained 50 contents, 25 envelopes, and 7241 bytes.
The native runtime retained 16 contents, two envelopes, and 30099 bytes.
The native total includes the original 312-byte shared compression response.
Four owned HTTP pairs completed. Both receiver windows closed with empty quiet suffixes.

The catalog control passed in `cheap-01`. The gate control failed in that scope because its inert runtime lacked `_envelopes`.
Root authorized that exact stand-in correction in message `msg_7c1491ddc346`.
The corrected gate control passed separately in `cheap-02`.
These scopes do not establish a fresh whole-file pass.
The failed scope retains its original source pin and private artifacts.

[Results](results.json) record the selected boundary and separate exception identities.
[Controls](controls.json) retain the successful phase rows and failed feedback.
[Costs](costs.json) separate actual retention charges from all-wall time.
[Preservation](preservation.json) records frozen inputs and original path checks.
[Checks](checks.json) records focused verification and its limits.
The [manifest](SHA256SUMS) covers the other eleven evidence files.

Private raw artifacts remain outside Git. Their hashes appear in [pins](pins.json).
The creation shell used umask 077 before directory and file creation.
The private directory used mode 0700. Private files used mode 0600.

Subsecond precision of the first workspace boundary is unknown. The ledger uses its conservative recorded UTC second.

Root owns acceptance, merge, Beads closure, and canonical reconciliation.
Full P10/P06 and preserved external qualification gates remain open.
This evidence does not qualify providers, private content, external egress, deployment, or activation.
