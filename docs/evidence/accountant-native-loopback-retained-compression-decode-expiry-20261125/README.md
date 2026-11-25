# Selected compression expiry after native decode

Outcome: FAILED. The two cheap controls passed. The sole genuine test failed after an unrecorded native denial.
The test tried to read an absent trigger receipt at line 498. This raised `KeyError` and masked the native exception details.

The frozen hook requires shared `issued.route` identity with an operator input route at line 184.
The shared constructor creates separate route objects at lines 383-386. Its dispatch uses those objects at line 696.
The inert control used the operator input object. It did not model this original shared behavior.

This is an invalid test assumption. No evidence established a missing runtime callback or assignment.

The source commit is `8e5f85c78413094e49dbd27afb5d382e7e01a0d2`.
No executable edit or functional command followed the genuine failure. A separate Root decision must select any correction.
The nine phase rows contain eight passes and one failed call. They do not establish three successful tests.

Partial receipts contain qualification charges of 50 contents, 25 envelopes, and 7241 bytes.
Native partial receipts contain 16 contents, two envelopes, and 30099 bytes. Main usage is 3/2/5.
Partial receipts record four complete HTTP pairs and clean receiver closure. The final quiet marks are absent.
Typed retention, expiry, scope refusal, double close, and terminal refusal remain UNKNOWN.

All 12294 original blobs and modes remain exact. The five environment files, two pytest files, protected receipts, and shared wheel remain exact.
Raw receipts, graphs, XML, JSONL, logs, socket metadata, and scripts remain in a private directory outside Git.
The private directory used mode 0700 at creation. Files used umask077 and mode 0600.

The first `pwd` and search preceded the launch receipt read. The exact initial timestamp is unknown.
Both ledgers charge that interval from the supplied Root epoch. The Task ledger also charges 60 preparation seconds.

Root owns acceptance, merge, tracker writes, and closure. Full P10, P06, and every preserved external qualification gate remain open.

[Draft PR 40](https://github.com/mersinvald/hermes-agent/pull/40) retains this failed attempt against `mkl`. It is not ready for acceptance.
