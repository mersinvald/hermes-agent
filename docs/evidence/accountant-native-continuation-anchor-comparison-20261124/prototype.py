def _continuation_matches_anchors(roots, original, *, skip_temporary_root=False):
    policy = shared_policy()
    records = (NativeMainResponse, NativeMainChoice, NativeMainMessage, NativeMainUsage,
               NativeMainSelection, NativeResultPolicy, policy.ExecutionBinding,
               policy.Destination, policy.DispatchEnvelope, policy.RouteBinding,
               NativeAuxiliaryInvocation, NativeAuxiliaryLocalResult, _AuxiliaryTitlePreparation,
               _CompressionBinding, _AuxiliaryBinding, _CompressionPreparation,
               _FallbackBinding, _FailedSendObservation, NativeFailedMainAttempt,
               _FallbackFrame, NativeFallbackInvocation, _TurnFrame, _PendingOperation,
               NativeDelegateBinding, NativeDelegateInvocation,
               policy.ProfilePolicy, policy.DispatchFailureFacts)
    anchors = iter(original)
    end = object()
    matched = True

    def visit(item):
        nonlocal matched
        if item is not next(anchors, end):
            matched = False
        if type(item) is dict:
            for key, child in item.items():
                _require(type(key) is str)
                visit(key)
                visit(child)
        elif type(item) in (list, tuple):
            for child in item:
                visit(child)
        elif type(item) in records:
            for field in fields(item):
                visit(object.__getattribute__(item, field.name))

    for root in roots:
        if skip_temporary_root:
            for child in root:
                visit(child)
        else:
            visit(root)
    return next(anchors, end) is end and matched
