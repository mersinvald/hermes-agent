def _nested_anchors(value):
    result = []
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
    visits = 0
    record_index = None
    def visit(item):
        nonlocal visits, record_index
        result.append(item)
        if type(item) is dict:
            for key, child in item.items():
                _require(type(key) is str)
                visit(key)
                visit(child)
        elif type(item) in (list, tuple):
            for child in item:
                visit(child)
        else:
            item_type = type(item)
            if record_index is not None and type(item_type) is type:
                recognized = item_type in record_index
            else:
                recognized = item_type in records
            visits += 1
            if visits == 8:
                for record in records:
                    if type(record) is not type:
                        break
                else:
                    record_index = frozenset(records)
            if recognized:
                for field in fields(item):
                    visit(object.__getattribute__(item, field.name))
    visit(value)
    return tuple(result)
