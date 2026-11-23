# Proposed next title result

Status: proposal for root selection after source acceptance.

The current component refuses a title source that becomes an empty snippet.
It does not release a bare `None` before a response.

The next component can support a truthful local result for this case.
It must use the same adapter, independent title capability, source parents, and complete transformation records.
It must derive `{"title": null}` before release and retain an explicit no-send result.
It must not create a transport envelope or response object.
The auxiliary send counter must remain unchanged.

A test must call the actual synchronous `generate_title()` on the constructed agent.
It must prove current source and target bindings, inherited restrictions, complete bytes, and exact output identity.
The record format needs root review before this extension.

This proposal changes no runtime behavior.
Background, async, streaming, private, and other native consumers remain unsupported.
Full P10/P06, egress, restoration, sensitive cross-thread, source, financial, and production qualification remain open.
