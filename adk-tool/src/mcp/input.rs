//! Adapt MRTR input to the handler that owns the running connection.

use rmcp::{
    RoleClient,
    model::{
        ClientResult, GetExtensions, GetMeta, InputRequest, InputRequests, InputResponses,
        NumberOrString, ServerRequest,
    },
    service::{RequestContext, RunningService, Service},
};

/// Answers MRTR input through the client's own handler.
///
/// Elicitation is always forwarded. Sampling and roots are forwarded only when
/// the client declared those capabilities in its handshake, so a server cannot
/// reach a handler the client never offered. The whole batch is checked before
/// any request is dispatched.
pub(super) async fn fulfill<S: Service<RoleClient>>(
    client: &RunningService<RoleClient, S>,
    requests: InputRequests,
) -> Result<InputResponses, String> {
    let capabilities = client.service().get_info().capabilities;
    for (key, input) in &requests {
        let undeclared = match input {
            InputRequest::Elicitation(_) => None,
            InputRequest::CreateMessage(_) => capabilities.sampling.is_none().then_some("sampling"),
            InputRequest::ListRoots(_) => capabilities.roots.is_none().then_some("roots"),
            _ => return Err(format!("input request '{key}' has an unsupported MCP input type")),
        };
        if let Some(capability) = undeclared {
            return Err(format!(
                "input request '{key}' asks for {capability}, which this MCP client did not \
                 declare; declare the capability in the client handler to allow it"
            ));
        }
    }
    let responses =
        futures::future::try_join_all(requests.into_iter().map(|(key, input)| async move {
            let mut request = match &input {
                InputRequest::Elicitation(request) => ServerRequest::ElicitRequest(request.clone()),
                InputRequest::CreateMessage(request) => {
                    ServerRequest::CreateMessageRequest(request.clone())
                }
                InputRequest::ListRoots(request) => {
                    ServerRequest::ListRootsRequest(request.clone())
                }
                _ => {
                    return Err(format!("input request '{key}' has an unsupported MCP input type"));
                }
            };
            let mut context = RequestContext::new(
                NumberOrString::String(key.clone().into()),
                client.peer().clone(),
            );
            std::mem::swap(&mut context.meta, request.get_meta_mut());
            std::mem::swap(&mut context.extensions, request.extensions_mut());
            let result = client
                .service()
                .handle_request(request, context)
                .await
                .map_err(|error| format!("input request '{key}' failed: {error}"))?;
            let value = match (input, result) {
                (InputRequest::Elicitation(_), ClientResult::ElicitResult(result)) => {
                    serde_json::to_value(result)
                }
                (InputRequest::CreateMessage(_), ClientResult::CreateMessageResult(result)) => {
                    serde_json::to_value(result)
                }
                (InputRequest::ListRoots(_), ClientResult::ListRootsResult(result)) => {
                    serde_json::to_value(result)
                }
                _ => {
                    return Err(format!(
                        "input request '{key}' received a response of the wrong type"
                    ));
                }
            }
            .map_err(|error| error.to_string())?;
            Ok::<_, String>((key, value))
        }))
        .await?;
    Ok(responses.into_iter().collect())
}
