//! Request customization at the HTTP boundary.

/// Applies provider-specific JSON fields and headers before a generation request.
///
/// The adapter must preserve the requested response mode and must not perform I/O.
/// It runs for each HTTP attempt under the model's configured retry policy.
///
/// # Example
///
/// ```
/// use adk_model::openai::{OpenAICompatible, OpenAICompatibleConfig, RequestAdapter};
/// use std::sync::Arc;
///
/// let adapter: RequestAdapter = Arc::new(|body, headers| {
///     body["metadata"] = serde_json::json!({"application": "example"});
///     headers.insert("x-request-source", "example".parse().expect("valid header"));
///     Ok(())
/// });
/// let client = OpenAICompatible::new(OpenAICompatibleConfig::new("sk-key", "gpt-5-mini"))?
///     .with_request_adapter(adapter);
/// # Ok::<(), adk_core::AdkError>(())
/// ```
pub type RequestAdapter = std::sync::Arc<
    dyn Fn(
            &mut serde_json::Value,
            &mut reqwest::header::HeaderMap,
        ) -> Result<(), adk_core::AdkError>
        + Send
        + Sync,
>;
