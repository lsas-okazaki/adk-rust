//! Amazon Bedrock client implementation.
//!
//! Uses the AWS SDK Converse API for both streaming and non-streaming
//! inference. Credentials are loaded automatically from the environment
//! via `aws-config` (environment variables, shared config, IMDS, etc.).

use super::config::BedrockConfig;
use super::convert::{
    adk_request_to_bedrock, bedrock_response_to_adk, bedrock_stream_content_start_to_adk,
    bedrock_stream_delta_to_adk, bedrock_stream_stop_to_adk,
};
use crate::retry::RetryConfig;
use adk_core::{AdkError, Llm, LlmRequest, LlmResponse, LlmResponseStream};
use async_stream::try_stream;
use async_trait::async_trait;
use aws_sdk_bedrockruntime::types::ConverseStreamOutput;
use aws_smithy_types::error::display::DisplayErrorContext;
use tracing::{debug, info, instrument};

/// Amazon Bedrock client backed by the AWS SDK Converse API.
///
/// Supports both streaming (`converse_stream`) and non-streaming (`converse`)
/// inference. AWS credentials are loaded from the standard credential chain
/// (environment variables, `~/.aws/credentials`, IMDS, etc.).
///
/// # Example
///
/// ```rust,ignore
/// use adk_model::bedrock::{BedrockClient, BedrockConfig};
///
/// let config = BedrockConfig::new("us-east-1", "us.anthropic.claude-sonnet-4-6");
/// let client = BedrockClient::new(config).await?;
///
/// // Use via the Llm trait
/// let response = client.generate_content(request, false).await?;
/// ```
pub struct BedrockClient {
    client: aws_sdk_bedrockruntime::Client,
    model_id: String,
    region: String,
    retry_config: RetryConfig,
    prompt_caching: Option<super::config::BedrockCacheConfig>,
    /// Bound on each wait for the next stream event.
    stream_idle: std::time::Duration,
}

impl BedrockClient {
    /// Create a new Bedrock client from the given configuration.
    ///
    /// Loads AWS credentials from the standard credential chain
    /// (environment variables, shared config, IMDS, etc.) and constructs
    /// an `aws_sdk_bedrockruntime::Client`.
    ///
    /// # Errors
    ///
    /// Returns `AdkError::Model` if the AWS SDK configuration fails to load.
    pub async fn new(config: BedrockConfig) -> Result<Self, AdkError> {
        let region = config.region.clone();
        let model_id = config.model_id.clone();
        let prompt_caching = config.prompt_caching.clone();

        let mut sdk_config_loader = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_config::Region::new(config.region.clone()))
            .timeout_config(timeout_config(crate::HttpTimeouts::default()));

        if let Some(endpoint_url) = &config.endpoint_url {
            sdk_config_loader = sdk_config_loader.endpoint_url(endpoint_url);
        }

        let sdk_config = sdk_config_loader.load().await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);

        info!("bedrock client created for region={region}, model={model_id}");

        Ok(Self {
            client,
            model_id,
            region,
            retry_config: RetryConfig::default(),
            prompt_caching,
            stream_idle: crate::HttpTimeouts::default().idle,
        })
    }

    /// Use an explicit Bedrock bearer key without reading the process credential chain.
    ///
    /// # Errors
    ///
    /// This constructor does not fail today; the `Result` leaves room for key validation.
    ///
    /// # Example
    ///
    /// ```
    /// use adk_model::bedrock::{BedrockClient, BedrockConfig};
    ///
    /// let config = BedrockConfig::new("us-east-1", "anthropic.claude-sonnet-4-6-v1:0");
    /// let client = BedrockClient::new_with_api_key(config, "bedrock-api-key")?;
    /// # Ok::<(), adk_core::AdkError>(())
    /// ```
    pub fn new_with_api_key(config: BedrockConfig, api_key: &str) -> Result<Self, AdkError> {
        let mut sdk = aws_sdk_bedrockruntime::Config::builder()
            .behavior_version_latest()
            .region(aws_sdk_bedrockruntime::config::Region::new(config.region.clone()))
            .bearer_token(aws_sdk_bedrockruntime::config::Token::new(api_key, None))
            .timeout_config(timeout_config(crate::HttpTimeouts::default()));
        if let Some(endpoint) = &config.endpoint_url {
            sdk = sdk.endpoint_url(endpoint);
        }
        Ok(Self::from_client(config, aws_sdk_bedrockruntime::Client::from_conf(sdk.build())))
    }

    /// Supply a configured SDK client, preserving its identity and transport ownership.
    ///
    /// ADK retries are disabled because the supplied client carries its own retry
    /// configuration. The client keeps its own connect and read timeouts; the default idle
    /// timeout bounds each stream event, and `with_timeouts` sets all three.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use adk_model::bedrock::{BedrockClient, BedrockConfig};
    ///
    /// # async fn example() {
    /// let sdk_config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    /// let sdk = aws_sdk_bedrockruntime::Client::new(&sdk_config);
    /// let config = BedrockConfig::new("us-east-1", "anthropic.claude-sonnet-4-6-v1:0");
    /// let client = BedrockClient::from_client(config, sdk);
    /// # }
    /// ```
    pub fn from_client(config: BedrockConfig, client: aws_sdk_bedrockruntime::Client) -> Self {
        Self {
            client,
            model_id: config.model_id,
            region: config.region,
            retry_config: RetryConfig::disabled(),
            prompt_caching: config.prompt_caching,
            stream_idle: crate::HttpTimeouts::default().idle,
        }
    }

    /// Sets the connect timeout and the idle timeout, so a stalled stream fails with a
    /// `Timeout` error instead of holding the run.
    ///
    /// The idle timeout bounds the wait for the first response byte and for each stream
    /// event; see [`HttpTimeouts`](crate::HttpTimeouts) for the defaults. The SDK client is
    /// rebuilt with the connect bound, keeping the rest of its configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use adk_model::bedrock::{BedrockClient, BedrockConfig};
    /// use adk_model::HttpTimeouts;
    ///
    /// let config = BedrockConfig::new("us-east-1", "anthropic.claude-sonnet-4-6-v1:0");
    /// let client = BedrockClient::new_with_api_key(config, "bedrock-api-key")?
    ///     .with_timeouts(HttpTimeouts::new(Duration::from_secs(5), Duration::from_secs(120)))?;
    /// # let _ = client;
    /// # Ok::<(), adk_core::AdkError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// This method does not fail today; the `Result` matches the other clients'
    /// `with_timeouts`.
    pub fn with_timeouts(mut self, timeouts: crate::HttpTimeouts) -> Result<Self, AdkError> {
        self.client = aws_sdk_bedrockruntime::Client::from_conf(
            self.client.config().to_builder().timeout_config(timeout_config(timeouts)).build(),
        );
        self.stream_idle = timeouts.idle;
        Ok(self)
    }

    /// Set the retry configuration, consuming and returning `self`.
    #[must_use]
    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Set the retry configuration by mutable reference.
    pub fn set_retry_config(&mut self, retry_config: RetryConfig) {
        self.retry_config = retry_config;
    }

    /// Returns a reference to the current retry configuration.
    pub fn retry_config(&self) -> &RetryConfig {
        &self.retry_config
    }
}

#[async_trait]
impl Llm for BedrockClient {
    // Only a request with `max_output_tokens` is bounded: without one Bedrock applies the model's own limit, which the catalog does not carry for Claude.
    fn max_cost_micro_usd(&self, request: &adk_core::LlmRequest) -> Option<u64> {
        crate::pricing::PricingCatalog::standard().max_cost_micro_usd(
            Some("bedrock"),
            self.name(),
            request,
            crate::pricing::request_output_cap(request),
        )
    }

    fn name(&self) -> &str {
        &self.model_id
    }

    #[instrument(name = "model.generate_content", skip_all, fields(model_id = %self.model_id, region = %self.region, stream))]
    async fn generate_content(
        &self,
        mut request: LlmRequest,
        stream: bool,
    ) -> Result<LlmResponseStream, AdkError> {
        // Native items reach this provider only from the turns its own origin marks.
        adk_core::history::retain_replayable(
            &mut request.contents,
            &adk_core::history::ProviderOrigin::new("bedrock", &self.model_id),
        );
        {
            use crate::generation_config::{Supports, reject_unsupported};
            let supports = Supports {
                stop_sequences: true,
                seed: false,
                penalties: false,
                top_logprobs: false,
            };
            reject_unsupported("bedrock", request.config.as_ref(), supports)?;
        }
        // This client sends no reasoning configuration (spec 3.1 §7.23).
        if let Some(level) = request.config.as_ref().and_then(|config| config.effort) {
            tracing::debug!(
                model.name = %self.model_id,
                effort = %level,
                "bedrock takes no reasoning effort from this client; the request's effort is ignored"
            );
        }
        let usage_span = adk_telemetry::llm_generate_span("bedrock", &self.model_id, stream);
        let bedrock_input = adk_request_to_bedrock(
            &request.contents,
            &request.tools,
            request.config.as_ref(),
            self.prompt_caching.as_ref(),
        )
        .map_err(|e| {
            AdkError::model(format!(
                "Bedrock request conversion failed for region={}, model={}: {e}",
                self.region, self.model_id
            ))
        })?;

        if stream && request.tools.is_empty() {
            let result = self.generate_streaming(bedrock_input).await?;
            Ok(crate::usage_tracking::with_priced_usage_tracking(
                result,
                usage_span,
                "bedrock",
                &self.model_id,
            ))
        } else {
            let result = self.generate_non_streaming(bedrock_input).await?;
            Ok(crate::usage_tracking::with_priced_usage_tracking(
                result,
                usage_span,
                "bedrock",
                &self.model_id,
            ))
        }
    }
}

/// The SDK timeouts for `timeouts`: `connect`, and `idle` as the wait for the response.
fn timeout_config(timeouts: crate::HttpTimeouts) -> aws_smithy_types::timeout::TimeoutConfig {
    aws_smithy_types::timeout::TimeoutConfig::builder()
        .connect_timeout(timeouts.connect)
        .read_timeout(timeouts.idle)
        .build()
}

/// Render an `SdkError` with its whole source chain.
///
/// `SdkError`'s own `Display` is only a category — a rejected request prints
/// the bare string `service error`, and the sentence that says what is wrong
/// (`ValidationException: ...`) lives one or more levels down in `source()`.
/// [`DisplayErrorContext`] is the AWS SDK's own wrapper for walking that chain.
fn describe<E>(err: &E) -> String
where
    E: std::error::Error + 'static,
{
    DisplayErrorContext(err).to_string()
}

/// Phrases Bedrock-hosted models use when a request exceeds the context
/// window, matched lowercase. OpenAI-compatible models (gpt-oss) say
/// "Input length (N) exceeds model's maximum context length (M)"; Anthropic
/// models say "prompt is too long: N tokens > M maximum" or "Input is too long
/// for requested model".
const CONTEXT_OVERFLOW_PHRASES: &[&str] = &[
    "maximum context length",
    "context length",
    "context window",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "input length",
];

/// True when a rendered Converse error is a `ValidationException` saying the
/// request is larger than the model's context window.
///
/// Gated on `ValidationException` so a throttling or transport error that
/// happens to mention tokens is never mistaken for an oversized request.
fn is_context_overflow(detail: &str) -> bool {
    let lower = detail.to_lowercase();
    lower.contains("validationexception")
        && CONTEXT_OVERFLOW_PHRASES.iter().any(|phrase| lower.contains(phrase))
}

/// Map a failed Converse call (or stream) to an [`AdkError`].
///
/// A context overflow is `InvalidInput`, which is what the runner's
/// `is_token_limit_error` requires before it compacts the history and retries.
/// Everything else stays the legacy `Internal` model error. Without the
/// distinction an oversized request was resent unchanged until the session
/// died, because no Bedrock error ever qualified for reactive compaction.
fn converse_error(what: &str, region: &str, model_id: &str, detail: String) -> AdkError {
    let message = format!("Bedrock {what} error for region={region}, model={model_id}: {detail}");
    if is_context_overflow(&detail) {
        AdkError::new(
            adk_core::ErrorComponent::Model,
            adk_core::ErrorCategory::InvalidInput,
            "model.bedrock.context_overflow",
            message,
        )
        .with_provider("bedrock")
    } else {
        AdkError::model(message)
    }
}

/// A failed Converse call, as `Timeout` when the connect or read timeout expired.
fn send_error<E, R>(
    error: &aws_sdk_bedrockruntime::error::SdkError<E, R>,
    region: &str,
    model_id: &str,
) -> AdkError
where
    aws_sdk_bedrockruntime::error::SdkError<E, R>: std::error::Error + 'static,
{
    use aws_sdk_bedrockruntime::error::SdkError;

    let detail = describe(error);
    let timed_out = matches!(error, SdkError::TimeoutError(_))
        || matches!(error, SdkError::DispatchFailure(failure) if failure.is_timeout());
    if timed_out {
        AdkError::new(
            adk_core::ErrorComponent::Model,
            adk_core::ErrorCategory::Timeout,
            "model.bedrock.timeout",
            format!("Bedrock API error for region={region}, model={model_id}: {detail}"),
        )
        .with_provider("bedrock")
    } else {
        converse_error("API", region, model_id, detail)
    }
}

/// One-line summary of an assembled Converse message list: role, block kinds,
/// and the ids that pair `toolUse` with `toolResult`.
///
/// Converse rejects a request on the *shape* of `messages` — an unpaired
/// `toolUse` id, a blank text block — and the rejection names a message index
/// (`messages.2: ...`) that means nothing without the shape to index into.
/// Logging kinds and ids rather than content keeps prompts and tool payloads
/// out of the log while still making those errors readable.
fn describe_messages(messages: &[aws_sdk_bedrockruntime::types::Message]) -> String {
    use aws_sdk_bedrockruntime::types::ContentBlock;

    messages
        .iter()
        .enumerate()
        .map(|(i, message)| {
            let blocks = message
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::Text(text) => {
                        if text.trim().is_empty() {
                            "text(BLANK)".to_string()
                        } else {
                            format!("text({})", text.len())
                        }
                    }
                    ContentBlock::ToolUse(tool_use) => {
                        format!("toolUse({}#{})", tool_use.name, tool_use.tool_use_id)
                    }
                    ContentBlock::ToolResult(result) => {
                        format!("toolResult(#{})", result.tool_use_id)
                    }
                    ContentBlock::Image(_) => "image".to_string(),
                    ContentBlock::Document(_) => "document".to_string(),
                    ContentBlock::ReasoningContent(_) => "reasoning".to_string(),
                    ContentBlock::CachePoint(_) => "cachePoint".to_string(),
                    _ => "other".to_string(),
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("[{i}]{:?}{{{blocks}}}", message.role)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl BedrockClient {
    /// Non-streaming: call `converse` and wrap the single response in a stream.
    async fn generate_non_streaming(
        &self,
        input: super::convert::BedrockConverseInput,
    ) -> Result<LlmResponseStream, AdkError> {
        if tracing::enabled!(tracing::Level::DEBUG) {
            debug!(
                model = %self.model_id,
                messages = %describe_messages(&input.messages),
                system_blocks = input.system.len(),
                "bedrock non-streaming converse"
            );
        }

        let response = self
            .client
            .converse()
            .model_id(&self.model_id)
            .set_messages(Some(input.messages))
            .set_system(Some(input.system))
            .set_inference_config(input.inference_config)
            .set_tool_config(input.tool_config)
            .send()
            .await
            .map_err(|e| send_error(&e, &self.region, &self.model_id))?;

        let output = response.output.ok_or_else(|| {
            AdkError::model(format!("Bedrock response missing output for model={}", self.model_id))
        })?;

        let adk_response =
            bedrock_response_to_adk(&output, &response.stop_reason, response.usage.as_ref());

        let response_stream = try_stream! {
            yield adk_response;
        };

        Ok(Box::pin(response_stream))
    }

    /// Streaming: call `converse_stream` and convert each event to an `LlmResponse`.
    async fn generate_streaming(
        &self,
        input: super::convert::BedrockConverseInput,
    ) -> Result<LlmResponseStream, AdkError> {
        if tracing::enabled!(tracing::Level::DEBUG) {
            debug!(
                model = %self.model_id,
                messages = %describe_messages(&input.messages),
                system_blocks = input.system.len(),
                "bedrock streaming converse"
            );
        }

        let mut stream_output = self
            .client
            .converse_stream()
            .model_id(&self.model_id)
            .set_messages(Some(input.messages))
            .set_system(Some(input.system))
            .set_inference_config(input.inference_config)
            .set_tool_config(input.tool_config)
            .send()
            .await
            .map_err(|e| send_error(&e, &self.region, &self.model_id))?;

        let model_id = self.model_id.clone();
        let region = self.region.clone();
        let stream_idle = self.stream_idle;

        let response_stream = try_stream! {
            // Track tool use state for accumulating partial JSON arguments.
            let mut tool_name: Option<String> = None;
            let mut tool_id: Option<String> = None;
            let mut tool_args_buf = String::new();
            // Track reasoning signature for the current reasoning content block.
            let mut reasoning_signature: Option<String> = None;
            // Buffer the stop response and usage metadata so they can be merged.
            // Bedrock sends Metadata after MessageStop, so we defer emitting
            // the final chunk until the stream ends.
            let mut pending_stop: Option<LlmResponse> = None;
            let mut pending_usage: Option<adk_core::UsageMetadata> = None;

            loop {
                let Ok(next) = tokio::time::timeout(stream_idle, stream_output.stream.recv()).await
                else {
                    Err(AdkError::new(
                        adk_core::ErrorComponent::Model,
                        adk_core::ErrorCategory::Timeout,
                        "model.bedrock.stream_idle_timeout",
                        format!(
                            "Bedrock stream for region={region}, model={model_id} sent no event within {stream_idle:?}"
                        ),
                    )
                    .with_provider("bedrock"))?;
                    break;
                };
                let Some(event) = next
                    .map_err(|e| converse_error("stream", &region, &model_id, describe(&e)))?
                else {
                    break;
                };
                match event {
                    ConverseStreamOutput::ContentBlockStart(start_event) => {
                        if let Some(start) = &start_event.start {
                            // If starting a tool use block, track the name and ID.
                            if let aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(
                                tool_start,
                            ) = start
                            {
                                tool_name = Some(tool_start.name.clone());
                                tool_id = Some(tool_start.tool_use_id.clone());
                                tool_args_buf.clear();
                            }

                            if let Some(response) = bedrock_stream_content_start_to_adk(start) {
                                yield response;
                            }
                        }
                    }
                    ConverseStreamOutput::ContentBlockDelta(delta_event) => {
                        if let Some(delta) = &delta_event.delta {
                            // Accumulate tool use argument deltas.
                            if let aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(
                                tool_delta,
                            ) = delta
                            {
                                tool_args_buf.push_str(&tool_delta.input);
                            }

                            // Capture reasoning signature deltas for later attachment.
                            if let aws_sdk_bedrockruntime::types::ContentBlockDelta::ReasoningContent(
                                reasoning_delta,
                            ) = delta
                                && let Ok(sig) = reasoning_delta.as_signature() {
                                    reasoning_signature = Some(sig.clone());
                                }

                            if let Some(response) = bedrock_stream_delta_to_adk(delta) {
                                yield response;
                            }
                        }
                    }
                    ConverseStreamOutput::ContentBlockStop(_) => {
                        // If we were accumulating tool arguments, emit the complete FunctionCall.
                        if let (Some(name), Some(id)) = (tool_name.take(), tool_id.take()) {
                            // A zero-argument tool streams no input, which decodes to `{}`.
                            let args = crate::tool_args::streamed_tool_arguments(
                                "bedrock",
                                &name,
                                &tool_args_buf,
                            );
                            tool_args_buf.clear();

                            yield LlmResponse {
                                content: Some(adk_core::Content {
                                    role: "model".to_string(),
                                    parts: vec![adk_core::Part::FunctionCall {
                                        name,
                                        args,
                                        id: Some(id),
                                        thought_signature: None,
                                    }],
                                }),
                                usage_metadata: None,
                                finish_reason: None,
                                citation_metadata: None,
                                partial: false,
                                turn_complete: false,
                                interrupted: false,
                                error_code: None,
                                error_message: None,
                                provider_metadata: None,
                                interaction_id: None,
                                model: None,
                                provider: None,
                            };
                        }

                        // If we accumulated a reasoning signature, emit it as a
                        // Part::Thinking with the signature so downstream consumers
                        // can attach it to the reasoning block.
                        if let Some(sig) = reasoning_signature.take() {
                            yield LlmResponse {
                                content: Some(adk_core::Content {
                                    role: "model".to_string(),
                                    parts: vec![adk_core::Part::Thinking {
                                        thinking: String::new(),
                                        signature: Some(sig),
                                    }],
                                }),
                                usage_metadata: None,
                                finish_reason: None,
                                citation_metadata: None,
                                partial: true,
                                turn_complete: false,
                                interrupted: false,
                                error_code: None,
                                error_message: None,
                                provider_metadata: None,
                                interaction_id: None,
                                model: None,
                                provider: None,
                            };
                        }
                    }
                    ConverseStreamOutput::MessageStart(_) => {
                        // MessageStart carries the role; no content to emit.
                    }
                    ConverseStreamOutput::MessageStop(stop_event) => {
                        // Buffer the stop chunk — Metadata arrives after this.
                        pending_stop = Some(bedrock_stream_stop_to_adk(&stop_event.stop_reason));
                    }
                    ConverseStreamOutput::Metadata(metadata_event) => {
                        if let Some(usage) = &metadata_event.usage {
                            pending_usage = Some(adk_core::UsageMetadata {
                                provider_usage: Some(serde_json::json!({
                                    "inputTokens":usage.input_tokens, "outputTokens":usage.output_tokens,
                                    "totalTokens":usage.total_tokens, "cacheReadInputTokens":usage.cache_read_input_tokens,
                                    "cacheWriteInputTokens":usage.cache_write_input_tokens,
                                })),
                                // Converse reports uncached input separately from cache reads and writes.
                                prompt_token_count: usage
                                    .input_tokens
                                    .saturating_add(usage.cache_read_input_tokens.unwrap_or(0))
                                    .saturating_add(usage.cache_write_input_tokens.unwrap_or(0)),
                                candidates_token_count: usage.output_tokens,
                                total_token_count: usage.total_tokens,
                                cache_read_input_token_count: usage.cache_read_input_tokens,
                                cache_creation_input_token_count: usage.cache_write_input_tokens,
                                ..Default::default()
                            });
                        }
                    }
                    _ => {
                        // Unknown event variant — skip.
                    }
                }
            }

            // Emit the final stop chunk with usage metadata merged in.
            if let Some(mut stop) = pending_stop {
                stop.usage_metadata = pending_usage.take();
                yield stop;
            } else {
                Err(AdkError::model("incomplete Bedrock event stream"))?;
            }
        };

        Ok(Box::pin(response_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn sdk_timeouts(client: &BedrockClient) -> (Option<Duration>, Option<Duration>) {
        let config = client.client.config().timeout_config();
        config.map_or((None, None), |config| (config.connect_timeout(), config.read_timeout()))
    }

    /// Verbatim detail of the gpt-oss-120b rejection that killed the GEO
    /// station-keeping NL run on 2026-09-28.
    const GPT_OSS_OVERFLOW: &str = "service error: ValidationException: The model returned the \
        following errors: {\"error\":{\"code\":\"validation_error\",\"message\":\"ErrorEvent { \
        error: APIError { type: \\\"BadRequestError\\\", code: Some(400), message: \\\"Input \
        length (135735) exceeds model's maximum context length (131072).\\\", param: None } }\"}}";

    #[test]
    fn a_gpt_oss_context_overflow_is_invalid_input() {
        let err = converse_error(
            "API",
            "ap-northeast-1",
            "openai.gpt-oss-120b-1:0",
            GPT_OSS_OVERFLOW.into(),
        );
        assert_eq!(err.component, adk_core::ErrorComponent::Model);
        assert_eq!(err.category, adk_core::ErrorCategory::InvalidInput);
        assert_eq!(err.code, "model.bedrock.context_overflow");
        assert!(err.message.contains("Input length (135735)"));
    }

    #[test]
    fn a_claude_prompt_too_long_is_invalid_input() {
        for detail in [
            "service error: ValidationException: prompt is too long: 210000 tokens > 200000 maximum",
            "service error: ValidationException: Input is too long for requested model.",
        ] {
            let err = converse_error("API", "us-east-1", "anthropic.claude", detail.into());
            assert_eq!(err.category, adk_core::ErrorCategory::InvalidInput, "{detail}");
        }
    }

    #[test]
    fn other_validation_errors_stay_internal() {
        let detail = "service error: ValidationException: messages.2: toolUse ids must be paired";
        let err = converse_error("API", "us-east-1", "anthropic.claude", detail.into());
        assert_eq!(err.category, adk_core::ErrorCategory::Internal);
    }

    #[test]
    fn a_non_validation_error_mentioning_tokens_stays_internal() {
        let detail = "service error: ThrottlingException: Too many tokens, please wait";
        let err = converse_error("API", "us-east-1", "anthropic.claude", detail.into());
        assert_eq!(err.category, adk_core::ErrorCategory::Internal);
    }

    #[test]
    fn the_default_timeouts_bound_connecting_and_the_response() {
        let client =
            BedrockClient::new_with_api_key(BedrockConfig::new("us-east-1", "m"), "key").unwrap();
        let defaults = crate::HttpTimeouts::default();
        assert_eq!(sdk_timeouts(&client), (Some(defaults.connect), Some(defaults.idle)));

        let client = client
            .with_timeouts(crate::HttpTimeouts::new(Duration::from_secs(2), Duration::from_secs(9)))
            .unwrap();
        assert_eq!(
            sdk_timeouts(&client),
            (Some(Duration::from_secs(2)), Some(Duration::from_secs(9)))
        );
    }
}
