// MCP HTTP Transport (Streamable HTTP)
//
// Provides HTTP-based transport for connecting to remote MCP servers.
// Uses the streamable HTTP transport from rmcp when the http-transport feature is enabled.

use super::auth::McpAuth;
use super::elicitation::ElicitationHandler;
use adk_core::{AdkError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Builder for HTTP-based MCP connections.
///
/// This builder creates connections to remote MCP servers using the
/// streamable HTTP transport (SEP-1686 compliant).
///
/// # Example
///
/// ```rust,ignore
/// use adk_tool::mcp::{McpHttpClientBuilder, McpAuth, OAuth2Config};
///
/// // Simple connection
/// let toolset = McpHttpClientBuilder::new("https://mcp.example.com/v1")
///     .connect()
///     .await?;
///
/// // With OAuth2 authentication
/// let toolset = McpHttpClientBuilder::new("https://mcp.example.com/v1")
///     .with_auth(McpAuth::oauth2(
///         OAuth2Config::new("client-id", "https://auth.example.com/token")
///             .with_secret("client-secret")
///             .with_scopes(vec!["mcp:read".into()])
///     ))
///     .timeout(Duration::from_secs(60))
///     .connect()
///     .await?;
/// ```
#[derive(Clone)]
pub struct McpHttpClientBuilder {
    /// MCP server endpoint URL
    endpoint: String,
    /// Authentication configuration
    auth: McpAuth,
    /// Request timeout
    timeout: Duration,
    /// Custom headers
    headers: HashMap<String, String>,
    /// Optional elicitation handler
    elicitation_handler: Option<Arc<dyn ElicitationHandler>>,
}

impl McpHttpClientBuilder {
    /// Create a new HTTP client builder for the given endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The MCP server URL (e.g., `https://mcp.example.com/v1`)
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            auth: McpAuth::None,
            timeout: Duration::from_secs(30),
            headers: HashMap::new(),
            elicitation_handler: None,
        }
    }

    /// Set authentication for the connection.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = McpHttpClientBuilder::new("https://mcp.example.com")
    ///     .with_auth(McpAuth::bearer("my-token"));
    /// ```
    pub fn with_auth(mut self, auth: McpAuth) -> Self {
        self.auth = auth;
        self
    }

    /// Set the request timeout.
    ///
    /// Default is 30 seconds.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add a custom header to all requests.
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Configure an elicitation handler for the HTTP connection.
    ///
    /// When set, use [`connect_with_elicitation`](Self::connect_with_elicitation)
    /// to create a toolset that advertises elicitation capabilities.
    pub fn with_elicitation_handler(mut self, handler: Arc<dyn ElicitationHandler>) -> Self {
        self.elicitation_handler = Some(handler);
        self
    }

    /// Get the endpoint URL.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the configured timeout.
    pub fn get_timeout(&self) -> Duration {
        self.timeout
    }

    /// Get the authentication configuration.
    pub fn get_auth(&self) -> &McpAuth {
        &self.auth
    }

    /// Build the rmcp transport config from `self`. Resolves the auth token
    /// (extracting it from `McpAuth` / running the OAuth2 flow as needed)
    /// and applies any custom headers configured via [`Self::header`].
    ///
    /// Factored out of `connect` / `connect_with_elicitation` so both paths
    /// stay in sync. Previously the headers were stored on the builder but
    /// never reached the transport — calling `.header(...)` silently did
    /// nothing.
    #[cfg(feature = "http-transport")]
    async fn build_transport_config(
        &self,
    ) -> Result<rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig> {
        use adk_core::{ErrorCategory, ErrorComponent};
        use reqwest::header::{HeaderName, HeaderValue};
        use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
        use std::collections::HashMap;

        // Resolve auth token (rmcp's bearer_auth adds the "Bearer " prefix).
        let token = match &self.auth {
            McpAuth::Bearer(token) => Some(token.clone()),
            McpAuth::OAuth2(config) => {
                let token = config.get_or_refresh_token().await.map_err(|e| {
                    AdkError::new(
                        ErrorComponent::Tool,
                        ErrorCategory::Unauthorized,
                        "mcp.oauth.token_fetch",
                        format!("OAuth2 authentication failed: {e}"),
                    )
                })?;
                Some(token)
            }
            // API-key auth uses a non-standard header that rmcp's auth_header
            // doesn't model. Callers who need that can pass the key via
            // .header(...) instead.
            McpAuth::ApiKey { .. } => None,
            McpAuth::None => None,
        };

        // Convert the builder's stored string-keyed headers into the typed
        // map rmcp wants. Surface conversion errors with the offending name
        // so misconfigurations are obvious.
        let mut custom_headers: HashMap<HeaderName, HeaderValue> = HashMap::new();
        for (name, value) in &self.headers {
            let header_name = HeaderName::try_from(name.as_str()).map_err(|e| {
                AdkError::tool(format!("invalid MCP header name {name:?}: {e}"))
            })?;
            let header_value = HeaderValue::try_from(value.as_str()).map_err(|e| {
                AdkError::tool(format!("invalid MCP header value for {name:?}: {e}"))
            })?;
            custom_headers.insert(header_name, header_value);
        }

        let mut config = StreamableHttpClientTransportConfig::with_uri(self.endpoint.as_str());
        if let Some(token) = token {
            config = config.auth_header(token);
        }
        if !custom_headers.is_empty() {
            config = config.custom_headers(custom_headers);
        }
        Ok(config)
    }

    /// Connect to the MCP server and create a toolset.
    ///
    /// This method establishes a connection to the remote MCP server
    /// using the streamable HTTP transport.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The `http-transport` feature is not enabled
    /// - Connection to the server fails
    /// - Authentication fails
    #[cfg(feature = "http-transport")]
    pub async fn connect(
        self,
    ) -> Result<super::McpToolset<impl rmcp::service::Service<rmcp::RoleClient>>> {
        use rmcp::ServiceExt;
        use rmcp::transport::streamable_http_client::StreamableHttpClientTransport;

        let config = self.build_transport_config().await?;
        let transport = StreamableHttpClientTransport::from_config(config);

        // Connect using the service extension
        let client = ()
            .serve(transport)
            .await
            .map_err(|e| AdkError::tool(format!("Failed to connect to MCP server: {e}")))?;

        Ok(super::McpToolset::new(client))
    }

    /// Connect to the MCP server (stub when http-transport feature is disabled).
    #[cfg(not(feature = "http-transport"))]
    pub async fn connect(self) -> Result<()> {
        Err(AdkError::tool(
            "HTTP transport requires the 'http-transport' feature. \
             Add `adk-tool = { features = [\"http-transport\"] }` to your Cargo.toml",
        ))
    }

    /// Connect with elicitation support.
    ///
    /// Requires [`with_elicitation_handler`](Self::with_elicitation_handler) to have been called.
    /// Returns a `McpToolset<AdkClientHandler>` that advertises elicitation capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if no elicitation handler was configured or if the connection fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use adk_tool::{McpHttpClientBuilder, AutoDeclineElicitationHandler};
    /// use std::sync::Arc;
    ///
    /// let toolset = McpHttpClientBuilder::new("https://mcp.example.com/v1")
    ///     .with_elicitation_handler(Arc::new(AutoDeclineElicitationHandler))
    ///     .connect_with_elicitation()
    ///     .await?;
    /// ```
    #[cfg(feature = "http-transport")]
    pub async fn connect_with_elicitation(
        self,
    ) -> Result<super::McpToolset<impl rmcp::service::Service<rmcp::RoleClient>>> {
        use rmcp::ServiceExt;
        use rmcp::transport::streamable_http_client::StreamableHttpClientTransport;

        let handler = self.elicitation_handler.clone().ok_or_else(|| {
            AdkError::tool(
                "connect_with_elicitation requires with_elicitation_handler to be called first",
            )
        })?;

        let config = self.build_transport_config().await?;
        let transport = StreamableHttpClientTransport::from_config(config);
        let adk_handler = super::elicitation::AdkClientHandler::new(handler);
        let client = adk_handler
            .serve(transport)
            .await
            .map_err(|e| AdkError::tool(format!("failed to connect to MCP server: {e}")))?;

        Ok(super::McpToolset::new(client))
    }

    /// Connect with elicitation support (stub when http-transport feature is disabled).
    #[cfg(not(feature = "http-transport"))]
    pub async fn connect_with_elicitation(self) -> Result<()> {
        Err(AdkError::tool(
            "HTTP transport requires the 'http-transport' feature. \
             Add `adk-tool = { features = [\"http-transport\"] }` to your Cargo.toml",
        ))
    }
}

impl std::fmt::Debug for McpHttpClientBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpHttpClientBuilder")
            .field("endpoint", &self.endpoint)
            .field("auth", &self.auth)
            .field("timeout", &self.timeout)
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_new() {
        let builder = McpHttpClientBuilder::new("https://mcp.example.com");
        assert_eq!(builder.endpoint(), "https://mcp.example.com");
        assert_eq!(builder.get_timeout(), Duration::from_secs(30));
    }

    #[test]
    fn test_builder_with_auth() {
        let builder = McpHttpClientBuilder::new("https://mcp.example.com")
            .with_auth(McpAuth::bearer("test-token"));
        assert!(builder.get_auth().is_configured());
    }

    #[test]
    fn test_builder_timeout() {
        let builder =
            McpHttpClientBuilder::new("https://mcp.example.com").timeout(Duration::from_secs(60));
        assert_eq!(builder.get_timeout(), Duration::from_secs(60));
    }

    #[test]
    fn test_builder_headers() {
        let builder =
            McpHttpClientBuilder::new("https://mcp.example.com").header("X-Custom", "value");
        assert!(builder.headers.contains_key("X-Custom"));
    }

    /// Regression test: previously `.header()` stored entries on the builder
    /// but `connect()` never applied them to the transport config. Now they
    /// flow through, and invalid names/values produce a clear error instead
    /// of silently disappearing.
    #[cfg(feature = "http-transport")]
    #[tokio::test]
    async fn test_build_transport_config_applies_headers() {
        let builder = McpHttpClientBuilder::new("https://mcp.example.com")
            .header("X-Custom", "value")
            .header("Authorization", "License abc.def.ghi");
        let config = builder.build_transport_config().await.unwrap();
        assert_eq!(config.custom_headers.len(), 2);
        assert!(
            config
                .custom_headers
                .contains_key(&reqwest::header::HeaderName::from_static("x-custom"))
        );
    }

    #[cfg(feature = "http-transport")]
    #[tokio::test]
    async fn test_build_transport_config_rejects_invalid_header_name() {
        let builder =
            McpHttpClientBuilder::new("https://mcp.example.com").header("bad header", "value");
        let err = builder.build_transport_config().await.unwrap_err();
        assert!(err.to_string().contains("invalid MCP header name"));
    }
}
