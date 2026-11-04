//! HTTP transport for outbound A2A requests.
//!
//! The A2A client sends through a boxed [`tower::Service`] over
//! [`reqwest::Request`] rather than a `reqwest::Client` directly, so callers
//! can wrap the client in layers that rewrite each request — per-request auth
//! headers (e.g. DPoP proofs whose value changes every call), logging, or
//! retries — while a bare [`reqwest::Client`] still works unchanged.

use std::sync::LazyLock;
use tower::{BoxError, Service, ServiceExt};

/// Boxed HTTP transport: takes a fully built [`reqwest::Request`] and yields
/// the [`reqwest::Response`]. Clone is cheap.
pub type HttpTransport =
    tower::util::BoxCloneSyncService<reqwest::Request, reqwest::Response, BoxError>;

/// Box any compatible service (a [`reqwest::Client`], or a `tower` stack
/// around one) into an [`HttpTransport`].
pub fn box_transport<S>(service: S) -> HttpTransport
where
    S: Service<reqwest::Request, Response = reqwest::Response> + Clone + Send + Sync + 'static,
    S::Error: Into<BoxError>,
    S::Future: Send + 'static,
{
    tower::util::BoxCloneSyncService::new(service.map_err(Into::into))
}

/// A default client with no layers, for the constructors that take none.
pub(crate) fn plain_transport() -> HttpTransport {
    box_transport(reqwest::Client::new())
}

/// The client requests are assembled with. It only builds them — a
/// [`reqwest::RequestBuilder`] needs a client — and never sends one, so the
/// client inside the transport still applies its own default headers and TLS
/// settings when it executes the request.
pub(crate) fn request_builder() -> &'static reqwest::Client {
    static BUILDER: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);
    &BUILDER
}

/// Builds `request` and sends it through `http`.
pub(crate) async fn send(
    http: &HttpTransport,
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, BoxError> {
    let request = request.build()?;
    http.clone().oneshot(request).await
}
