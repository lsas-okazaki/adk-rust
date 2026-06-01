//! HTTP transport for outbound A2A requests.
//!
//! The A2A client sends through a boxed [`tower::Service`] over
//! [`reqwest::Request`] rather than a `reqwest::Client` directly, so callers
//! can wrap the client in layers that rewrite each request — per-request auth
//! headers (e.g. DPoP proofs whose value changes every call), logging, or
//! retries — while a bare [`reqwest::Client`] still works unchanged.

use serde::Serialize;
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

/// `GET url` through the transport.
pub(crate) async fn get(http: &HttpTransport, url: &str) -> Result<reqwest::Response, BoxError> {
    let url = reqwest::Url::parse(url)?;
    http.clone().oneshot(reqwest::Request::new(reqwest::Method::GET, url)).await
}

/// `POST url` with a JSON body through the transport.
///
/// The request is assembled by hand rather than via `reqwest::RequestBuilder`
/// so it does not depend on which client executes it: the client inside the
/// transport still applies its own default headers and TLS settings.
pub(crate) async fn post_json<T: Serialize + ?Sized>(
    http: &HttpTransport,
    url: &str,
    body: &T,
) -> Result<reqwest::Response, BoxError> {
    use reqwest::header::{CONTENT_TYPE, HeaderValue};

    let url = reqwest::Url::parse(url)?;
    let mut request = reqwest::Request::new(reqwest::Method::POST, url);
    request.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    *request.body_mut() = Some(serde_json::to_vec(body)?.into());
    http.clone().oneshot(request).await
}
