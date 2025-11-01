use actix_web::{rt::time::sleep, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Data structure for an incoming question.
#[derive(Deserialize)]
struct AskRequest {
    /// User's question.
    query: String,
    /// Number of documents to retrieve (optional).
    #[serde(default)]
    top_k: Option<u8>,
}

/// Data structure for the response returned by the AI layer.
#[derive(Serialize)]
struct AskResponse {
    /// Final answer text.
    answer: String,
    /// List of citations augmented with retrieved text.
    citations: Vec<Citation>,
    /// Total latency in milliseconds.
    latency_ms: u128,
}

/// Citation payload shared between the Rust gateway and Python backend.
#[derive(Clone, Serialize, Deserialize)]
struct Citation {
    source: String,
    text: String,
}

/// Shape of the response returned by the Python FastAPI service.
#[derive(Deserialize)]
struct PythonAskResponse {
    answer: String,
    citations: Vec<Citation>,
}

/// Payload for ingesting documents.
#[derive(Deserialize, Serialize)]
struct AddDocRequest {
    text: String,
    #[serde(default)]
    metadata: Option<HashMap<String, String>>,
}

/// Response returned when a document is added through the gateway.
#[derive(Serialize)]
struct AddDocResponse {
    document_id: String,
    latency_ms: u128,
}

/// Shape of the Python add document response.
#[derive(Deserialize)]
struct PythonAddDocResponse {
    document_id: String,
}

const DEFAULT_TOP_K: u8 = 4;
const MAX_TOP_K: u8 = 20;
const PYTHON_DEFAULT_URL: &str = "http://127.0.0.1:8001";
const PYTHON_ASK_ENDPOINT: &str = "/ask";
const PYTHON_ADD_DOC_ENDPOINT: &str = "/add_doc";
const API_KEY_HEADER: &str = "X-API-KEY";
const MAX_RETRIES: usize = 3;
const BASE_BACKOFF_MS: u64 = 120;

/// Extracts the API key and validates that it is present and non-empty.
fn extract_api_key(req: &HttpRequest) -> Result<String, HttpResponse> {
    match req.headers().get(API_KEY_HEADER) {
        Some(value) => {
            let api_key = value.to_str().unwrap_or("").trim();
            if api_key.is_empty() {
                log_gateway_event(
                    "gateway.auth_failed",
                    json!({
                        "path": req.path(),
                        "method": req.method().as_str(),
                        "reason": "empty"
                    }),
                );
                Err(HttpResponse::Unauthorized().json(json!({
                    "error": "X-API-KEY header must not be empty"
                })))
            } else {
                Ok(api_key.to_owned())
            }
        }
        None => {
            log_gateway_event(
                "gateway.auth_failed",
                json!({
                    "path": req.path(),
                    "method": req.method().as_str(),
                    "reason": "missing"
                }),
            );
            Err(HttpResponse::Unauthorized().json(json!({
                "error": "Missing X-API-KEY header"
            })))
        }
    }
}

/// Pretty-print structured gateway logs.
fn log_gateway_event(event: &str, details: Value) {
    let log_entry = json!({
        "event": event,
        "details": details,
    });
    match serde_json::to_string_pretty(&log_entry) {
        Ok(pretty) => println!("{}", pretty),
        Err(_) => println!("{}", log_entry),
    }
}

/// Helper to compute the python service base url from the environment.
fn python_service_base_url() -> String {
    std::env::var("PYTHON_AI_URL").unwrap_or_else(|_| PYTHON_DEFAULT_URL.to_string())
}

/// Sends a JSON payload to the Python backend with retry and exponential backoff.
async fn post_with_retry<T, U>(
    client: &Client,
    endpoint: &str,
    payload: &T,
) -> Result<U, HttpResponse>
where
    T: Serialize,
    U: DeserializeOwned,
{
    let mut last_status: Option<u16> = None;
    let mut last_error: Option<String> = None;
    let base_url = python_service_base_url();
    let url = format!("{}{}", base_url.trim_end_matches('/'), endpoint);

    for attempt in 0..MAX_RETRIES {
        match client.post(&url).json(payload).send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_server_error() && attempt + 1 < MAX_RETRIES {
                    last_status = Some(status.as_u16());
                    log_gateway_event(
                        "gateway.retry",
                        json!({
                            "url": url,
                            "attempt": attempt + 1,
                            "status": status.as_u16(),
                            "reason": "upstream_server_error"
                        }),
                    );
                    sleep(Duration::from_millis(BASE_BACKOFF_MS * (1 << attempt))).await;
                    continue;
                }

                match resp.json::<U>().await {
                    Ok(body) => return Ok(body),
                    Err(err) => {
                        last_status = Some(status.as_u16());
                        last_error = Some(format!("Failed to parse upstream response: {}", err));
                        break;
                    }
                }
            }
            Err(err) => {
                last_error = Some(err.to_string());
                if attempt + 1 < MAX_RETRIES {
                    log_gateway_event(
                        "gateway.retry",
                        json!({
                            "url": url,
                            "attempt": attempt + 1,
                            "status": "network_error",
                            "error": err.to_string()
                        }),
                    );
                    sleep(Duration::from_millis(BASE_BACKOFF_MS * (1 << attempt))).await;
                    continue;
                } else {
                    break;
                }
            }
        }
    }

    Err(HttpResponse::BadGateway().json(json!({
        "error": "Failed to reach Python service",
        "upstream_status": last_status,
        "last_error": last_error,
    })))
}

/// Handler for the `/api/ask` endpoint.
async fn ask_handler(
    http_req: HttpRequest,
    client: web::Data<Client>,
    req: web::Json<AskRequest>,
) -> impl Responder {
    let api_key = match extract_api_key(&http_req) {
        Ok(key) => key,
        Err(resp) => return resp,
    };

    let top_k = req.top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);
    let start = Instant::now();

    let query = req.query.clone();
    let query_length = query.len();
    let payload = json!({
        "query": query,
        "top_k": top_k
    });

    match post_with_retry::<_, PythonAskResponse>(&client, PYTHON_ASK_ENDPOINT, &payload).await {
        Ok(body) => {
            let latency_ms = start.elapsed().as_millis();
            let response = HttpResponse::Ok().json(AskResponse {
                answer: body.answer,
                citations: body.citations,
                latency_ms,
            });
            log_gateway_event(
                "gateway.request",
                json!({
                    "path": "/api/ask",
                    "method": "POST",
                    "status": response.status().as_u16(),
                    "latency_ms": latency_ms,
                    "api_key_present": !api_key.is_empty(),
                    "request": {
                        "query_length": query_length,
                        "top_k": top_k,
                    }
                }),
            );
            response
        }
        Err(resp) => {
            let latency_ms = start.elapsed().as_millis();
            log_gateway_event(
                "gateway.request",
                json!({
                    "path": "/api/ask",
                    "method": "POST",
                    "status": resp.status().as_u16(),
                    "latency_ms": latency_ms,
                    "api_key_present": !api_key.is_empty(),
                }),
            );
            resp
        }
    }
}

/// Handler to forward document ingestion to the Python backend.
async fn add_doc_handler(
    http_req: HttpRequest,
    client: web::Data<Client>,
    req: web::Json<AddDocRequest>,
) -> impl Responder {
    let api_key = match extract_api_key(&http_req) {
        Ok(key) => key,
        Err(resp) => return resp,
    };

    let start = Instant::now();
    let metadata_keys: Vec<String> = req
        .metadata
        .as_ref()
        .map(|map| map.keys().cloned().collect())
        .unwrap_or_else(Vec::new);
    let text_length = req.text.len();

    match post_with_retry::<_, PythonAddDocResponse>(&client, PYTHON_ADD_DOC_ENDPOINT, &*req).await
    {
        Ok(body) => {
            let latency_ms = start.elapsed().as_millis();
            let response = HttpResponse::Ok().json(AddDocResponse {
                document_id: body.document_id,
                latency_ms,
            });
            log_gateway_event(
                "gateway.request",
                json!({
                    "path": "/api/add_doc",
                    "method": "POST",
                    "status": response.status().as_u16(),
                    "latency_ms": latency_ms,
                    "api_key_present": !api_key.is_empty(),
                    "request": {
                        "text_length": text_length,
                        "metadata_keys": metadata_keys,
                    }
                }),
            );
            response
        }
        Err(resp) => {
            let latency_ms = start.elapsed().as_millis();
            log_gateway_event(
                "gateway.request",
                json!({
                    "path": "/api/add_doc",
                    "method": "POST",
                    "status": resp.status().as_u16(),
                    "latency_ms": latency_ms,
                    "api_key_present": !api_key.is_empty(),
                }),
            );
            resp
        }
    }
}

/// Health check endpoint.
async fn health_handler() -> impl Responder {
    HttpResponse::Ok().body("OK")
}

/// Entry point.  Starts the Actix server and registers routes.
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let port = std::env::var("RUST_API_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8000);
    println!("Starting Rust API on port {}", port);
    HttpServer::new(|| {
        App::new()
            .app_data(web::Data::new(Client::new()))
            .route("/api/ask", web::post().to(ask_handler))
            .route("/api/add_doc", web::post().to(add_doc_handler))
            .route("/api/health", web::get().to(health_handler))
    })
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
