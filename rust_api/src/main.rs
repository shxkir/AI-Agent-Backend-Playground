use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;

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
    /// List of citations in the form "source:line-start-line-end".
    citations: Vec<String>,
    /// Total latency in milliseconds.
    latency_ms: u128,
}

/// Shape of the response returned by the Python FastAPI service.
#[derive(Deserialize)]
struct PythonAskResponse {
    answer: String,
    citations: Vec<String>,
}

/// Handler for the `/api/ask` endpoint.
async fn ask_handler(req: web::Json<AskRequest>) -> impl Responder {
    let start = Instant::now();
    let top_k = req.top_k.unwrap_or(4);
    let query = req.query.clone();
    let client = Client::new();
    let service_url =
        std::env::var("PYTHON_AI_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    let endpoint = format!("{}/ask", service_url.trim_end_matches('/'));

    let payload = json!({
        "query": query,
        "top_k": top_k
    });

    match client.post(endpoint).json(&payload).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<PythonAskResponse>().await {
            Ok(body) => {
                let latency_ms = start.elapsed().as_millis();
                HttpResponse::Ok().json(AskResponse {
                    answer: body.answer,
                    citations: body.citations,
                    latency_ms,
                })
            }
            Err(err) => {
                HttpResponse::BadGateway().body(format!("Failed to parse Python response: {}", err))
            }
        },
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            HttpResponse::BadGateway().body(format!(
                "Python service returned status {} with body: {}",
                status, text
            ))
        }
        Err(err) => {
            HttpResponse::BadGateway().body(format!("Failed to reach Python service: {}", err))
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
            .route("/api/ask", web::post().to(ask_handler))
            .route("/api/health", web::get().to(health_handler))
    })
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
