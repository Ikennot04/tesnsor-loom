use axum::{routing::post, Json, Router};
use serde::Deserialize;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Deserialize)]
struct GreetArgs {
    name: String,
}

async fn greet(Json(a): Json<GreetArgs>) -> Json<String> {
    Json(tensorloom_core::greet(&a.name))
}

#[tokio::main]
async fn main() {
    // Serves the built frontend (run `npm run build` first to create dist/)
    let ui = ServeDir::new("dist").not_found_service(ServeFile::new("dist/index.html"));

    let app = Router::new()
        .route("/api/greet", post(greet))
        .fallback_service(ui);

    let addr = "127.0.0.1:7878";
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("Tensor Loom running at http://{addr}");

    if std::env::args().any(|a| a == "--open") {
        let _ = open::that(format!("http://{addr}"));
    }

    axum::serve(listener, app).await.unwrap();
}
