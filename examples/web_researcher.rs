//! # Autonomous Web Research Agent Example
//!
//! Demonstrates an agent that searches the web, fetches pages,
//! dynamically writes them to files, indexes them into a [RagPipeline],
//! and answers the query using the dynamic RAG context.
//!
//! Run with:
//! ```bash
//! cargo run --example web_researcher --features rag
//! ```

#![cfg_attr(not(feature = "rag"), allow(unused))]

#[cfg(feature = "rag")]
mod researcher_main {
    use agent_rs::agent::embeddings::EmbeddingService;
    use agent_rs::agent::permission::PermissionPolicy;
    use agent_rs::agent::tools::{
        ManageRagTool, RagSourceRegistry, ToolRegistryBuilder, WriteDocumentTool,
    };
    use agent_rs::rag::{ErasedEmbedder, RagPipeline};
    use agent_rs::security::{SandboxConfig, SharedSandbox};
    use anyhow::Result;
    use dotenvy::dotenv;
    use rig_core::completion::{Prompt, ToolDefinition};
    use rig_core::prelude::*;
    use rig_core::providers::openai;
    use rig_core::tool::Tool;
    use serde_json::json;
    use std::collections::HashSet;
    use std::env;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    // Define custom errors for our mock tools
    #[derive(Debug, thiserror::Error)]
    #[error("Mock tool error: {0}")]
    struct MockToolError(String);

    // Mock search tool args
    #[derive(Debug, serde::Deserialize, serde::Serialize)]
    struct SearchArgs {
        query: String,
    }

    // Mock web search tool
    struct WebSearchTool;

    impl Tool for WebSearchTool {
        const NAME: &'static str = "web_search";
        type Error = MockToolError;
        type Args = SearchArgs;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "Search the web for resources and documentation. Returns list of page titles and URLs.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            println!("  [WebSearchTool] Searching for: '{}'", args.query);
            let q = args.query.to_lowercase();
            if q.contains("rust") || q.contains("async") {
                Ok(json!([
                    {
                        "title": "Rust Async Book",
                        "url": "https://rust-lang.github.io/async-rust/index.html",
                        "snippet": "Asynchronous programming in Rust allows you to run multiple tasks concurrently on a small number of OS threads."
                    },
                    {
                        "title": "Tokio Tutorial",
                        "url": "https://tokio.rs/tokio/tutorial",
                        "snippet": "Tokio is an asynchronous runtime for the Rust programming language. Tasks are lightweight green threads spawned using tokio::spawn."
                    }
                ]).to_string())
            } else {
                Ok(json!([
                    {
                        "title": "AgentRS Github",
                        "url": "https://github.com/skaarfundgandr/agent_rs",
                        "snippet": "AgentRS is a high-performance Rust-based AI agent framework with native RAG support and MCP integration."
                    }
                ]).to_string())
            }
        }
    }

    // Mock web fetch tool args
    #[derive(Debug, serde::Deserialize, serde::Serialize)]
    struct FetchArgs {
        url: String,
    }

    // Mock web fetch tool
    struct WebFetchTool;

    impl Tool for WebFetchTool {
        const NAME: &'static str = "web_fetch";
        type Error = MockToolError;
        type Args = FetchArgs;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "Fetch full text content of a web page by its URL.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch"
                        }
                    },
                    "required": ["url"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            println!("  [WebFetchTool] Fetching URL: '{}'", args.url);
            let url = args.url.as_str();
            if url.contains("async-rust") {
                Ok("Async Rust is built on the Future trait. A Future represents an asynchronous computation. Futures must be polled to make progress. The async/await syntax makes writing asynchronous code look synchronous. Tokio is a common runtime used to execute async Rust.".to_string())
            } else if url.contains("tokio.rs") {
                Ok("Tokio provides a multi-threaded work-stealing scheduler, reactor, and timers. Tasks are lightweight green threads spawned using tokio::spawn. Tokio handles non-blocking I/O efficiently, allowing thousands of concurrent network tasks.".to_string())
            } else {
                Ok("AgentRS is a Rust agent framework. It integrates MCP tools and RAG pipelines. Developers can build smart agents with automatic memory compaction, permission gates, and OpenTelemetry logging.".to_string())
            }
        }
    }

    pub async fn run() -> Result<()> {
        dotenv().ok();

        let chat_model_name =
            env::var("CHAT_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());
        let chat_base_url =
            env::var("CHAT_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
        let fastembed_model_name =
            env::var("FASTEMBED_MODEL").unwrap_or_else(|_| "Xenova/bge-small-en-v1.5".to_string());

        let api_key = match env::var("API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => anyhow::bail!("Missing API_KEY env var in .env"),
        };

        let db_path = PathBuf::from("./rag_data/research_rag.db");
        let index_path = PathBuf::from("./rag_data/research_rag.tvim");

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        println!("Initializing RAG services...");
        let fastembed_variant = fastembed_model_name.parse().map_err(|e: String| {
            anyhow::anyhow!("Unknown FASTEMBED_MODEL '{fastembed_model_name}': {e}")
        })?;
        let embedding_service = EmbeddingService::from_fastembed(fastembed_variant)?;
        let embedding_dim = embedding_service.ndims();
        let embedder_arc: Arc<dyn ErasedEmbedder> = Arc::new(embedding_service);

        let rag_extensions = HashSet::from(["txt", "md"].map(String::from));
        let pipeline = Arc::new(
            RagPipeline::open_or_create(
                &db_path,
                &index_path,
                embedding_dim,
                4,
                Some(rag_extensions.clone()),
            )
            .await?,
        );
        let index = pipeline.build(Arc::clone(&embedder_arc));

        let shared_sandbox = Arc::new(SharedSandbox::from(SandboxConfig::single("./")?));
        let policy = PermissionPolicy::AllowAll; // Allow the agent to search/write/RAG without prompts in this example

        let rag_registry = Arc::new(Mutex::new(RagSourceRegistry::new(rag_extensions)));

        // Register tools
        let registry = ToolRegistryBuilder::new()
            .register("research", || Box::new(WebSearchTool))?
            .register("research", || Box::new(WebFetchTool))?
            .register("filesystem", {
                let sb = Arc::clone(&shared_sandbox);
                let exts = HashSet::from(["txt", "md"].map(String::from));
                let pol = policy.clone();
                move || {
                    Box::new(WriteDocumentTool::new(
                        Arc::clone(&sb),
                        exts.clone(),
                        pol.clone(),
                    ))
                }
            })?
            .register("rag", {
                let sb = Arc::clone(&shared_sandbox);
                let reg = Arc::clone(&rag_registry);
                let pipe = Arc::clone(&pipeline);
                let emb = Arc::clone(&embedder_arc);
                let pol = policy.clone();
                move || {
                    Box::new(ManageRagTool::new(
                        Arc::clone(&reg),
                        Arc::clone(&pipe),
                        Arc::clone(&emb),
                        Arc::clone(&sb),
                        pol.clone(),
                    ))
                }
            })?
            .enable(&["research", "filesystem", "rag"])
            .build();

        let chat_client = openai::CompletionsClient::builder()
            .base_url(chat_base_url)
            .api_key(api_key)
            .build()?;

        let agent = chat_client
            .agent(&chat_model_name)
            .tools(registry.active_tools())
            .preamble(
                "You are an autonomous web researcher. \
                Steps to follow: \
                1. Use `web_search` to find relevant URLs. \
                2. Use `web_fetch` to get the page contents. \
                3. Use `write_document` to write the fetched contents to a file (e.g. `research_notes.txt`). \
                4. Use `manage_rag` with action='add' and path='research_notes.txt' to index it. \
                5. Once indexed, answer the user's research question in detail based on the dynamic RAG context."
            )
            .dynamic_context(2, index)
            .default_max_turns(20)
            .build();

        let args: Vec<String> = env::args().collect();
        let prompt = if args.len() > 1 {
            args[1..].join(" ")
        } else {
            "Research asynchronous Rust programming features and summarize them.".to_string()
        };

        println!("Research Prompt: {}", prompt);
        println!("Running agent. Please wait...");
        let output = agent.prompt(&prompt).await?;
        println!("\n=== Research Report ===\n{}", output);

        // Clean up temporary research note file if it was created
        if Path::new("research_notes.txt").exists() {
            let _ = std::fs::remove_file("research_notes.txt");
        }

        Ok(())
    }
}

#[cfg(feature = "rag")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    researcher_main::run().await
}

#[cfg(not(feature = "rag"))]
fn main() {
    eprintln!("web_researcher example requires --features rag");
}
