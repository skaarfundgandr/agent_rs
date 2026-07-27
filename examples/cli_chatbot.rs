#![cfg_attr(not(feature = "rag"), allow(unused))]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

#[cfg(feature = "rag")]
mod rag_main {
    use agent_rs::agent::embeddings::EmbeddingService;
    use agent_rs::agent::permission::PermissionPolicy;
    use agent_rs::agent::tools::{
        CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ReadDocumentTool,
        ToolRegistryBuilder, WriteDocumentTool,
    };
    use agent_rs::mcp::registry::McpRegistry;
    use agent_rs::rag::RagPipeline;
    use agent_rs::security::{SandboxConfig, SharedSandbox};
    use anyhow::Result;
    use dotenvy::dotenv;
    use rig_core::integrations::cli_chatbot::ChatBotBuilder;
    use rig_core::prelude::*;
    use rig_core::providers::openai;
    use std::collections::HashSet;
    use std::env;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn build_shared_sandbox() -> Result<Arc<SharedSandbox>> {
        let config = match env::var("SANDBOX_ROOTS") {
            Ok(s) if !s.trim().is_empty() => {
                SandboxConfig::new(s.split(',').map(|p| PathBuf::from(p.trim())).collect())?
            }
            _ => SandboxConfig::single("./")?,
        };
        Ok(Arc::new(SharedSandbox::from(config)))
    }

    pub async fn run() -> Result<()> {
        dotenv().ok();

        // ---------- env-var configuration ----------
        let chat_model_name =
            env::var("CHAT_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());
        let chat_base_url =
            env::var("CHAT_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
        let fastembed_model_name =
            env::var("FASTEMBED_MODEL").unwrap_or_else(|_| "BGESmallENV15".to_string());
        let rag_top_k = env::var("RAG_TOP_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let rag_db_path =
            env::var("RAG_DB_PATH").unwrap_or_else(|_| "./rag_data/rag.db".to_string());
        let rag_index_path =
            env::var("RAG_INDEX_PATH").unwrap_or_else(|_| "./rag_data/rag.tvim".to_string());
        let db_path = PathBuf::from(&rag_db_path);
        let index_path = PathBuf::from(&rag_index_path);

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // ---------- startup banner ----------
        println!("--- cli_chatbot configuration ---");
        println!("  CHAT_BASE_URL    = {chat_base_url}");
        println!("  CHAT_MODEL       = {chat_model_name}");
        println!("  FASTEMBED_MODEL  = {fastembed_model_name}");
        println!("  RAG_DB_PATH      = {rag_db_path}");
        println!("  RAG_INDEX_PATH   = {rag_index_path}");
        if chat_base_url == "http://127.0.0.1:1234/v1" {
            println!(
                "  WARNING: CHAT_BASE_URL is the local default (127.0.0.1:1234). \
                 Set CHAT_BASE_URL in .env or the environment to point at your provider."
            );
        }
        println!("-------------------------------");

        // ---------- chat client (OpenAI-compatible local endpoint) ----------
        let chat_client = openai::CompletionsClient::builder()
            .base_url(chat_base_url)
            .api_key(env::var("API_KEY").expect("Missing API_KEY env var"))
            .build()?;

        // ---------- embeddings (local fastembed) ----------
        let fastembed_variant: agent_rs::agent::embeddings::FastembedModel =
            fastembed_model_name.parse().map_err(|e: String| {
                anyhow::anyhow!("Unknown FASTEMBED_MODEL '{fastembed_model_name}': {e}")
            })?;
        println!("Loading fastembed model '{fastembed_model_name}' (downloads on first run)...");
        let embedding_service = EmbeddingService::builder()
            .model(fastembed_variant)
            .show_progress(true)
            .build()
            .map_err(|e| anyhow::anyhow!("failed to load fastembed model: {e}"))?;
        let embedding_dim = embedding_service.ndims();
        println!("Embedding model ready ({embedding_dim} dims).");

        let shared_sandbox = build_shared_sandbox()?;

        let rag = RagPipeline::builder()
            .embedder(embedding_service)
            .db_path(&db_path)
            .index_path(&index_path)
            .extensions(["txt", "md", "pdf"])
            .sandbox(shared_sandbox.clone())
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("failed to build RAG pipeline: {e}"))?;
        println!(
            "RAG ready ({} chunks).",
            rag.indexer.chunk_count().await.unwrap_or(0)
        );

        // ---------- MCP tools + internal tools ----------
        let policy = match env::var("PERMISSION_POLICY").as_deref() {
            Ok("deny") => PermissionPolicy::DenyAll,
            Ok("prompt") => PermissionPolicy::CliPrompt,
            _ => {
                eprintln!(
                    "WARNING: PERMISSION_POLICY not set; defaulting to CliPrompt. \
                     Set PERMISSION_POLICY=allow|deny|prompt to silence this warning."
                );
                PermissionPolicy::CliPrompt
            }
        };

        let mcp_runtime = if std::path::Path::new("./mcp.json").exists() {
            Some(
                McpRegistry::from_path("./mcp.json")?
                    .connect(policy.clone())
                    .await?,
            )
        } else {
            eprintln!("  No mcp.json found — skipping MCP tools.");
            None
        };

        let read_extensions = HashSet::from(["txt", "md", "pdf"].map(String::from));
        let write_extensions = HashSet::from(["txt", "md"].map(String::from));
        let grep_extensions = HashSet::from(["txt", "md"].map(String::from));

        let mut builder = ToolRegistryBuilder::new();

        if let Some(ref runtime) = mcp_runtime {
            builder = builder.register_mcp("mcp", runtime)?;
        }

        let chat_client_for_ctx = chat_client.clone();
        let chat_model_name_for_ctx = chat_model_name.clone();

        builder = builder
            .register("filesystem", {
                let sb = Arc::clone(&shared_sandbox);
                let exts = read_extensions.clone();
                let pol = policy.clone();
                move || {
                    Box::new(ReadDocumentTool::new(
                        Arc::clone(&sb),
                        exts.clone(),
                        pol.clone(),
                    ))
                }
            })?
            .register("filesystem", {
                let sb = Arc::clone(&shared_sandbox);
                let exts = write_extensions.clone();
                let pol = policy.clone();
                move || {
                    Box::new(WriteDocumentTool::new(
                        Arc::clone(&sb),
                        exts.clone(),
                        pol.clone(),
                    ))
                }
            })?
            .register("filesystem", {
                let sb = Arc::clone(&shared_sandbox);
                let pol = policy.clone();
                move || Box::new(ListDirectoryTool::new(Arc::clone(&sb), pol.clone()))
            })?
            .register("filesystem", {
                let sb = Arc::clone(&shared_sandbox);
                let exts = grep_extensions.clone();
                let pol = policy.clone();
                move || {
                    Box::new(GrepSearchTool::new(
                        Arc::clone(&sb),
                        exts.clone(),
                        pol.clone(),
                    ))
                }
            })?
            .register("filesystem", {
                let sb = Arc::clone(&shared_sandbox);
                let pol = policy.clone();
                move || Box::new(GlobSearchTool::new(Arc::clone(&sb), pol.clone()))
            })?
            .register("rag", {
                let idx = rag.indexer.clone();
                let pol = policy.clone();
                move || Box::new(idx.tool(pol.clone()))
            })?
            .register("rag", {
                let idx = rag.indexer.clone();
                move || Box::new(idx.search_tool())
            })?
            .register("context", move || {
                let agent = chat_client_for_ctx
                    .agent(&chat_model_name_for_ctx)
                    .default_max_turns(20)
                    .preamble("You are a summarization assistant.")
                    .build();
                Box::new(CompactTool::new(agent)) as Box<dyn rig_core::tool::ToolDyn>
            })?
            .enable(&["mcp", "filesystem", "rag", "context"]);

        let registry = builder.build();
        let tools = registry.active_tools();

        // ---------- agent ----------
        let agent = chat_client
            .agent(&chat_model_name)
            .tools(tools)
            .preamble(
                "You are a helpful AI agent capable of RAG and reading documents \
                 using internal tools. Use `manage_rag` with action='add' to index a \
                 document on the fly, then query it via your knowledge — relevant \
                 passages will be supplied automatically as dynamic context. \
                 Use `rag_search` to run an explicit semantic search over indexed sources \
                 with your own query.",
            )
            .dynamic_context(rag_top_k, rag.vector_index)
            .default_max_turns(20)
            .temperature(0.6)
            .build();

        let chatbot = ChatBotBuilder::new().agent(agent).show_usage().build();

        let save_path = index_path;

        match chatbot.run().await {
            Ok(()) => {}
            Err(_) => {
                if let Err(e) = rag.indexer.pipeline().save(&save_path).await {
                    eprintln!("warning: failed to save RAG index on shutdown: {e}")
                } else {
                    println!("Saved to {save_path:?}.");
                }
            }
        };

        if let Err(e) = rag.indexer.pipeline().save(&save_path).await {
            eprintln!("warning: failed to save RAG index on shutdown: {e}")
        } else {
            println!("Saved to {save_path:?}.");
        }

        Ok(())
    }
}

#[cfg(feature = "rag")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rag_main::run().await
}

#[cfg(not(feature = "rag"))]
fn main() {
    eprintln!("cli_chatbot example requires --features rag");
}
