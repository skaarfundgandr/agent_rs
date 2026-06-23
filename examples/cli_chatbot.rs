#![cfg_attr(not(feature = "rag"), allow(unused))]

#[cfg(feature = "rag")]
mod rag_main {
    use agent_rs_lib::agent::embeddings::EmbeddingService;
    use agent_rs_lib::agent::permission::PermissionPolicy;
    use agent_rs_lib::agent::tools::{
        CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ManageRagTool,
        RagSourceRegistry, ReadDocumentTool, WriteDocumentTool,
    };
    use agent_rs_lib::config::McpConfig;
    use agent_rs_lib::mcp::client::McpClient;
    use agent_rs_lib::rag::{ErasedEmbedder, RagPipeline};
    use agent_rs_lib::security::{SandboxConfig, SharedSandbox};
    use anyhow::Result;
    use dotenvy::dotenv;
    use rig_core::integrations::cli_chatbot::ChatBotBuilder;
    use rig_core::prelude::*;
    use rig_core::providers::openai;
    use rig_core::tool::ToolDyn;
    use std::collections::HashSet;
    use std::env;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

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
            env::var("FASTEMBED_MODEL").unwrap_or_else(|_| "Xenova/bge-small-en-v1.5".to_string());
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
        let fastembed_variant: rig_fastembed::FastembedModel =
            fastembed_model_name.parse().map_err(|e: String| {
                anyhow::anyhow!("Unknown FASTEMBED_MODEL '{fastembed_model_name}': {e}")
            })?;
        println!("Loading fastembed model '{fastembed_model_name}' (downloads on first run)...");
        let embedding_service = EmbeddingService::from_fastembed(fastembed_variant)
            .map_err(|e| anyhow::anyhow!("failed to load fastembed model: {e}"))?;
        let embedding_dim = embedding_service.ndims();
        println!("Embedding model ready ({embedding_dim} dims).");

        let embedder_arc: Arc<dyn ErasedEmbedder> = Arc::new(embedding_service);

        // ---------- RAG extensions ----------
        let rag_extensions = HashSet::from(["txt", "md", "pdf"].map(String::from));

        // ---------- RAG pipeline (open-or-create) ----------
        let pipeline = Arc::new(
            RagPipeline::open_or_create(&db_path, &index_path, embedding_dim, 4, Some(rag_extensions.clone()))
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "failed to open RAG pipeline (db={db_path:?}, idx={index_path:?}, dim={embedding_dim}): {e}"
                    )
                })?,
        );
        println!(
            "RAG pipeline ready ({} chunks already indexed).",
            pipeline.chunk_count().await.unwrap_or(0)
        );

        let index = pipeline.build(Arc::clone(&embedder_arc));

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

        let mut tools = McpClient::new(McpConfig::from_path("./mcp.json").unwrap())
            .tools(policy.clone())
            .await?;

        let compaction_agent = chat_client
            .agent(&chat_model_name)
            .default_max_turns(20)
            .preamble("You are a summarization assistant.")
            .build();
        let compaction_tool = CompactTool::new(compaction_agent);

        let read_extensions = HashSet::from(["txt", "md", "pdf"].map(String::from));
        let write_extensions = HashSet::from(["txt", "md"].map(String::from));
        let grep_extensions = HashSet::from(["txt", "md"].map(String::from));
        let rag_registry = Arc::new(Mutex::new(RagSourceRegistry::new(rag_extensions)));

        let shared_sandbox = build_shared_sandbox()?;

        let internal_tools: Vec<Box<dyn ToolDyn>> = vec![
            Box::new(ReadDocumentTool::new(
                Arc::clone(&shared_sandbox),
                read_extensions,
                policy.clone(),
            )),
            Box::new(WriteDocumentTool::new(
                Arc::clone(&shared_sandbox),
                write_extensions,
                policy.clone(),
            )),
            Box::new(ListDirectoryTool::new(
                Arc::clone(&shared_sandbox),
                policy.clone(),
            )),
            Box::new(GrepSearchTool::new(
                Arc::clone(&shared_sandbox),
                grep_extensions,
                policy.clone(),
            )),
            Box::new(GlobSearchTool::new(
                Arc::clone(&shared_sandbox),
                policy.clone(),
            )),
            Box::new(ManageRagTool::new(
                rag_registry,
                Arc::clone(&pipeline),
                Arc::clone(&embedder_arc),
                Arc::clone(&shared_sandbox),
                policy,
            )),
        ];
        tools.extend(internal_tools);
        tools.push(Box::new(compaction_tool));

        // ---------- agent ----------
        let agent = chat_client
            .agent(&chat_model_name)
            .tools(tools)
            .preamble(
                "You are a helpful AI agent capable of RAG and reading documents \
                 using internal tools. Use `manage_rag` with action='add' to index a \
                 document on the fly, then query it via your knowledge — relevant \
                 passages will be supplied automatically as dynamic context.",
            )
            .dynamic_context(rag_top_k, index)
            .default_max_turns(20)
            .temperature(0.6)
            .build();

        let chatbot = ChatBotBuilder::new().agent(agent).show_usage().build();

        // ---------- run with graceful shutdown ----------
        let pipeline_for_save = Arc::clone(&pipeline);
        let save_path = index_path;
        tokio::select! {
            res = chatbot.run() => {
                if let Err(e) = pipeline_for_save.save(&save_path).await {
                    eprintln!("warning: failed to save RAG index on exit: {e}");
                }
                res?;
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nReceived Ctrl-C, saving RAG index...");
                if let Err(e) = pipeline_for_save.save(&save_path).await {
                    eprintln!("warning: failed to save RAG index on shutdown: {e}");
                } else {
                    println!("Saved to {save_path:?}.");
                }
            }
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
