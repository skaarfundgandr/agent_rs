use agent_rs_lib::agent::embeddings::EmbeddingService;
use agent_rs_lib::agent::permission::PermissionPolicy;
use agent_rs_lib::agent::rag::{DocumentLoader, PdfLoader, RagPipeline, WordSplitter};
use agent_rs_lib::agent::tools::{
    CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ManageRagTool,
    RagSourceRegistry, ReadDocumentTool, WriteDocumentTool,
};
use agent_rs_lib::config::McpConfig;
use agent_rs_lib::mcp::client::McpClient;
use agent_rs_lib::security::SandboxConfig;
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

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let embedding_model_name = env::var("EMBEDDING_MODEL")
        .unwrap_or_else(|_| "text-embedding-embeddinggemma-300m-qa".to_string());
    let chat_model_name =
        env::var("CHAT_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());
    let rag_top_k = env::var("RAG_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let client = openai::Client::builder()
        .base_url("http://127.0.0.1:1234/v1")
        .api_key(env::var("API_KEY").expect("Missing API_KEY env var"))
        .build()?;

    let embedding_model = client.embedding_model(embedding_model_name);
    let embedding_service = EmbeddingService::new(embedding_model.clone());

    // RAG source registry shared across the manage_rag tool
    let rag_extensions = HashSet::from(["txt", "md", "pdf"].map(String::from));
    let rag_registry = Arc::new(Mutex::new(RagSourceRegistry::new(rag_extensions)));

    let mut tools = McpClient::new(McpConfig::from_path("./mcp.json").unwrap())
        .tools()
        .await?;

    let compaction_agent = client
        .agent(&chat_model_name)
        .default_max_turns(20)
        .preamble("You are a summarization assistant.")
        .build();

    let compaction_tool = CompactTool::new(compaction_agent);

    // Build RAG index using the new decoupled pipeline
    let pdf_document = PdfLoader::new().load(std::path::Path::new(
        "./Orientation-ASEAN-AI-HACKATHON-14.4.2026.pdf",
    ))?;
    let splitter = WordSplitter::new(220, 40);

    let index = RagPipeline::new()
        .add_document(&pdf_document, &splitter)
        .build_index(&embedding_service)
        .await?;

    let read_extensions = HashSet::from(["txt", "md", "pdf"].map(String::from));
    let write_extensions = HashSet::from(["txt", "md"].map(String::from));

    let grep_extensions = HashSet::from(["txt", "md"].map(String::from));

    let policy = match env::var("PERMISSION_POLICY").as_deref() {
        Ok("deny") => PermissionPolicy::DenyAll,
        Ok("prompt") => PermissionPolicy::CliPrompt,
        _ => PermissionPolicy::AllowAll,
    };

    // Sandbox configuration: SANDBOX_ROOTS env var overrides default "./"
    // Format: comma-separated paths, first is primary (default for writes)
    // Example: SANDBOX_ROOTS="./,/tmp/shared,/home/user/docs"
    let sandbox = match env::var("SANDBOX_ROOTS") {
        Ok(s) if !s.trim().is_empty() => {
            SandboxConfig::new(s.split(',').map(|p| PathBuf::from(p.trim())).collect())?
        }
        _ => SandboxConfig::single("./")?,
    };

    let internal_tools: Vec<Box<dyn ToolDyn>> = vec![
        Box::new(ReadDocumentTool::new(
            sandbox.clone(),
            read_extensions,
            policy.clone(),
        )),
        Box::new(WriteDocumentTool::new(
            sandbox.clone(),
            write_extensions,
            policy.clone(),
        )),
        Box::new(ListDirectoryTool::new(sandbox.clone(), policy.clone())),
        Box::new(GrepSearchTool::new(
            sandbox.clone(),
            grep_extensions,
            policy.clone(),
        )),
        Box::new(GlobSearchTool::new(sandbox.clone(), policy.clone())),
        Box::new(ManageRagTool::new(rag_registry, sandbox, policy)),
    ];
    tools.extend(internal_tools);

    tools.push(Box::new(compaction_tool));

    let agent = client
        .agent(&chat_model_name)
        .tools(tools)
        .preamble(
            "You are a helpful AI agent \
            capable of RAG and reading documents \
            using internal tools.",
        )
        .dynamic_context(rag_top_k, index)
        .default_max_turns(20)
        .temperature(0.6)
        .build();

    let chatbot = ChatBotBuilder::new().agent(agent).build();

    chatbot.run().await?;

    Ok(())
}
