use anyhow::Result;
use dotenvy::dotenv;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::ToolDyn;
use rs_agent::agent::embeddings::EmbeddingService;
use rs_agent::agent::rag::{ChunkingOptions, RagStoreBuilder};
use rs_agent::agent::tools::{ReadDocumentTool, WriteDocumentTool};
use rs_agent::config::McpConfig;
use rs_agent::mcp::client::McpClient;
use std::env;

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

    let mut tools = McpClient::new(McpConfig::from_path("./mcp.json").unwrap())
        .tools()
        .await?;

    // Build RAG index using the new ergonomic builder from the agent module
    let index = RagStoreBuilder::new(embedding_service)
        .with_chunking(ChunkingOptions {
            chunk_words: 220,
            chunk_overlap_words: 40,
        })
        .add_pdf("./Orientation-ASEAN-AI-HACKATHON-14.4.2026.pdf")?
        .build_index()
        .await?;

    let internal_tools: Vec<Box<dyn ToolDyn>> =
        vec![Box::new(ReadDocumentTool), Box::new(WriteDocumentTool)];
    tools.extend(internal_tools);

    let agent = client
        .agent(&chat_model_name)
        .tools(tools)
        .preamble(
            "You are a helpful AI agent \
            capable of RAG and reading documents \
            using internal tools.",
        )
        .dynamic_context(rag_top_k, index)
        .temperature(0.6)
        .build();

    let chatbot = ChatBotBuilder::new().agent(agent).build();

    chatbot.run().await?;

    Ok(())
}
