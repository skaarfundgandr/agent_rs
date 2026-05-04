use anyhow::{bail, Context, Result};
use dotenvy::dotenv;
use pdf_extract::extract_text;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::openai;
use rs_agent::agent::embeddings::EmbeddingService;
use rs_agent::config::McpConfig;
use rs_agent::mcp::client::McpClient;
use rig::{vector_store::in_memory_store::InMemoryVectorStore, OneOrMany};
use std::env;
use std::path::Path;

const EMBEDDING_MODEL: &str = "text-embedding-embeddinggemma-300m-qa";
const CHAT_MODEL: &str = "google/gemma-4-e4b";
const CHUNK_WORDS: usize = 220;
const CHUNK_OVERLAP_WORDS: usize = 40;
const RAG_TOP_K: usize = 4;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let client = openai::Client::builder()
        .base_url("http://127.0.0.1:1234/v1")
        .api_key(env::var("API_KEY").expect("Missing API_KEY env var"))
        .build()?;

    let embedding_model = client.embedding_model(EMBEDDING_MODEL);
    let embedding_service = EmbeddingService::new(embedding_model.clone());

    let mcp_client = McpClient::new(McpConfig::from_path("./mcp.json").unwrap())
        .tools()
        .await?;

    let rag_documents = build_rag_documents(&[
        "./Stellaron Architecture Overview.pdf",
        "./Orientation-ASEAN-AI-HACKATHON-14.4.2026.pdf",
    ])?;

    let embeddings = embedding_service.embed_texts(rag_documents.clone()).await?;
    let mut vector_store = InMemoryVectorStore::<String>::default();
    vector_store.add_documents(
        rag_documents
            .into_iter()
            .zip(embeddings)
            .map(|(document, embedding)| (document, OneOrMany::one(embedding))),
    );

    let index = vector_store.index(embedding_model);

    let agent = client
        .agent(CHAT_MODEL)
        .tools(mcp_client)
        .preamble(
            "You are a helpful RAG assistant. Answer using the retrieved PDF context first, cite\n\
             the source chunk when possible, and say when the documents do not contain enough\n\
             information to answer confidently.",
        )
        .dynamic_context(RAG_TOP_K, index)
        .temperature(0.6)
        .build();

    let chatbot = ChatBotBuilder::new()
        .agent(agent)
        .build();

    Ok(chatbot.run().await?)
}

fn build_rag_documents(paths: &[&str]) -> Result<Vec<String>> {
    let mut documents = Vec::new();

    for path in paths {
        let source = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);

        let pdf_text = extract_pdf_text(path)?;
        let chunks = chunk_text(&pdf_text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS);

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            if chunk.trim().is_empty() {
                continue;
            }

            documents.push(format!(
                "[source: {source} | chunk: {chunk_idx}]\n{chunk}"
            ));
        }
    }

    if documents.is_empty() {
        bail!("no embeddable text was extracted from the provided PDFs");
    }

    Ok(documents)
}

fn chunk_text(text: &str, max_words: usize, overlap_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return Vec::new();
    }

    let max_words = max_words.max(1);
    let overlap_words = overlap_words.min(max_words.saturating_sub(1));
    let step = max_words - overlap_words;

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + max_words).min(words.len());
        let chunk = words[start..end].join(" ");
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }

        if end == words.len() {
            break;
        }

        start += step;
    }

    chunks
}

pub fn extract_pdf_text<P: AsRef<Path>>(path: P) -> Result<String> {
    extract_text(path).context("Failed to extract text from PDF")
}