use dotenvy::dotenv;
use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openai;
use rs_agent::config::McpConfig;
use rs_agent::mcp::client::McpClient;
use std::env;

#[tokio::main]
async fn main() {
    dotenv().ok();

    let client = openai::Client::builder()
        .base_url("http://127.0.0.1:1234/v1")
        .api_key(env::var("API_KEY").expect("Missing API_KEY env var"))
        .build()
        .unwrap();

    let mcp_client = McpClient::new(McpConfig::from_path("./mcp.json").unwrap())
        .tools()
        .await
        .unwrap();

    let agent = client
        .agent("google/gemma-4-e4b")
        .tools(mcp_client)
        .preamble("You are an ai agent")
        .temperature(0.6)
        .build();

    let response = agent
        .prompt("Use context7 to fetch documentation for React Native Router")
        .await
        .unwrap();

    println!("Response: {}", response);
}