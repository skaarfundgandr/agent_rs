//! # Secure Database Assistant Example
//!
//! Demonstrates the use of [PermissionPolicy] and [PolicyMap] to enforce
//! security gates. Read operations are auto-approved, while write/update
//! operations prompt the user in the CLI.
//!
//! Run with:
//! ```bash
//! cargo run --example db_assistant
//! ```

use agent_rs::agent::permission::{PermissionPolicy, PolicyMap};
use anyhow::{Result, bail};
use dotenvy::dotenv;
use rig_core::completion::{Prompt, ToolDefinition};
use rig_core::prelude::*;
use rig_core::providers::openai;
use rig_core::tool::Tool;
use serde_json::json;
use std::env;
use std::sync::{Arc, Mutex};

// Simulated database
struct Database {
    users: Mutex<std::collections::HashMap<String, i32>>, // username -> balance
}

// Custom error type
#[derive(Debug, thiserror::Error)]
#[error("Database error: {0}")]
struct DbError(String);

// Read tool args
#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct QueryArgs {
    username: String,
}

// Read tool
struct DbQueryTool {
    db: Arc<Database>,
}

impl Tool for DbQueryTool {
    const NAME: &'static str = "db_query";
    type Error = DbError;
    type Args = QueryArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Query user balance from the database.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "The username to query"
                    }
                },
                "required": ["username"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let users = self
            .db
            .users
            .lock()
            .map_err(|e| e.to_string())
            .map_err(DbError)?;
        if let Some(balance) = users.get(&args.username) {
            Ok(format!("User '{}' balance: ${}", args.username, balance))
        } else {
            Ok(format!("User '{}' not found.", args.username))
        }
    }
}

// Write tool args
#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct WriteArgs {
    username: String,
    amount: i32,
}

// Write tool
struct DbWriteTool {
    db: Arc<Database>,
    policy_map: PolicyMap,
}

impl Tool for DbWriteTool {
    const NAME: &'static str = "db_write";
    type Error = DbError;
    type Args = WriteArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Deduct money or write transactions to user accounts.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "The user to modify"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "The amount to add or deduct (can be negative)"
                    }
                },
                "required": ["username", "amount"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let action_desc = format!(
            "wants to modify balance of user '{}' by ${}",
            args.username, args.amount
        );

        // Evaluate policy for security checks
        println!("\n[DbWriteTool] Checking permission for: {}", action_desc);
        let res = self.policy_map.evaluate(Self::NAME, &action_desc).await;
        if !res.is_allow() {
            return Err(DbError("Permission denied by user".to_string()));
        }

        let mut users = self
            .db
            .users
            .lock()
            .map_err(|e| e.to_string())
            .map_err(DbError)?;
        let balance = users.entry(args.username.clone()).or_insert(100);
        *balance += args.amount;

        Ok(format!(
            "Transaction successful. New balance for '{}': ${}",
            args.username, balance
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let chat_model_name =
        env::var("CHAT_MODEL").unwrap_or_else(|_| "google/gemma-4-e4b".to_string());
    let chat_base_url =
        env::var("CHAT_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let api_key = match env::var("API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => bail!("Missing API_KEY env var in .env (set it to your LLM provider key)"),
    };

    println!("Initializing Secure DB Assistant...");

    // Create DB with some initial mock data
    let mut initial_users = std::collections::HashMap::new();
    initial_users.insert("alice".to_string(), 150);
    initial_users.insert("bob".to_string(), 50);
    let db = Arc::new(Database {
        users: Mutex::new(initial_users),
    });

    // Create a policy map: allow queries automatically, but prompt for writes
    let policy_map =
        PolicyMap::new(PermissionPolicy::AllowAll).tool("db_write", PermissionPolicy::CliPrompt);

    let chat_client = openai::CompletionsClient::builder()
        .base_url(chat_base_url)
        .api_key(api_key)
        .build()?;

    // Register query and write tools
    let agent = chat_client
        .agent(&chat_model_name)
        .tool(DbQueryTool {
            db: Arc::clone(&db),
        })
        .tool(DbWriteTool {
            db: Arc::clone(&db),
            policy_map,
        })
        .preamble(
            "You are a helpful bank database clerk. \
            You can check balances using `db_query`, and deposit/withdraw funds using `db_write`. \
            Execute whatever the user requests.",
        )
        .default_max_turns(5)
        .build();

    // Query request (should run automatically without prompt)
    let prompt1 = "How much money does bob have?";
    println!("\nRequest 1: \"{}\"", prompt1);
    let response1 = agent.prompt(prompt1).await?;
    println!("Response 1:\n{}", response1);

    // Write request (should prompt the user)
    let prompt2 = "Deduct $20 from alice's balance.";
    println!("\nRequest 2: \"{}\"", prompt2);
    let response2 = agent.prompt(prompt2).await?;
    println!("Response 2:\n{}", response2);

    Ok(())
}
