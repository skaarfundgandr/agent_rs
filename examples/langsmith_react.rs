//! # LangSmith ReAct CLI Example
//!
//! Demonstrates the ReAct (Reasoning + Acting) loop with OpenTelemetry
//! trace export to LangSmith in an interactive CLI session.
//!
//! ## Required env vars
//!
//! - `API_KEY` — OpenAI-compatible chat endpoint key (default endpoint: `http://127.0.0.1:1234/v1`)
//! - `CHAT_BASE_URL` — OpenAI-compatible chat endpoint base URL (default: `http://127.0.0.1:1234/v1`)
//! - `CHAT_MODEL` — model identifier (default: `google/gemma-4-e4b`)
//! - `LANGSMITH_API_KEY` — LangSmith API key (`ls_...`)
//! - `LANGSMITH_PROJECT` — LangSmith project name (default: `default`)
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP/HTTP base URL (default: `https://api.smith.langchain.com/otel`)
//! - `OTEL_SERVICE_NAME` — OTel `service.name` (default: `agent_rs`)
//! - `RUST_LOG` — tracing env-filter (e.g. `RUST_LOG=info`)
//! - `SANDBOX_ROOTS` — comma-separated allowed filesystem roots (default: `./`)
//!
//! ## Run
//!
//! ```bash
//! LANGSMITH_API_KEY=ls_***  LANGSMITH_PROJECT=agent_rs-react  \
//!   cargo run --example langsmith_react --features opentelemetry
//! ```

#![cfg_attr(not(feature = "opentelemetry"), allow(unused))]

#[cfg(feature = "opentelemetry")]
mod otel_main {
    use agent_rs_lib::agent::ReActExt;
    use agent_rs_lib::agent::permission::PermissionPolicy;
    use agent_rs_lib::agent::tools::{
        GlobSearchTool, GrepSearchTool, ListDirectoryTool, ReadDocumentTool,
    };
    use agent_rs_lib::config::McpConfig;
    use agent_rs_lib::domain::observability::LangSmithConfig;
    use agent_rs_lib::mcp::client::McpClient;
    use agent_rs_lib::observability::{
        LangSmithAgentHook, LangSmithReActEmitter, init_tracing, shutdown_tracing,
    };
    use agent_rs_lib::security::{SandboxConfig, SharedSandbox};
    use anyhow::{Result, bail};
    use dotenvy::dotenv;
    use rig_core::client::CompletionClient;
    use rig_core::providers::openai;
    use rig_core::tool::ToolDyn;
    use std::collections::HashSet;
    use std::env;
    use std::io::Write;
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
        let api_key = match env::var("API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => bail!("Missing API_KEY env var (set it to your LLM provider key)"),
        };

        // ---------- LangSmith / OTel ----------
        let langsmith_cfg = LangSmithConfig::from_env_or_default("LANGSMITH_API_KEY")?;
        let handle = init_tracing(&langsmith_cfg)?;
        eprintln!(
            "Tracing initialized — open the LangSmith UI to view your '{}' project traces.",
            langsmith_cfg.project
        );

        // ---------- startup banner ----------
        eprintln!("--- langsmith_react configuration ---");
        eprintln!("  CHAT_BASE_URL      = {chat_base_url}");
        eprintln!("  CHAT_MODEL         = {chat_model_name}");
        eprintln!("  LANGSMITH_PROJECT  = {}", langsmith_cfg.project);
        eprintln!("  OTEL endpoint      = {}", langsmith_cfg.endpoint);
        eprintln!("  OTEL service.name  = {}", langsmith_cfg.service_name);
        if chat_base_url == "http://127.0.0.1:1234/v1" {
            eprintln!(
                "  WARNING: CHAT_BASE_URL is the local default (127.0.0.1:1234). \
                 Set CHAT_BASE_URL in .env or the environment to point at your provider."
            );
        }
        eprintln!("-------------------------------------");

        // ---------- chat client (OpenAI-compatible local endpoint) ----------
        let chat_client = openai::CompletionsClient::builder()
            .base_url(chat_base_url)
            .api_key(api_key)
            .build()?;

        // ---------- sandbox ----------
        let shared_sandbox = build_shared_sandbox()?;

        // ---------- permission policy ----------
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

        // ---------- internal tools (read-only surface, no rag/fastembed) ----------
        let read_extensions = HashSet::from(["txt", "md"].map(String::from));
        let grep_extensions = HashSet::from(["txt", "md"].map(String::from));

        let internal_tools: Vec<Box<dyn ToolDyn>> = vec![
            Box::new(ReadDocumentTool::new(
                Arc::clone(&shared_sandbox),
                read_extensions,
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
        ];
        let mut tools: Vec<Box<dyn ToolDyn>> = internal_tools;

        // ---------- optional MCP tools ----------
        match McpConfig::from_path("./mcp.json") {
            Ok(cfg) => match McpClient::new(cfg).tools(policy.clone()).await {
                Ok(mcp_tools) => {
                    eprintln!("Loaded {} MCP tools from ./mcp.json", mcp_tools.len());
                    tools.extend(mcp_tools);
                }
                Err(e) => {
                    eprintln!("Warning: failed to connect MCP servers: {e}");
                }
            },
            Err(_) => {
                eprintln!("No ./mcp.json found, skipping MCP tools.");
            }
        }

        // ---------- build agent ----------
        let agent = chat_client
            .agent(&chat_model_name)
            .tools(tools)
            .default_max_turns(20)
            .temperature(0.6)
            .hook(LangSmithAgentHook)
            .build();

        // ---------- interactive prompt loop ----------
        println!("LangSmith ReAct CLI Chatbot. Type 'exit' or 'quit' to end.");

        use agent_rs_lib::observability::conventions::KIND_AGENT;
        use tracing::Instrument;

        let react = agent
            .react()
            .max_cycles(20)
            .react_preamble(None)
            .with_span_emitter(Arc::new(LangSmithReActEmitter))
            .on_action(|a| eprintln!("→ action: {}", a.tool_name))
            .on_observation(|o| {
                eprintln!(
                    "← obs: {} ({} bytes, err={})",
                    o.tool_name,
                    o.result.len(),
                    o.is_error
                )
            })
            .on_final(|f| {
                eprintln!("\n✓ final answer ({} cycles):\n{}\n", f.cycles, f.text);
                tracing::Span::current().record("output.value", f.text.as_str());
            })
            .with_compaction()
            .compaction_model(agent.clone())
            .threshold(128_000)
            .build();

        loop {
            print!("\nreact> ");
            let _ = std::io::stdout().flush();
            let mut prompt = String::new();
            if std::io::stdin().read_line(&mut prompt).is_err() {
                break;
            }
            let prompt = prompt.trim();
            if prompt.is_empty() {
                continue;
            }
            if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
                break;
            }

            // ---------- run ReAct loop ----------
            let parent_span = tracing::info_span!(
                "react_agent",
                "langsmith.span.kind" = KIND_AGENT,
                "openinference.span.kind" = "AGENT",
                "input.value" = prompt,
                "output.value" = tracing::field::Empty,
            );

            let trace = async { react.chat_compact(prompt).await }
                .instrument(parent_span)
                .await;

            let answer = match trace {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("ReAct loop error: {e}");
                    continue;
                }
            };

            // ---------- print answer ----------
            let mut stdout = std::io::stdout();
            writeln!(stdout, "{answer}")?;
            stdout.flush()?;
        }

        // ---------- shutdown ----------
        shutdown_tracing(handle).await?;

        Ok(())
    }
}

#[cfg(feature = "opentelemetry")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    otel_main::run().await
}

#[cfg(not(feature = "opentelemetry"))]
fn main() {
    eprintln!("langsmith_react example requires --features opentelemetry");
}
