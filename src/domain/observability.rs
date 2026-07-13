#![cfg(feature = "opentelemetry")]

//! Pure configuration types for LangSmith OpenTelemetry export.
//!
//! No SDK types (no `SpanExporter`, `TracerProvider`) live in this module —
//! those are runtime concerns of `src/observability/`.

use serde::{Deserialize, Serialize};

/// Configuration for LangSmith OTLP/HTTP trace export.
///
/// Use [`LangSmithConfig::from_env`] to read from conventional environment
/// variables, or [`Default::default`] for sensible defaults (with an empty
/// API key that you must fill in).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangSmithConfig {
    /// Full OTLP/HTTP traces URL, including the signal path
    /// (e.g. `/v1/traces`). Default: `https://api.smith.langchain.com/otel/v1/traces`.
    ///
    /// The endpoint is used **as-is** — no path is appended. This is the
    /// behavior of the OTel Rust SDK's `with_endpoint()` (programmatic
    /// config) — unlike the env-var lookup path which auto-appends the
    /// signal path. If you set `OTEL_EXPORTER_OTLP_ENDPOINT` to a bare
    /// base URL (e.g. `https://api.smith.langchain.com/otel`) and the
    /// SDK doesn't append the path for you, supply the full traces URL
    /// explicitly.
    pub endpoint: String,
    /// `x-api-key` header value. Required for actual export.
    pub api_key: String,
    /// `Langsmith-Project` header value. Defaults to `"default"`.
    pub project: String,
    /// OTel `service.name` resource attribute. Default: `"agent_rs"`.
    pub service_name: String,
    /// Also install a `tracing_subscriber::fmt` layer for console output.
    pub console: bool,
    /// Use batch processor (true) vs simple (false). Batch for prod, simple for tests.
    pub batch: bool,
    /// Batch span processor scheduled delay in milliseconds. `0` selects a
    /// synchronous simple exporter. Default: `1000`. Overridden by
    /// `LANGSMITH_OTEL_BATCH_DELAY_MS` when that env var is set.
    pub batch_delay_ms: u64,
}

impl Default for LangSmithConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.smith.langchain.com/otel/v1/traces".to_string(),
            api_key: String::new(),
            project: "default".to_string(),
            service_name: "agent_rs".to_string(),
            console: false,
            batch: true,
            batch_delay_ms: 1000,
        }
    }
}

impl LangSmithConfig {
    /// Read configuration from environment variables with conventional names.
    ///
    /// | Env var | Field | Default |
    /// |---|---|---|
    /// | `LANGSMITH_API_KEY` | `api_key` | `""` (must be set for export) |
    /// | `OTEL_EXPORTER_OTLP_ENDPOINT` | `endpoint` | `https://api.smith.langchain.com/otel` |
    /// | `LANGSMITH_PROJECT` | `project` | `"default"` |
    /// | `OTEL_SERVICE_NAME` | `service_name` | `"agent_rs"` |
    /// | `LANGSMITH_OTEL_CONSOLE` | `console` | `false` |
    /// | `LANGSMITH_OTEL_BATCH` | `batch` | `true` |
    /// | `LANGSMITH_OTEL_BATCH_DELAY_MS` | `batch_delay_ms` | `1000` |
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(val) = std::env::var("LANGSMITH_API_KEY") {
            cfg.api_key = val;
        }
        if let Ok(val) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            cfg.endpoint = val;
        }
        if let Ok(val) = std::env::var("LANGSMITH_PROJECT") {
            cfg.project = val;
        }
        if let Ok(val) = std::env::var("OTEL_SERVICE_NAME") {
            cfg.service_name = val;
        }
        if let Ok(val) = std::env::var("LANGSMITH_OTEL_CONSOLE") {
            cfg.console = matches!(val.as_str(), "1" | "true" | "yes");
        }
        if let Ok(val) = std::env::var("LANGSMITH_OTEL_BATCH") {
            cfg.batch = !matches!(val.as_str(), "0" | "false" | "no");
        }
        if let Ok(val) = std::env::var("LANGSMITH_OTEL_BATCH_DELAY_MS") {
            if let Ok(ms) = val.parse::<u64>() {
                cfg.batch_delay_ms = ms;
            }
        }

        cfg
    }

    /// Like [`from_env`](Self::from_env) but returns an error if the API key
    /// env var is missing.
    ///
    /// `key_env` is the name of the env var to read the API key from. If
    /// empty, defaults to `"LANGSMITH_API_KEY"`.
    pub fn from_env_or_default(key_env: &str) -> anyhow::Result<Self> {
        let mut cfg = Self::from_env();

        let env_name = if key_env.is_empty() {
            "LANGSMITH_API_KEY"
        } else {
            key_env
        };

        match std::env::var(env_name) {
            Ok(val) if !val.is_empty() => {
                cfg.api_key = val;
            }
            _ => {
                anyhow::bail!("missing env var {env_name}");
            }
        }

        Ok(cfg)
    }
}
