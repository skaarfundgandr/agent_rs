//! OpenTelemetry / LangSmith tracing setup.
//!
//! Build a [`TracerHandle`] via [`init_tracing`] to install a global
//! `tracing` subscriber with a `tracing-opentelemetry` layer that exports
//! spans to LangSmith via OTLP/HTTP. Call [`shutdown_tracing`] on shutdown
//! to flush pending spans.
//!
//! ## Environment variables
//!
//! - `LANGSMITH_API_KEY` — `x-api-key` header. **Required** for actual export.
//! - `LANGSMITH_PROJECT` — `Langsmith-Project` header. Default: `"default"`.
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — full OTLP/HTTP traces URL, including the signal path (e.g. `/v1/traces`). Default: `https://api.smith.langchain.com/otel/v1/traces`. Used as-is — no path is appended.
//! - `OTEL_SERVICE_NAME` — OTel `service.name`. Default: `"agent_rs"`.
//! - `RUST_LOG` — controls the `EnvFilter` layer. Set e.g. `RUST_LOG=info` to surface log-level events.
//!
//! Once the subscriber is installed, **rig's existing GenAI spans** (e.g. `invoke_agent`,
//! `chat`, `execute_tool`) export to LangSmith automatically — no call-site changes needed.
//!
//! To also see spans on stdout, install a `tracing_subscriber::fmt` layer
//! yourself (composing a `Registry` that wraps the OTel layer) and call
//! `set_global_default` with that combined subscriber instead of `init_tracing`.

#[cfg(feature = "opentelemetry")]
pub mod conventions;
#[cfg(feature = "opentelemetry")]
pub mod hooks;
#[cfg(feature = "opentelemetry")]
pub mod langsmith;
#[cfg(feature = "opentelemetry")]
pub mod react_spans;

#[cfg(feature = "opentelemetry")]
pub use hooks::LangSmithAgentHook;
#[cfg(feature = "opentelemetry")]
pub use langsmith::{TracerHandle, init_tracing, shutdown_tracing};
#[cfg(feature = "opentelemetry")]
pub use react_spans::LangSmithReActEmitter;
