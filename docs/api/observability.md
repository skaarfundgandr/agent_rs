# Observability (LangSmith / OpenTelemetry)

The `opentelemetry` Cargo feature wires a `tracing-opentelemetry` layer + OTLP/HTTP
exporter pointed at LangSmith. Once the subscriber is installed, **rig's already-emitted
GenAI spans** (`invoke_agent`, `chat`, `execute_tool`, etc.) export to LangSmith
automatically — no call-site changes are needed.

## Public API (feature-gated on `opentelemetry`)

```rust,ignore
use agent_rs_lib::observability::{
    init_tracing, shutdown_tracing, TracerHandle,
    LangSmithReActEmitter, LangSmithAgentHook,
};
use agent_rs_lib::domain::observability::LangSmithConfig;
```

- **`LangSmithConfig`** (in `domain::observability`) — pure data: `endpoint`,
  `api_key`, `project`, `service_name`, `console`, `batch`. Constructors:
  - `LangSmithConfig::default()` — LangSmith cloud defaults; `api_key` is empty.
  - `LangSmithConfig::from_env()` — read from conventional env vars.
  - `LangSmithConfig::from_env_or_default(key_env)` — like `from_env` but
    requires the API key env var to be set (else returns `Err`).
- **`TracerHandle`** — owns the `SdkTracerProvider`; calling `shutdown()`
  (or dropping) flushes pending spans.
- **`init_tracing(&cfg) -> Result<TracerHandle>`** — installs a global
  `tracing` subscriber with the OTel layer. Returns an error if a global
  subscriber has already been installed.
- **`shutdown_tracing(handle) -> Result<()>`** — async; flush + shut down.
- **`LangSmithReActEmitter`** — `ReActSpanEmitter` impl that records
  LangSmith run-typing on the current `tracing` span.
- **`LangSmithAgentHook<M>`** — `rig_core::agent::PromptHook<M>` impl that
  tags rig's `chat` / `execute_tool` spans with `langsmith.span.kind` and
  records token-usage counters.

## Environment Variables

| Var | Field | Default |
|---|---|---|
| `LANGSMITH_API_KEY` | `api_key` | `""` (must be set for export) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `endpoint` | `https://api.smith.langchain.com/otel/v1/traces` (full URL — used as-is) |
| `LANGSMITH_PROJECT` | `project` | `"default"` |
| `OTEL_SERVICE_NAME` | `service_name` | `"agent_rs"` |
| `LANGSMITH_OTEL_CONSOLE` | `console` | `false` (reserved for future use) |
| `LANGSMITH_OTEL_BATCH` | `batch` | `true` (currently always batch) |
| `RUST_LOG` | `EnvFilter` | `info` |

The endpoint is passed verbatim to the OTel SDK — no path is appended or
transformed. Set the **full** URL (including the signal path) for the
collector / vendor you're targeting. For LangSmith cloud this is
`https://api.smith.langchain.com/otel/v1/traces`; for a local OTel
collector it's typically `http://localhost:4318/v1/traces`.

The OTLP/HTTP exporter sets the `x-api-key` and `Langsmith-Project` headers
via `WithHttpConfig::with_headers` — no env-var pollution or unsafe.

## GenAI + LangSmith Attribute Conventions

Defined in `src/observability/conventions.rs`:

- **GenAI (OTel)**: `gen_ai.system`, `gen_ai.operation.name`, `gen_ai.tool.name`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`.
- **LangSmith run-typing**: `langsmith.span.kind ∈ { llm, chain, tool, agent, retriever, embedding }`.
- **OpenInference (also recognised by LangSmith)**: `openinference.span.kind`,
  `input.value`, `output.value`.

The ReAct bridge layers `chain` / `agent` / `tool` runs on top of rig's
`chat` / `execute_tool` spans, and the `PromptHook` layer records token
usage onto the same spans.

## Usage Example

```rust,ignore
use agent_rs_lib::observability::{init_tracing, shutdown_tracing, LangSmithReActEmitter, LangSmithAgentHook};
use agent_rs_lib::domain::observability::LangSmithConfig;

let cfg = LangSmithConfig::from_env_or_default("LANGSMITH_API_KEY")?;
let handle = init_tracing(&cfg)?;

// ... run agents, ReAct loops, etc. ...

shutdown_tracing(handle).await?;
```

The `examples/langsmith_react.rs` example wires all of this end-to-end with
a minimal read-only filesystem toolset.

## Regional LangSmith Endpoints

- US GCP: `https://api.smith.langchain.com/otel`
- EU GCP: `https://eu.api.smith.langchain.com/otel`
- APAC GCP: `https://apac.api.smith.langchain.com/otel`
- AWS US: `https://aws.api.smith.langchain.com/otel`
- Self-hosted: `https://<your-host>/api/v1/otel`

Set via `OTEL_EXPORTER_OTLP_ENDPOINT` env var or `LangSmithConfig::endpoint`.
