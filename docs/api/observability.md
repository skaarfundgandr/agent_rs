# Observability (LangSmith / OpenTelemetry)

The `opentelemetry` Cargo feature wires a `tracing-opentelemetry` layer + OTLP/HTTP
exporter pointed at LangSmith. Once the subscriber is installed, **rig's already-emitted
GenAI spans** (`invoke_agent`, `chat`, `execute_tool`, etc.) export to LangSmith
automatically — no call-site changes are needed.

## Public API (feature-gated on `opentelemetry`)

```rust,ignore
use agent_rs::observability::{
    init_tracing, shutdown_tracing, TracerHandle,
    LangSmithReActEmitter, LangSmithAgentHook,
};
use agent_rs::domain::observability::LangSmithConfig;
```

- **`LangSmithConfig`** (in `domain::observability`) — pure data: `endpoint`,
  `api_key`, `project`, `service_name`, `console`, `batch`, `batch_delay_ms`.
  Constructors:
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
- **`LangSmithAgentHook<M>`** — `rig_core::agent::AgentHook<M>` impl that
  tags rig's `chat` / `execute_tool` spans with `langsmith.span.kind` and
  fills `gen_ai.input.messages` / `gen_ai.output.messages` (which rig
  declares but does not populate). rig 0.40 natively emits
  `gen_ai.operation.name`, `gen_ai.usage.*`, and `gen_ai.tool.name` — the
  hook no longer records these to avoid duplication.

## Environment Variables

| Var | Field | Default |
|---|---|---|
| `LANGSMITH_API_KEY` | `api_key` | `""` (must be set for export) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `endpoint` | `https://api.smith.langchain.com/otel/v1/traces` (full URL — used as-is) |
| `LANGSMITH_PROJECT` | `project` | `"default"` |
| `OTEL_SERVICE_NAME` | `service_name` | `"agent_rs"` |
| `LANGSMITH_OTEL_CONSOLE` | `console` | `false` (reserved for future use) |
| `LANGSMITH_OTEL_BATCH` | `batch` | `true` (read into the config, but not consulted by `init_tracing`) |
| `LANGSMITH_OTEL_BATCH_DELAY_MS` | `batch_delay_ms` | `1000` |
| `RUST_LOG` | `EnvFilter` | `info` |

The exporter selection in `init_tracing` keys off **`batch_delay_ms` only** —
the `batch` field is not read:
- `batch_delay_ms = 0` selects a **synchronous simple exporter** (no batching) —
  useful for local development and debugging.
- Any positive value uses a `BatchSpanProcessor` with that scheduled delay
  (default `1000` ms, down from the OTel SDK default of 5000 ms).

The endpoint is passed verbatim to the OTel SDK — no path is appended or
transformed. Set the **full** URL (including the signal path) for the
collector / vendor you're targeting. For LangSmith cloud this is
`https://api.smith.langchain.com/otel/v1/traces`; for a local OTel
collector it's typically `http://localhost:4318/v1/traces`.

The OTLP/HTTP exporter sets the `x-api-key` and `Langsmith-Project` headers
via `WithHttpConfig::with_headers` — no env-var pollution or unsafe.

## GenAI + LangSmith Attribute Conventions

Defined in `src/observability/conventions.rs`:

- **GenAI (OTel, emitted natively by rig 0.40)**: `gen_ai.operation.name`,
  `gen_ai.tool.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`.
  These are recorded by rig on its `invoke_agent`, `chat`, and `execute_tool`
  spans — the hook does NOT record them.
- **GenAI (filled by the hook)**: `gen_ai.input.messages`,
  `gen_ai.output.messages` — rig declares these on the `chat` span but never
  populates them; the hook fills them on the `invoke_agent` span.
- **LangSmith run-typing**: `langsmith.span.kind ∈ { llm, chain, tool, agent, retriever, embedding }`.
- **OpenInference (also recognised by LangSmith)**: `openinference.span.kind`,
  `input.value`, `output.value`.

The ReAct bridge layers `chain` / `agent` / `tool` runs on top of rig's
`chat` / `execute_tool` spans, and the `AgentHook` layer adds LangSmith
run-typing and input/output messages to the same spans.

## Usage Example

```rust,ignore
use agent_rs::observability::{init_tracing, shutdown_tracing, LangSmithReActEmitter, LangSmithAgentHook};
use agent_rs::domain::observability::LangSmithConfig;

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
