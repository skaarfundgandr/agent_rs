# AgentRS — Agent Guide

## Project Snapshot

Rust AI agent framework (edition 2024, v0.6.0). Library crate with examples.
Library consumers import as `agent_rs`.

Core deps: `rig-core` 0.38.2 (with `rmcp` feature), `rmcp` 1.7, `tokio` (full), `reqwest`, `pdf-extract`, `tracing` 0.1.
Optional deps (feature-gated): `opentelemetry` 0.32, `opentelemetry_sdk` 0.32, `opentelemetry-otlp` 0.32, `tracing-opentelemetry` 0.33, `tracing-subscriber` 0.3 (the `opentelemetry` feature).

## Commands

```bash
cargo build --release          # LTO + strip enabled in release profile
cargo check                    # fast compile check
cargo test                     # run all tests (some #[ignore]d — see below)
cargo test -- --include-ignored # run ignored tests too (requires local PDF files)
cargo clippy                   # lint (no custom clippy.toml)
cargo fmt                      # format code
cargo run --example cli_chatbot --features rag # run the CLI chatbot example
```

CI pipeline and release workflows live in `.github/workflows/`. No pre-commit hooks, no rustfmt.toml — use `cargo fmt` with defaults.

## Running the Example

Requires `.env` with `API_KEY` and `mcp.json` (copy from `mcp.json.example`).
`CHAT_MODEL` (default `google/gemma-4-e4b`) selects the chat model via the OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1` (overridable via `CHAT_BASE_URL`).
`FASTEMBED_MODEL` (default `Xenova/bge-small-en-v1.5`) selects the local fastembed embedding model. First run downloads from Hugging Face (~50MB for BGESmall, larger for others). Set `FASTEMBED_CACHE_DIR` to use a pre-populated cache.
`RAG_DB_PATH` / `RAG_INDEX_PATH` (defaults `./rag_data/rag.db`, `./rag_data/rag.tvim`) — the SQLite + turbovec on-disk artifacts. They must stay in sync; deleting both is the recovery procedure if `open_or_create` errors with "out of sync".
The old `EMBEDDING_MODEL` env var (which used the OpenAI-compatible endpoint for embeddings) is removed. Only `CHAT_MODEL` still uses that endpoint.
turbovec requires AVX2 on x86_64. Apple Silicon and ARM64 Linux work via the SSE/NEON fallback paths the crate provides.
`SANDBOX_ROOTS` — comma-separated list of allowed filesystem paths (first is primary, default for writes). Example: `SANDBOX_ROOTS="./,/tmp/shared,/home/user/docs"`

### LangSmith ReAct example

Requires `.env` with `API_KEY`, `CHAT_MODEL` (e.g. `google/gemma-4-e4b`), and `LANGSMITH_API_KEY`. Optional: `LANGSMITH_PROJECT` (default `default`).

```bash
cargo run --example langsmith_react --features opentelemetry
```

This example is feature-gated on `opentelemetry` and independent of `rag`. It demonstrates the ReAct (Reasoning + Acting) loop with per-cycle OpenTelemetry spans exported to LangSmith. No fastembed/first-run download is required.

Required env vars: `API_KEY`, `LANGSMITH_API_KEY`. Optional: `CHAT_MODEL`, `LANGSMITH_PROJECT`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `RUST_LOG`, `SANDBOX_ROOTS`.

## Architecture

```
examples/
├── cli_chatbot.rs           # CLI chatbot example wiring
└── langsmith_react.rs       # LangSmith ReAct + OTel example (requires --features opentelemetry)
src/
├── lib.rs               # re-exports: agent, config, domain, mcp
├── config.rs            # McpConfig loader + validation
├── security/
│   └── sandbox.rs       # SandboxConfig, SharedSandbox, validate_sandboxed_path, find_containing_root, relative_display_path (+ _shared variants)
├── agent/
│   ├── agents.rs         # strip_reasoning_from_history
│   ├── managed.rs        # ManagedExt, ManagedBuilder, BuiltManagedAgent, ManagedStream
│   ├── embeddings.rs     # EmbeddingService<M> — generic over Rig EmbeddingModel
│   ├── permission.rs     # PermissionPolicy enum + PermissionGate trait
│   ├── dispatch/         # AgentDefinition trait + AgentDispatcher + ReAct/Managed adapters
│   ├── state/            # AgentCheckpoint + save/load helpers
│   ├── memory/           # ContextManager — token estimation + auto-summarize
│   ├── model/            # execute_chat / execute_stream_chat helpers
│   ├── tools/            # Read/Write/Grep/Glob/ListDir/ManageRag/Compact/ToolRegistry tools
│   └── react/            # ReAct loop: builder → built → streaming
│       ├── builder.rs         # ReActBuilder, NoCompaction, CompactionConfig typestates
│       ├── built.rs           # BuiltReAct — prompt(), chat(msg, &mut history), run_loop() orchestrator
│       ├── built_methods.rs   # checkpoint, internal callback helpers
│       ├── streaming.rs       # ReActStream, ReActStreamItem — async Stream impl
│       ├── stream_loop.rs     # spawned streaming loop
│       ├── stream_process.rs  # assistant/tool item processing
│       ├── cycle_compaction.rs# compaction trigger
│       ├── model_call.rs      # model call + retry recovery
│       ├── assistant_parse.rs # assistant content parsing + final-answer detection
│       ├── tool_dispatch.rs   # tool execution loop
│       ├── constants.rs       # REACT_PREAMBLE
│       ├── ext.rs             # ReActExt trait
│       ├── emitter.rs         # ReActSpanEmitter trait, NoopSpanEmitter
│       ├── helpers.rs         # detect_final_answer, tool_error_to_string
│       ├── callbacks.rs       # ThoughtCb, ActionCb, ObservationCb, FinalCb, ErrorCb
│       └── mod.rs             # Re-exports
├── mcp/
│   └── registry/        # McpRegistry — stdio/HTTP transport, tool dedup, keepalive
├── rag/                 # RAG pipeline (in-memory + SQLite/turbovec persistence), feature-gated on `rag`
│   └── pipeline/
│       ├── builder.rs   # RagPipelineBuilder, BuiltRag, RagIndexer (public builder API)
│       ├── state.rs     # RagPipeline struct definition
│       ├── lifecycle.rs # open_or_create (pub(crate)), save, build
│       ├── ingest.rs    # add_source, add_directory, remove_source
│       ├── staging.rs   # add_chunks, add_document, commit_pending (staging API)
│       └── ...          # sync, walker, mod.rs
├── observability/       # OpenTelemetry / LangSmith tracing (feature-gated on opentelemetry)
│   ├── mod.rs           # Re-exports: TracerHandle, init_tracing, shutdown_tracing, LangSmithReActEmitter, LangSmithAgentHook
│   ├── langsmith.rs     # OTLP/HTTP exporter + tracing subscriber wiring
│   ├── conventions.rs   # GenAI/LangSmith attribute string constants
│   ├── react_spans.rs   # LangSmithReActEmitter — ReActSpanEmitter impl for OTel spans
│   └── hooks.rs         # LangSmithAgentHook — PromptHook impl for rig spans
└── domain/              # pure data types + errors (no business logic)
    ├── config.rs        # McpConfig struct
    ├── mcp.rs           # transport specs, server defs
    ├── rag.rs           # Document, Chunk, RagSource types (feature-gated on `rag`)
    ├── agent.rs         # ReAct step types (Thought, Action, Observation, etc.)
    ├── observability.rs # LangSmithConfig (feature-gated on opentelemetry)
    └── errors.rs        # DocumentError, CompactError, ReActError (thiserror)
```

Key wiring: `McpRegistry` → `McpRegistryRuntime` → `ToolRegistry` → `Vec<Box<dyn ToolDyn>>`.
Internal tools (filesystem, RAG, compact) register into `ToolRegistry` alongside MCP tools in `cli_chatbot.rs`.

## Testing

All tests must reside in the `tests/` directory rather than inside `src/`. No unit tests should be placed inline within `src/` to keep production code clean.

Tests in `tests/` (17+ files): `agents_tests.rs`, `document_store.rs`, `embeddings.rs`, `manage_rag.rs`, `mcp_registry.rs`, `observability_tests.rs` (feature-gated on `opentelemetry`), `permission.rs`, `rag.rs`, `react_otel_tests.rs` (feature-gated on `opentelemetry`), `react_recovery_tests.rs`, `react_tests.rs`, `sandbox_tests.rs`, `shared_sandbox.rs`, `tool_tests.rs`, `turbo_index.rs`, `tool_registry.rs`, `dispatch.rs`, `state.rs`, `react_e2e.rs` (plus `common/mod.rs`, `mod.rs`).
- `test_read_pdf` in `tool_tests.rs` is `#[ignore]` — needs a local PDF file, will fail in CI.
- Tool tests use `tempfile` for sandbox isolation.
- MCP tests need live MCP servers or will fail — not safe to run blindly.
- Observability/ReAct-OTel tests install a global tracing subscriber. Because a process can only have one, they guard `init_tracing()` with `OnceLock` or `let _ =` to avoid `set_global_default` panics when other test modules set it first.

## Conventions

- All fallible operations use `anyhow::Result`. Domain errors use `thiserror`.
- Tools enforce sandbox via `security::sandbox::validate_sandboxed_path`: path traversal (`../`) returns `DocumentError::SandboxEscape`.
- Tools that enforce sandbox hold an `Arc<SharedSandbox>` rather than a `SandboxConfig`. `SharedSandbox` supports incremental `add_root` / `remove_root` / `add_roots` / `contains_root` for per-root changes; `set` remains the full-replacement escape hatch when swapping the whole config.
- MCP tool name deduplication: duplicate names across servers cause a hard error at connect time.
- `RagStoreBuilder` was removed in v0.2.0 — use `PdfLoader` + `WordSplitter` + `RagPipeline` instead.
- RAG construction: use `RagPipeline::builder()` → `.embedder(service).store_at(dir).build().await` → `BuiltRag { vector_index, indexer }`. The `RagIndexer` provides `add/remove/list/tool` methods. `open_or_create` and `from_parts` are `pub(crate)`.
- RAG persistence: `RagPipeline` stores chunk metadata in SQLite and vectors in turbovec (`.tvim`). Both files must stay in sync; delete both to recover from "out of sync" errors.
- The `rag` feature gates all RAG code. Without it, RAG types are compiled out entirely.
- The `opentelemetry` feature gates the LangSmith OTel tracing path. Without it, `src/observability/` compiles to nothing and `domain/observability::LangSmithConfig` is `cfg`-out.
- `domain/` holds pure data types + errors only. Behaviour lives in root-level modules (`agent/`, `observability/`). Pure config structs that mirror loader-side modules live in `domain/<topic>.rs` (e.g. `LangSmithConfig` in `domain/observability.rs` mirrors the runtime wiring in `src/observability/langsmith.rs`).

## MCP Config

`mcp.json` structure: `{ "mcpServers": { "<name>": { "type": "stdio"|"http", ... } } }`.
- Stdio: `command`, `args`, `env`, `cwd`
- HTTP: `url`, `headers`
- Transport type aliases accepted: `"http"`, `"jsonrpc"`, `"json-rpc"`, `"streamable_http"`
- File is gitignored. Example at `mcp.json.example`.

## Docs

- `docs/api/` — API reference docs (split by section). See [API Reference Overview](docs/api/README.md)
- `docs/migration-0.2.0.md` — migration guide
- `docs/diagrams/` — architecture diagrams (C4, class, sequence, state, module dependency)
