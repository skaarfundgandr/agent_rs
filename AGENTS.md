# AgentRS — Agent Guide

## Project Snapshot

Rust AI agent framework (edition 2024, v0.2.0). Library crate with examples.
Library consumers import as `agent_rs_lib` (not `agent_rs`).

Core deps: `rig-core` 0.36 (with `rmcp` feature), `rmcp` 1.6, `tokio` (full), `reqwest`, `pdf-extract`.

## Commands

```bash
cargo build --release          # LTO + strip enabled in release profile
cargo check                    # fast compile check
cargo test                     # run all tests (some #[ignore]d — see below)
cargo test -- --include-ignored # run ignored tests too (requires local PDF files)
cargo clippy                   # lint (no custom clippy.toml)
cargo fmt                      # format code
cargo doc --open               # local API docs
cargo run --example cli_chatbot # run the CLI chatbot example (requires --features rag)
```

!NOTE: Some commands are automatically truncated by `rtk` and `aft` e.g. `cargo test`, `git diff`.

No CI pipeline, no pre-commit hooks, no rustfmt.toml — use `cargo fmt` with defaults.

## Running the Example

Requires `.env` with `API_KEY` and `mcp.json` (copy from `mcp.json.example`).
`CHAT_MODEL` (default `google/gemma-4-e4b`) selects the chat model via the OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1`.
`FASTEMBED_MODEL` (default `Xenova/bge-small-en-v1.5`) selects the local fastembed embedding model. First run downloads from Hugging Face (~50MB for BGESmall, larger for others). Set `FASTEMBED_CACHE_DIR` to use a pre-populated cache.
`RAG_DB_PATH` / `RAG_INDEX_PATH` (defaults `./rag_data/rag.db`, `./rag_data/rag.tvim`) — the SQLite + turbovec on-disk artifacts. They must stay in sync; deleting both is the recovery procedure if `open_or_create` errors with "out of sync".
The old `EMBEDDING_MODEL` env var (which used the OpenAI-compatible endpoint for embeddings) is removed. Only `CHAT_MODEL` still uses that endpoint.
turbovec requires AVX2 on x86_64. Apple Silicon and ARM64 Linux work via the SSE/NEON fallback paths the crate provides.
`SANDBOX_ROOTS` — comma-separated list of allowed filesystem paths (first is primary, default for writes). Example: `SANDBOX_ROOTS="./,/tmp/shared,/home/user/docs"`

## Architecture

```
examples/
└── cli_chatbot.rs           # CLI chatbot example wiring
src/
├── lib.rs               # re-exports: agent, config, domain, mcp
├── config.rs            # McpConfig loader + validation
├── security/
│   └── sandbox.rs       # SandboxConfig, SharedSandbox, validate_sandboxed_path, find_containing_root, relative_display_path (+ _shared variants)
├── agent/
│   ├── embeddings.rs    # EmbeddingService<M> — generic over Rig EmbeddingModel
│   ├── rag.rs           # DocumentLoader, WordSplitter, RagPipeline (in-memory only)
│   ├── permission.rs    # PermissionPolicy enum + PermissionGate trait
│   ├── memory/          # ContextManagedAgent — token estimation + auto-summarize
│   ├── tools/           # Read/Write/Grep/Glob/ListDir/ManageRag/Compact tools
│   └── react.rs         # PLACEHOLDER — commented out in mod.rs, not compiled
├── mcp/
│   ├── client.rs        # McpClient — config → connect → tools
│   └── registry.rs      # McpRegistry — stdio/HTTP transport, tool dedup, keepalive
└── domain/              # pure data types + errors (no business logic)
    ├── config.rs        # McpConfig struct
    ├── mcp.rs           # transport specs, server defs
    ├── rag.rs           # Document, Chunk, RagSource types
    └── errors.rs        # DocumentError, CompactError (thiserror)
```

Key wiring: `McpClient` → `McpRegistry` → `McpRegistryRuntime` → `Vec<Box<dyn ToolDyn>>`.
Internal tools (filesystem, RAG, compact) are added to the same tool vec in `cli_chatbot.rs`.

## Testing

Tests in `tests/` (13 files): `agents_tests.rs`, `document_store.rs`, `embeddings.rs`, `manage_rag.rs`, `mcp_client.rs`, `mcp_registry.rs`, `permission.rs`, `rag.rs`, `sandbox_tests.rs`, `shared_sandbox.rs`, `tool_tests.rs`, `turbo_index.rs` (plus `mod.rs`).
- `test_read_pdf` in `tool_tests.rs` is `#[ignore]` — needs a local PDF file, will fail in CI.
- Tool tests use `tempfile` for sandbox isolation.
- MCP tests need live MCP servers or will fail — not safe to run blindly.

## Conventions

- All fallible operations use `anyhow::Result`. Domain errors use `thiserror`.
- Tools enforce sandbox via `security::sandbox::validate_sandboxed_path`: path traversal (`../`) returns `DocumentError::SandboxEscape`.
- Tools that enforce sandbox hold an `Arc<SharedSandbox>` rather than a `SandboxConfig`. `SharedSandbox` supports incremental `add_root` / `remove_root` / `add_roots` / `contains_root` for per-root changes; `set` remains the full-replacement escape hatch when swapping the whole config.
- MCP tool name deduplication: duplicate names across servers cause a hard error at connect time.
- `react.rs` is a stub — do not import from it. The `// pub mod react;` line in `agent/mod.rs` confirms it's excluded.
- `RagStoreBuilder` was removed in v0.2.0 — use `PdfLoader` + `WordSplitter` + `RagPipeline` instead.
- RAG persistence: `RagPipeline` stores chunk metadata in SQLite and vectors in turbovec (`.tvim`). Both files must stay in sync; delete both to recover from "out of sync" errors.
- The `rag` feature gates all RAG code. Without it, RAG types are compiled out entirely.

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
- `ROADMAP.md` — project roadmap and known gaps
