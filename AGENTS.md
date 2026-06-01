# AgentRS — Agent Guide

## Project Snapshot

Rust AI agent framework (edition 2024, v0.2.0). Single crate with binary + library.
Library consumers import as `agent_rs_lib` (not `agent_rs`).

Core deps: `rig-core` 0.36 (with `rmcp` feature), `rmcp` 1.6, `tokio` (full), `reqwest`, `pdf-extract`.

## Commands

```bash
cargo build --release          # LTO + strip enabled in release profile
cargo check                    # fast compile check
cargo test                     # run all tests (some #[ignore]d — see below)
cargo test -- --include-ignored # run ignored tests too (requires local PDF files)
cargo clippy                   # lint (no custom clippy.toml)
cargo doc --open               # local API docs
```

No CI pipeline, no pre-commit hooks, no rustfmt.toml — use `cargo fmt` with defaults.

## Running the Binary

Requires `.env` with `API_KEY` and `mcp.json` (copy from `mcp.json.example`).
Defaults: `EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qa`, `CHAT_MODEL=google/gemma-4-e4b`.
Connects to OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1` by default.
`SANDBOX_ROOTS` — comma-separated list of allowed filesystem paths (first is primary, default for writes). Example: `SANDBOX_ROOTS="./,/tmp/shared,/home/user/docs"`

## Architecture

```
src/
├── main.rs              # binary entrypoint — CLI chatbot wiring
├── lib.rs               # re-exports: agent, config, domain, mcp
├── config.rs            # McpConfig loader + validation
├── security/
│   └── sandbox.rs       # SandboxConfig, validate_sandboxed_path, find_containing_root, relative_display_path
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
Internal tools (filesystem, RAG, compact) are added to the same tool vec in `main.rs`.

## Testing

Tests in `tests/` (6 files): `embeddings.rs`, `rag.rs`, `tool_tests.rs`, `mcp_client.rs`, `mcp_registry.rs`.
- `test_read_pdf` in `tool_tests.rs` is `#[ignore]` — needs a local PDF file, will fail in CI.
- Tool tests use `tempfile` for sandbox isolation.
- MCP tests need live MCP servers or will fail — not safe to run blindly.

## Conventions

- All fallible operations use `anyhow::Result`. Domain errors use `thiserror`.
- Tools enforce sandbox via `security::sandbox::validate_sandboxed_path`: path traversal (`../`) returns `DocumentError::SandboxEscape`.
- MCP tool name deduplication: duplicate names across servers cause a hard error at connect time.
- `react.rs` is a stub — do not import from it. The `// pub mod react;` line in `agent/mod.rs` confirms it's excluded.
- `RagStoreBuilder` in `rag.rs` is deprecated (v0.2.0) — use `PdfLoader` + `WordSplitter` + `RagPipeline` instead.
- Vector store is in-memory only (`InMemoryVectorStore`/`InMemoryVectorIndex`). No persistence.

## MCP Config

`mcp.json` structure: `{ "mcpServers": { "<name>": { "type": "stdio"|"http", ... } } }`.
- Stdio: `command`, `args`, `env`, `cwd`
- HTTP: `url`, `headers`
- Transport type aliases accepted: `"http"`, `"jsonrpc"`, `"json-rpc"`, `"streamable_http"`
- File is gitignored. Example at `mcp.json.example`.

## Docs

- `docs/api_docs.md` — full API reference (379 lines)
- `docs/migration-0.2.0.md` — migration guide
- `docs/diagrams/` — architecture diagrams (C4, class, sequence, state, module dependency)
- `ROADMAP.md` — project roadmap and known gaps
