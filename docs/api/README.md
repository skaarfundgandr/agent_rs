# API Reference

`agent_rs_lib` is a modular Rust library designed for building agentic AI workflows. It provides robust integrations with **Rig**, support for **RAG (Retrieval-Augmented Generation)**, **context history compaction**, and dynamic **MCP (Model Context Protocol) client registries**.

> **Feature flags**
> This crate exposes an optional `rag` Cargo feature that gates the RAG subsystem (turbovec ANN index, rig-fastembed local embeddings, SQLite chunk metadata).
> - **Default build** (`cargo build`): RAG code is fully compiled out, no extra deps pulled in. The `manage_rag` tool and RAG pipeline types are unavailable; the rest of the library works as before.
> - **With RAG** (`cargo build --features rag`): adds `rig-fastembed`, `rig-sqlite`, `tokio-rusqlite`, `turbovec` as optional deps; enables `RagPipeline`, `DocumentStore`, `TurboIndex`, `TurboVectorIndex`, `EmbeddingService::from_fastembed()`, and `ManageRagTool`.

## Modules & Reference Sections

1. [Config and MCP Modules](config_and_mcp_modules.md) — Parser and client interfaces for connecting to MCP servers.
2. [Security Sandbox](security_sandbox.md) — Multi-root filesystem sandbox path verification and canonicalization.
3. [Embedding Service](embedding_service.md) — Rig embedding wrappers, batching, and local fastembed.
4. [RAG Pipeline](rag_pipeline.md) — Persistent SQLite + turbovec document chunking and vector search pipeline.
5. [Memory and Agent Context](memory_and_agent_context.md) — Conversation history compaction and token-tracking agent wrapper.
6. [Permission System](permission_system.md) — Permission policy and custom execution gates for agent tools.
7. [Agent Tools](agent_tools.md) — Filesystem, search, compact, and RAG tools for agents.
8. [Domain Errors](domain_errors.md) — Error enums and error handling types.
