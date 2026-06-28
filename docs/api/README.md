# API Reference

`agent_rs` is a modular Rust library designed for building agentic AI workflows. It provides robust integrations with **Rig**, support for **RAG (Retrieval-Augmented Generation)**, **context history compaction**, and dynamic **MCP (Model Context Protocol) client registries**.

> **Feature flags**
> This crate exposes optional `rag` and `opentelemetry` Cargo features.
> - **Default build** (`cargo build`): RAG and OTel code are fully compiled out, no extra deps pulled in. The `manage_rag` tool, RAG pipeline types, and observability module are unavailable; the rest of the library works as before.
> - **With RAG** (`cargo build --features rag`): adds `rig-fastembed`, `rig-sqlite`, `tokio-rusqlite`, `turbovec` as optional deps; enables `RagPipeline`, `DocumentStore`, `TurboIndex`, `TurboVectorIndex`, `EmbeddingService::from_fastembed()`, and `ManageRagTool`.
> - **With OpenTelemetry** (`cargo build --features opentelemetry`): adds `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`, `tracing-subscriber` as optional deps; enables `LangSmithConfig`, `TracerHandle`, `init_tracing()`, `shutdown_tracing()`, `LangSmithReActEmitter`, and `LangSmithAgentHook`.
> - **Both** (`cargo build --features "rag,opentelemetry"`): mutually orthogonal, both subsystems coexist.

## Modules & Reference Sections

1. [Config and MCP Modules](config_and_mcp_modules.md) — Parser and client interfaces for connecting to MCP servers.
2. [Security Sandbox](security_sandbox.md) — Multi-root filesystem sandbox path verification and canonicalization.
3. [Embedding Service](embedding_service.md) — Rig embedding wrappers, batching, and local fastembed.
4. [RAG Pipeline](rag_pipeline.md) — Persistent SQLite + turbovec document chunking and vector search pipeline.
5. [Memory and Agent Context](memory_and_agent_context.md) — Conversation history compaction and token-tracking agent wrapper.
6. [Permission System](permission_system.md) — Permission policy and custom execution gates for agent tools.
7. [Agent Tools](agent_tools.md) — Filesystem, search, compact, and RAG tools for agents.
8. [ReAct Loop](react_loop.md) — Per-cycle Reasoning + Acting agent loop with serializable trace.
9. [Observability](observability.md) — LangSmith OpenTelemetry tracing setup and rig span enrichment.
10. [Domain Errors](domain_errors.md) — Error enums and error handling types.
