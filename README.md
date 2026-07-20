# AgentRS

**A high-performance Rust-based AI agent framework extending [Rig](https://github.com/0xPlaygrounds/rig) with native RAG support, MCP integration, and advanced memory management.**

> [!WARNING]
> **Active Development:** This project is under active development. Breaking changes are to be expected as we stabilize the APIs toward v1.0.

---

## 📖 About

**AgentRS** is an extension library built on top of the **[Rig](https://github.com/0xPlaygrounds/rig)** agent framework (`rig-core`). It adds enterprise-grade capabilities for context control, execution security, and vector store operations in Rust.

It integrates local semantic search via **[fastembed-rs](https://github.com/Anush008/fastembed-rs)** and persistent on-disk vector databases using **[turbovec](https://github.com/RyanCodrai/turbovec)**, enabling highly efficient Retrieval-Augmented Generation (RAG) workflows out-of-the-box.

---

## ✨ Key Features

*   **Rig Extension:** Extends standard `rig-core` agents with customizable ReAct loops, structured callbacks, and agent dispatching.
*   **Dynamic RAG Store:** Seamlessly chunk and index PDFs, Markdown, and text files using `fastembed` and `turbovec`.
*   **Model Context Protocol (MCP):** Connect stdio or HTTP MCP servers dynamically to load tools with deduplication.
*   **Secure Execution Sandbox:** Enforce path-traversal constraints and run commands/filesystem tools in restricted roots.
*   **Permission Gateways:** Fine-grained policies (`AllowAll`, `DenyAll`, `CliPrompt`, or `Custom`) to intercept tool calls before execution.
*   **Auto-Compacting Memory:** Smart history compression/summarization that automatically reduces token usage without losing conversation context.
*   **OpenTelemetry & LangSmith:** Native OTel span exporters to trace agent execution loops directly inside LangSmith.

---

## 🚀 Installation

Add the following to your `Cargo.toml` to use `agent_rs` in your project:

```toml
[dependencies]
agent_rs = { git = "https://github.com/skaarfundgandr/agent_rs.git" }
```

---

## 💻 Code Examples

### 1. Build a ReAct Loop Agent

Build an agent that reasons and calls tools iteratively:

```rust
use agent_rs::agent::ReActExt;
use rig_core::providers::openai;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::CompletionsClient::from_env();
    
    // Build agent with registered tools
    let agent = client
        .agent("gpt-4")
        .preamble("You are a helpful calculation assistant.")
        .tool(CalculatorTool)
        .build();

    // Wrap agent in the ReAct loop executor
    let react = agent
        .react()
        .max_cycles(5)
        .build();

    // Prompt the agent (returns a trace containing all reasoning steps and final answer)
    let trace = react.prompt("Calculate 15 + 27 * 3").await?;
    if let Some(final_answer) = trace.final_answer {
        println!("Answer: {}", final_answer.text);
    }
    Ok(())
}
```

### 2. Managed Agent with Auto-Compaction

Automatically compact history with summaries when conversation length exceeds a token threshold:

```rust
use agent_rs::agent::ManagedExt;
use rig_core::message::Message;
use rig_core::providers::openai;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::CompletionsClient::from_env();
    let agent = client.agent("gpt-4").build();

    // Construct a managed agent with automatic token-based memory compaction
    let managed = agent
        .managed()
        .with_compaction()
        .threshold(8000) // triggers summary compaction at 8k tokens
        .build();

    let mut history: Vec<Message> = vec![];
    let response = managed.chat_compact("Let's plan a coding project...", &mut history).await?;
    println!("Response: {}", response);
    Ok(())
}
```

### 3. Dynamic RAG Pipeline (turbovec + fastembed)

Set up a vector database index that updates dynamically on-the-fly:

```rust
use std::path::Path;
use agent_rs::agent::embeddings::EmbeddingService;
use agent_rs::rag::RagPipeline;
use rig_core::providers::openai;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::CompletionsClient::from_env();

    // Load local fastembed embedding model
    let embedder = EmbeddingService::from_fastembed("BGESmallENV15".parse()?)?;

    // Build a persistent RAG pipeline with the builder API
    let rag = RagPipeline::builder()
        .embedder(embedder)
        .store_at("./rag_data/")
        .extensions(["txt", "md", "pdf"])
        .build()
        .await?;

    // Ingest a document dynamically during agent execution
    rag.indexer.add(Path::new("./notes.pdf")).await?;

    // Expose RAG pipeline as a Rig index for dynamic context loading
    let agent = client
        .agent("gpt-4")
        .dynamic_context(3, rag.vector_index) // retrieves top 3 chunks automatically
        .build();

    Ok(())
}

### 4. GPU Acceleration (opt-in)

By default the `rag` feature builds with CPU-only ONNX Runtime. GPU acceleration requires one of these feature flags:

| Feature | Hardware |
|---|---|
| `rag-cuda` | NVIDIA GPUs (CUDA) |
| `rag-directml` | Windows GPUs (DirectML) |
| `rag-rocm` | AMD GPUs (ROCm) |
| `rag-load-dynamic` | System-provided ORT dylib (set `ORT_DYLIB_PATH`) |

```rust,no_run
use agent_rs::agent::embeddings::ort::execution_providers::{
    CUDAExecutionProvider,
    CPUExecutionProvider,
};
use agent_rs::agent::embeddings::EmbeddingService;

let embedder = EmbeddingService::from_fastembed_with_providers(
    "BGESmallENV15".parse()?,
    vec![
        CUDAExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(), // fallback
    ],
)?;
```

The default `rag` build is CPU-only and behavior-identical to v0.9.x.

---

## 📚 Documentation Reference

Detailed API documentation is located in the [docs/api/](docs/api/README.md) directory:

*   **Core Agent Loops**: [ReAct Loop](docs/api/react_loop.md) | [Memory & Context](docs/api/memory_and_agent_context.md)
*   **Integrations**: [MCP Registry](docs/api/config_and_mcp_modules.md) | [RAG Pipeline](docs/api/rag_pipeline.md)
*   **Security & Gates**: [Security Sandbox](docs/api/security_sandbox.md) | [Permission System](docs/api/permission_system.md)
*   **Diagnostics**: [Observability & OpenTelemetry](docs/api/observability.md) | [Domain Errors](docs/api/domain_errors.md)

### Architectural Diagrams

See visual layouts and sequence flows in the [docs/diagrams/](docs/diagrams/) directory:
*   [C4 System Architecture](docs/diagrams/c4-architecture.md)
*   [Class Struct Dependency](docs/diagrams/class-diagram.md)
*   [Execution Sequence Flowchart](docs/diagrams/flowchart.md)

---

## 🛠️ Examples

To see the framework in action, refer to the [examples/README.md](examples/README.md) file.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Format with `cargo fmt` and run checks/tests:
   ```bash
   cargo clippy --all-targets --all-features
   cargo test --all-features
   ```
4. Push to branch and open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

**Maintainer:** skaarfundgandr
**Project Link:** [https://github.com/skaarfundgandr/agent_rs](https://github.com/skaarfundgandr/agent_rs)

---
*Built with ❤️ using [Rig](https://github.com/0xPlaygrounds/rig) and [Rust](https://www.rust-lang.org/).*
