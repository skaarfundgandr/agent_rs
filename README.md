# AgentRS

**A high-performance Rust-based AI agent framework with native RAG support and MCP integration.**

---

## 📖 About

**AgentRS** is a lightweight yet powerful framework designed for developers building agentic workflows in Rust. It bridges the gap between large language models and real-world tools by providing a seamless integration layer for the **Model Context Protocol (MCP)** and **Retrieval-Augmented Generation (RAG)**.

Whether you're building a CLI-based research assistant or a complex automated workflow, AgentRS handles the heavy lifting of context management, document processing, and tool orchestration, allowing you to focus on the logic that matters.

### Why AgentRS?
- **Efficiency:** Built on Rust for maximum performance and safety.
- **Context Awareness:** Native support for PDF extraction and semantic search.
- **Memory Management:** Intelligent auto-compaction to keep your context window clean and cost-effective.
- **Extensibility:** First-class support for MCP tools and custom internal implementations.

---

## ✨ Key Features

*   **Model Context Protocol (MCP):** Easily connect to and consume tools from any MCP-compliant server.
*   **Dynamic RAG Store:** Built-in PDF parsing and chunking engine with vector search capabilities.
*   **Auto-Compacting Memory:** Automatically summarizes long conversation histories to manage token limits without losing critical context.
*   **Internal Toolset:** Ready-to-use tools for reading/writing local documents and managing session state.
*   **Ergonomic API:** A fluent builder pattern for configuring agents, embeddings, and chat interfaces.

---

## 🚀 Getting Started

### Prerequisites

*   **Rust (edition 2024):** Ensure you have the latest stable Rust toolchain installed.
*   **OpenAI-Compatible Provider:** A local or cloud-based model provider (e.g., LM Studio, OpenAI, or Google Gemma).
*   **Environment Variables(Optional):** An `.env` file with the following keys:
    ```env
    API_KEY=your_api_key_here
    EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qa
    CHAT_MODEL=google/gemma-4-e4b
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/skaarfundgandr/agent_rs.git
    cd agent_rs
    ```

2.  **Build the project:**
    ```bash
    cargo build --release
    ```

3. **Run tests to verify setup:**
    ```bash
    cargo test
    ```

### Adding as a Dependency

To use **AgentRS** as a library in your own project, add the following to your `Cargo.toml`:

```toml
[dependencies]
agent_rs = { git = "https://github.com/skaarfundgandr/agent_rs.git" }
```

> [!NOTE]
> The crate will be available in your code as `agent_rs_lib`.

---

## 🛠️ Usage

Running the built-in CLI chatbot is simple:

1. Ensure your `.env` is configured.
2. Set up your MCP servers. Copy `mcp.json.example` to `mcp.json` and customize it with your MCP server details:
   ```bash
   cp mcp.json.example mcp.json
   ```

### Running the Agent
```bash
cargo run
```

### Code Example: Building a Custom Agent
```rust
use agent_rs_lib::agent::rag::RagStoreBuilder;
use agent_rs_lib::agent::embeddings::EmbeddingService;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize Embeddings
    let embedding_service = EmbeddingService::new(client.embedding_model("my-model"));

    // 2. Build a RAG Store from a PDF
    let index = RagStoreBuilder::new(embedding_service)
        .add_pdf("./my_docs.pdf")?
        .build_index()
        .await?;

    // 3. Create the Agent
    let agent = client
        .agent("chat-model")
        .preamble("You are a helpful research assistant.")
        .dynamic_context(4, index)
        .build();

    Ok(())
}
```

## 🤝 Contributing

We welcome contributions! To get started:
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

Please ensure all tests pass and your code follows the established Rust idioms.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

**Maintainer:** skaarfundgandr
**Project Link:** [https://github.com/skaarfundgandr/agent_rs](https://github.com/skaarfundgandr/agent_rs)

---
*Built with ❤️ using [Rig](https://github.com/0xPlayground/rig) and [Rust](https://www.rust-lang.org/).*
