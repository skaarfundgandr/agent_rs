# API Documentation - agent_rs_lib

`agent_rs_lib` is a modular Rust library designed for building agentic AI workflows. It provides robust integrations with **Rig**, support for **RAG (Retrieval-Augmented Generation)**, **context history compaction**, and dynamic **MCP (Model Context Protocol) client registries**.

---

## Table of Contents
1. [Config and MCP Modules](#1-config-and-mcp-modules)
2. [Embedding Service](#2-embedding-service)
3. [RAG Pipeline](#3-rag-pipeline)
4. [Memory and Agent Context](#4-memory-and-agent-context)
5. [Agent Tools](#5-agent-tools)
6. [Domain Errors](#6-domain-errors)

---

## 1. Config and MCP Modules

Provides configuration parser and client interfaces to load and connect to stdio or HTTP-based MCP servers.

### `McpConfig`
Stores parsed configuration definitions for one or more MCP servers. Compatible with standard MCP `mcp.json` layouts.

#### Methods
* **`from_path(path: impl AsRef<Path>) -> Result<Self>`**
  Loads and parses an MCP configuration from a JSON file.
* **`validate(&self) -> Result<()>`**
  Validates that all configured servers have valid transport settings (e.g. valid URLs, stdio arguments, and no mixing of Stdio/HTTP configuration).
* **`resolved_server(&self, name: &str) -> Result<ResolvedMcpServer>`**
  Resolves transport specifications for a single named server.
* **`resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>>`**
  Resolves transport specifications for all configured servers.

---

### `McpClient`
Manages connections and tool listing for the configured MCP servers.

#### Methods
* **`from_config_path(path: &str) -> Result<Self>`**
  Initializes the client from a path to an `mcp.json` file.
* **`new(config: McpConfig) -> Self`**
  Constructs a new client using an existing configuration struct.
* **`connect(self) -> Result<McpRegistryRuntime>`**
  Establishes standard I/O processes or HTTP streams with all configured MCP servers.
* **`tools(self) -> Result<Vec<Box<dyn ToolDyn>>>`**
  Connects to all servers and returns all exposed endpoints as a list of dynamic Rig `ToolDyn` objects.

#### Example Usage: Loading MCP Tools
```rust
use agent_rs_lib::config::McpConfig;
use agent_rs_lib::mcp::client::McpClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Read and connect to MCP servers defined in mcp.json
    let client = McpClient::from_config_path("./mcp.json")?;
    let mcp_tools = client.tools().await?;
    
    println!("Loaded {} tools from MCP servers.", mcp_tools.len());
    Ok(())
}
```

---

## 2. Embedding Service

Wraps any Rig `EmbeddingModel` to provide structured document splitting, order-preserving batching, and error handling.

### `EmbeddingService<M>`
Generic over `M: EmbeddingModel`.

#### Methods
* **`new(model: M) -> Self`**
  Wraps a concrete Rig embedding model.
* **`ndims(&self) -> usize`**
  Returns the dimensions of the embedding vectors.
* **`max_documents(&self) -> usize`**
  Returns the maximum batch size accepted by the model provider in a single request.
* **`async embed_text(&self, text: impl AsRef<str>) -> Result<Embedding>`**
  Embeds a single string of text.
* **`async embed_texts<I, S>(&self, texts: I) -> Result<Vec<Embedding>>`**
  Embeds an iterator of text slices. Automatically batches the requests to respect `max_documents` while preserving original ordering.
* **`async embed_document<T: Embed>(&self, document: T) -> Result<(T, OneOrMany<Embedding>)>`**
  Extracts text fragments from a document implementing Rig's `Embed` trait and embeds them.
* **`async embed_documents<T: Embed, I>(&self, documents: I) -> Result<Vec<(T, OneOrMany<Embedding>)>>`**
  Batches and embeds multiple `Embed` documents, maintaining original ordering.

---

## 3. RAG Pipeline

A decoupled ingestion pipeline that transforms files into chunked, embedded vector indexes.

### Data Models (`src/domain/rag.rs`)
```rust
pub struct Document {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

pub struct Chunk {
    pub text: String,
    pub metadata: HashMap<String, String>,
}
```

### `DocumentLoader`
Trait for reading source files into generic `Document` structs.

```rust
pub trait DocumentLoader {
    fn load(&self, path: &Path) -> Result<Document>;
}
```
* **`PdfLoader`**: Extracts plain text from `.pdf` files.
* **`TextLoader`**: Reads plaintext content from `.txt` and `.md` files.

### `TextSplitter`
Trait for chunking documents into smaller searchable units.

```rust
pub trait TextSplitter {
    fn split(&self, document: &Document) -> Vec<Chunk>;
}
```
* **`WordSplitter`**: Splits document text by word boundaries using a sliding window.
  - Initialized via `WordSplitter::new(chunk_words: usize, chunk_overlap_words: usize)`.

---

### `RagPipeline`
Assembles document chunks, generates embeddings, and constructs Rig vector stores/indexes.

#### Methods
* **`new() -> Self`**
  Initializes an empty RAG pipeline.
* **`add_chunks(mut self, chunks: Vec<Chunk>) -> Self`**
  Directly appends an array of pre-built `Chunk`s.
* **`add_document<S: TextSplitter>(mut self, document: &Document, splitter: &S) -> Self`**
  Splits and appends a `Document` using the given splitter.
* **`add_documents<S: TextSplitter>(mut self, documents: &[Document], splitter: &S) -> Self`**
  Splits and appends multiple documents.
* **`async build_store<M: EmbeddingModel>(&self, embedding_service: &EmbeddingService<M>) -> Result<InMemoryVectorStore<String>>`**
  Generates embeddings and builds a Rig `InMemoryVectorStore` populated with formatted chunk strings.
* **`async build_index<M: EmbeddingModel + Clone>(&self, embedding_service: &EmbeddingService<M>) -> Result<InMemoryVectorIndex<M, String>>`**
  Builds and indexes the vector store for querying.

#### Example Usage: Building RAG Index
```rust
use std::path::Path;
use agent_rs_lib::agent::embeddings::EmbeddingService;
use agent_rs_lib::agent::rag::{DocumentLoader, PdfLoader, RagPipeline, WordSplitter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let openai = rig::providers::openai::Client::from_env()?;
    let embed_model = openai.embedding_model("text-embedding-3-small");
    let service = EmbeddingService::new(embed_model);

    // 1. Load document
    let doc = PdfLoader::new().load(Path::new("orientation.pdf"))?;

    // 2. Define chunking splitter (200 words per chunk, 40 words overlap)
    let splitter = WordSplitter::new(200, 40);

    // 3. Assemble and build Rig Vector Store Index
    let index = RagPipeline::new()
        .add_document(&doc, &splitter)
        .build_index(&service)
        .await?;

    Ok(())
}
```

---

## 4. Memory and Agent Context

Automates history size management to prevent context window overflows and excessive token costs.

### `ContextManagedAgent<M, C>`
Wraps an `Agent<M>` (where `M: CompletionModel`) and a compaction model `C: Prompt` to automatically summarize conversation history when it crosses a character-based token approximation threshold.

#### Methods
* **`async chat(&self, prompt: &str, history: &mut Vec<Message>) -> Result<String, PromptError>`**
  Executes an LLM chat turn. Summarizes conversation history in-place if threshold is crossed, then appends the current user prompt and assistant response.
* **`agent(&self) -> &Agent<M>`**
  Returns a reference to the inner wrapped `Agent`.

### `AgentContextExt`
Extension trait implemented for all standard Rig `Agent<M>` structs.

* **`with_compaction<C: Prompt>(self, threshold: usize, compaction_model: C) -> ContextManagedAgent<M, C>`**
  Wraps the receiver agent in a context managed wrapper.

#### Example Usage: Context Compaction
```rust
use agent_rs_lib::agent::memory::AgentContextExt;
use rig::message::Message;

let chat_agent = openai.agent("gpt-4o").build();
let compaction_agent = openai.agent("gpt-4o-mini").build();

// Wrap the chat agent to automatically compact context when it exceeds ~2000 tokens
let managed_agent = chat_agent.with_compaction(2000, compaction_agent);

let mut history = vec![];
let response = managed_agent.chat("What were my previous requests?", &mut history).await?;
```

---

## 5. Agent Tools

Standard Rig `Tool` implementations available to agents.

### `CompactTool`
Invokes a completion model to summarize conversation history.
- **Name**: `compact`
- **Arguments**: `CompactArgs { text: String }`

### `ReadDocumentTool`
Reads document contents from the filesystem. Supports `.txt`, `.md`, and `.pdf` files.
- **Name**: `read_document`
- **Arguments**: `ReadDocumentArgs { path: String }`

### `WriteDocumentTool`
Writes or appends content to a text file.
- **Name**: `write_document`
- **Arguments**: `WriteDocumentArgs { path: String, content: String, append: Option<bool> }`

---

## 6. Domain Errors

Robust, typed errors used across tools and modules.

### `DocumentError`
* `Io(std::io::Error)`: File read/write failures.
* `Pdf(String)`: PDF parsing and extraction failures.
* `UnsupportedExtension(String)`: Ingestion attempted on an unsupported file format.

### `CompactError`
* `Model(String)`: Errors returned by the compaction model.
