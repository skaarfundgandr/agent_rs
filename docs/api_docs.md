# API Documentation - agent_rs_lib

`agent_rs_lib` is a modular Rust library designed for building agentic AI workflows. It provides robust integrations with **Rig**, support for **RAG (Retrieval-Augmented Generation)**, **context history compaction**, and dynamic **MCP (Model Context Protocol) client registries**.

---

## Table of Contents
1. [Config and MCP Modules](#1-config-and-mcp-modules)
2. [Embedding Service](#2-embedding-service)
3. [RAG Pipeline](#3-rag-pipeline)
4. [Memory and Agent Context](#4-memory-and-agent-context)
5. [Permission System](#5-permission-system)
6. [Agent Tools](#6-agent-tools)
7. [Domain Errors](#7-domain-errors)

---

## 1. Config and MCP Modules

Provides configuration parser and client interfaces to load and connect to stdio or HTTP-based MCP servers.

> **Architecture reference:** See the [C4 architecture diagram](diagrams/c4-architecture.md) for how these modules fit into the system, the [class diagram](diagrams/class-diagram.md) for type relationships (`McpConfig`, `McpServerDef`, `McpTransportSpec`), and the [module dependency graph](diagrams/module-dependency.md) for crate-level structure.

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

> **Lifecycle reference:** See the [MCP connection state diagram](diagrams/state-diagram.md) for the full connection lifecycle and the [startup sequence diagram](diagrams/sequence-diagram.md) for how `connect()` fits into application bootstrap.

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

> **Pipeline reference:** See the [RAG flowchart](diagrams/flowchart.md) for how embedding batching integrates with the ingestion pipeline, and the [class diagram](diagrams/class-diagram.md) for `EmbeddingService`'s generic type constraints and relationships.

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

> **Pipeline reference:** See the [RAG processing flowchart](diagrams/flowchart.md) for the end-to-end document ingestion flow, the [class diagram](diagrams/class-diagram.md) for trait relationships (`DocumentLoader`, `TextSplitter`, `RagPipeline`), and the [sequence diagram](diagrams/sequence-diagram.md) for how the pipeline is invoked at startup.

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

pub enum RagSourceType {
    File,
    Directory,
}

pub struct RagSource {
    pub path: PathBuf,
    pub source_type: RagSourceType,
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

> **Runtime reference:** See the [history compaction flowchart](diagrams/flowchart.md) for the compaction algorithm, the [runtime sequence diagram](diagrams/sequence-diagram.md) for how context management interacts with the chat loop, and the [class diagram](diagrams/class-diagram.md) for `ContextManagedAgent` and `AgentContextExt`.

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

## 5. Permission System

> **Type reference:** See the [class diagram](diagrams/class-diagram.md) for the `PermissionPolicy` type hierarchy.

### `PermissionPolicy`
Controls whether tool execution is allowed, denied, or requires confirmation.
```rust
pub enum PermissionPolicy {
    AllowAll,
    DenyAll,
    CliPrompt,
    Custom(Arc<dyn PermissionGate>),
}
```
- `AllowAll` — automatically permits every tool call.
- `DenyAll` — automatically denies every tool call.
- `CliPrompt` — prints a prompt to stderr and reads `y/N` from stdin.
- `Custom(gate)` — delegates to a user-defined `PermissionGate`.

### `PermissionGate` trait
```rust
#[async_trait::async_trait]
pub trait PermissionGate: Send + Sync {
    async fn check_permission(&self, tool_name: &str, description: &str) -> bool;
}
```

---

## 6. Agent Tools

Standard Rig `Tool` implementations available to agents.

> **Architecture reference:** See the [C4 component diagram](diagrams/c4-architecture.md) for how tools relate to the agent core, the [sandbox validation flowchart](diagrams/flowchart.md) for path security enforcement, and the [class diagram](diagrams/class-diagram.md) for tool type hierarchy.

All filesystem tools accept a `PermissionPolicy` in their constructor. When the policy denies an operation, the tool returns `DocumentError::PermissionDenied` (see [§7 Domain Errors](#7-domain-errors)).

### `CompactTool`
Invokes a completion model to summarize conversation history.
- **Name**: `compact`
- **Arguments**: `CompactArgs { text: String }`

### `ReadDocumentTool`
Reads document contents from the filesystem. Access is restricted to a configurable sandbox root directory and an explicit set of allowed file extensions.
- **Name**: `read_document`
- **Constructor**: `ReadDocumentTool::new(sandbox_root: impl Into<PathBuf>, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `ReadDocumentArgs { path: String }` (resolved relative to the sandbox root)
- **Note**: When `"pdf"` is in the allowed set, PDF parsing is handled by `pdf-extract`; all other extensions are read as plain text.

### `WriteDocumentTool`
Writes or appends content to a text file. Access is restricted to a configurable sandbox root directory and an explicit set of allowed file extensions.
- **Name**: `write_document`
- **Constructor**: `WriteDocumentTool::new(sandbox_root: impl Into<PathBuf>, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `WriteDocumentArgs { path: String, content: String, append: Option<bool> }` (resolved relative to the sandbox root)

### `ListDirectoryTool`
Lists the contents of a directory within the sandbox root. Directories are prefixed with `[DIR]`, files with `[FILE]` (including byte size). Entries are sorted directories-first, then case-insensitively by name.
- **Name**: `list_directory`
- **Constructor**: `ListDirectoryTool::new(sandbox_root: impl Into<PathBuf>, policy: PermissionPolicy)`
- **Arguments**: `ListDirectoryArgs { path: Option<String> }` (defaults to sandbox root)

### `GlobSearchTool`
Finds files and directories matching a glob pattern within the sandbox root. Uses the [`glob`](https://crates.io/crates/glob) crate. Rejects absolute patterns and path traversals containing `..`. Returns up to 100 results.
- **Name**: `glob_search`
- **Constructor**: `GlobSearchTool::new(sandbox_root: impl Into<PathBuf>, policy: PermissionPolicy)`
- **Arguments**: `GlobSearchArgs { pattern: String }` (relative to sandbox root, e.g. `"src/**/*.rs"`)

### `GrepSearchTool`
Searches for a substring pattern in workspace text files within the sandbox root. Only searches files whose extension is in the configured allowlist. Results are returned in `path:line: content` format, capped at 100 matches.
- **Name**: `grep_search`
- **Constructor**: `GrepSearchTool::new(sandbox_root: impl Into<PathBuf>, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `GrepSearchArgs { query: String, path: Option<String>, case_sensitive: Option<bool> }`

### `ManageRagTool`
Unified tool for managing RAG sources. Supports three actions via a string enum: add a file or directory, remove a source, or list all indexed sources. After add/remove, the consumer should rebuild the RAG pipeline from the updated registry.
- **Name**: `manage_rag`
- **Constructor**: `ManageRagTool::new(registry: Arc<Mutex<RagSourceRegistry>>, sandbox_root: impl Into<PathBuf>, policy: PermissionPolicy)`
- **Arguments**: `ManageRagArgs { action: String, path: Option<String> }`
  - `action`: One of `"add"`, `"remove"`, or `"list"`.
  - `path`: Path to the file or directory (relative to sandbox root). Required for `"add"` and `"remove"`.

### `RagSourceRegistry`
Thread-safe registry that tracks which files and directories are indexed for RAG. Does not rebuild the vector index itself — consumers read [`sources()`](RagSourceRegistry::sources) and rebuild the pipeline when needed. Intended to be wrapped in `Arc<Mutex<...>>` for shared ownership across tools.

#### Methods
* **`new(supported_extensions: HashSet<String>) -> Self`**
  Creates an empty registry. `supported_extensions` is the set of file extensions (without the dot) the consumer can load.
* **`add_source(&mut self, path: &Path, sandbox_root: &Path) -> Result<String, DocumentError>`**
  Validates the path against the sandbox root, checks the file extension, rejects duplicates, and registers the source.
* **`remove_source(&mut self, path: &str) -> Result<String, DocumentError>`**
  Removes a source by its path string. Returns an error if no source matches.
* **`list_sources(&self) -> String`**
  Returns a formatted string listing all registered sources with their type and index.
* **`sources(&self) -> &[RagSource]`**
  Returns a read-only slice of registered sources for consumers to iterate when rebuilding the pipeline.
* **`is_empty(&self) -> bool`**
  Returns `true` if no sources are registered.

---

> **Migration from v0.1.0**: See [`migration-0.2.0.md`](migration-0.2.0.md) for breaking changes to tool constructors.

### Module Re-exports (`src/agent/tools/mod.rs`)

```rust
pub use context::CompactTool;
pub use directory::ListDirectoryTool;
pub use document::{ReadDocumentTool, WriteDocumentTool};
pub use glob::GlobSearchTool;
pub use search::GrepSearchTool;
pub use rag::{ManageRagTool, RagSourceRegistry};
```

Crate-level re-exports (`src/agent/mod.rs`):

```rust
pub mod permission;
pub use permission::{PermissionGate, PermissionPolicy};
pub use rag::{
    Chunk, Document, DocumentLoader, PdfLoader, RagPipeline, RagSource, RagSourceType,
    TextLoader, TextSplitter, WordSplitter,
};
pub use tools::{
    CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ManageRagTool,
    ReadDocumentTool, RagSourceRegistry, WriteDocumentTool,
};
```

---

## 7. Domain Errors

Robust, typed errors used across tools and modules.

> **Type reference:** See the [class diagram](diagrams/class-diagram.md) for error enum variants and their usage across the system.

### `DocumentError`
* `Io(std::io::Error)`: File read/write failures.
* `Pdf(String)`: PDF parsing and extraction failures.
* `UnsupportedExtension(String)`: Ingestion or write attempted on an unsupported file format.
* `SandboxEscape(String)`: Unauthorized path traversal attempt outside the configured sandbox root folder.
* `PermissionDenied(String)`: Tool execution denied by the configured `PermissionPolicy`.
* `Rag(String)`: RAG registry errors — duplicate source, source not found, invalid action, or missing required arguments.
