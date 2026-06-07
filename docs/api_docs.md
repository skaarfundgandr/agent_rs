# API Documentation - agent_rs_lib

`agent_rs_lib` is a modular Rust library designed for building agentic AI workflows. It provides robust integrations with **Rig**, support for **RAG (Retrieval-Augmented Generation)**, **context history compaction**, and dynamic **MCP (Model Context Protocol) client registries**.

---

## Table of Contents
1. [Config and MCP Modules](#1-config-and-mcp-modules)
2. [Security Sandbox](#2-security-sandbox)
3. [Embedding Service](#3-embedding-service)
4. [RAG Pipeline](#4-rag-pipeline)
5. [Memory and Agent Context](#5-memory-and-agent-context)
6. [Permission System](#6-permission-system)
7. [Agent Tools](#7-agent-tools)
8. [Domain Errors](#8-domain-errors)

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
* **`async connect(self) -> Result<McpRegistryRuntime>`**
  Establishes standard I/O processes or HTTP streams with all configured MCP servers.
* **`async tools(self) -> Result<Vec<Box<dyn ToolDyn>>>`**
  Connects to all servers and returns all exposed endpoints as a list of dynamic Rig `ToolDyn` objects.

---

### `McpRegistry`
Registry that resolves MCP server definitions from `mcp.json` into Rig tools and performs name deduplication.

#### Methods
* **`new(config: McpConfig) -> Self`**
  Creates a registry from a validated configuration.
* **`from_path(path: impl AsRef<Path>) -> Result<Self>`**
  Creates a registry from a configuration file path.
* **`from_client(client: McpClient) -> Self`**
  Creates a registry from an existing client manager.
* **`async connect(&self) -> Result<McpRegistryRuntime>`**
  Connects to all configured MCP servers and collects their tools.
* **`async tools(&self) -> Result<Vec<Box<dyn ToolDyn>>>`**
  Connects to all configured MCP servers and returns Rig-compatible boxed tools.

---

### `McpRegistryRuntime`
Runtime registry returned after connecting to the MCP servers, holding the active connections and resolved tools.

#### Methods
* **`servers(&self) -> &[RegisteredMcpServer]`**
  Returns registered servers in connection order.
* **`server(&self, name: &str) -> Option<&RegisteredMcpServer>`**
  Looks up a server by name.
* **`tools(&self) -> &[RegisteredMcpTool]`**
  Returns registered tools in connection order.
* **`tool_names(&self) -> impl Iterator<Item = &str>`**
  Convenience iterator over tool names.
* **`into_tools(self) -> Vec<Box<dyn ToolDyn>>`**
  Converts the runtime registry into boxed Rig tools.

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

## 2. Security Sandbox

Multi-root filesystem sandboxing. Controls which paths tools can access and where new files are written.

> **Architecture reference:** See the [sandbox validation flowchart](diagrams/flowchart.md) for the path resolution algorithm, and the [module dependency graph](diagrams/module-dependency.md) for how `security::sandbox` fits into the crate.

### `SandboxConfig`
Stores an ordered list of allowed filesystem roots. The first root is the **primary** — used as the default target when creating new files. All roots participate equally in read/search/glob operations.

```rust
pub struct SandboxConfig {
    roots: Vec<PathBuf>,
    canonical_roots: Vec<PathBuf>,
}
```

#### Methods
* **`single(root: impl Into<PathBuf>) -> Result<Self, DocumentError>`**
  Creates a sandbox with a single root. Returns an IO error if the root cannot be canonicalized.
* **`new(roots: Vec<PathBuf>) -> Result<Self, DocumentError>`**
  Creates a sandbox with multiple roots. The first root is primary. Returns an error if `roots` is empty or if any root cannot be canonicalized. All roots are canonicalized at construction time.
* **`primary(&self) -> &Path`**
  Returns the primary (first) root.
* **`roots(&self) -> &[PathBuf]`**
  Returns the original (non-canonicalized) roots. Use for display paths and user-facing operations.
* **`canonical_roots(&self) -> &[PathBuf]`**
  Returns canonicalized roots. Use for security validation.
* **`len(&self) -> usize`**
  Returns the number of configured roots.
* **`is_empty(&self) -> bool`**
  Returns `true` if no roots are configured (should never happen).

#### Trait Implementations
* `Default` — uses `"."` as a single root.
* `TryFrom<PathBuf>`, `TryFrom<&Path>`, `TryFrom<&str>` — each creates a single-root sandbox. Returns an error if the root cannot be canonicalized.

### `validate_sandboxed_path()`
```rust
pub fn validate_sandboxed_path(
    sandbox: &SandboxConfig,
    user_path: &Path,
) -> Result<PathBuf, DocumentError>
```
Validates that `user_path` resolves to a path within one of the sandbox roots. Uses a two-phase algorithm:
1. **Phase 1 (reads):** Try each canonical root in order. If the joined+canonicalized path falls within a root **and** the file exists, return it.
2. **Phase 2 (writes):** If no root has the file, use the primary root to allow creating new files.
Returns `DocumentError::SandboxEscape` if the resolved path falls outside all roots.

### `find_containing_root()`
```rust
pub fn find_containing_root<'a>(
    sandbox: &'a SandboxConfig,
    path: &Path,
) -> Option<&'a PathBuf>
```
Returns the original (non-canonicalized) root that contains `path`, or `None` if the path does not fall under any root. Uses original roots for comparison to avoid Windows `\\?\` prefix issues.

### `relative_display_path()`
```rust
pub fn relative_display_path(sandbox: &SandboxConfig, path: &Path) -> String
```
Computes a relative display path for a file, preferring the shortest prefix among the sandbox roots for readability. Falls back to the full path if no root matches.

### Environment Variables
* **`SANDBOX_ROOTS`** — comma-separated list of allowed filesystem paths. The first path is the primary root (default for writes). Example: `SANDBOX_ROOTS="./,/tmp/shared,/home/user/docs"`. Defaults to `"./"` if unset.

---

## 3. Embedding Service

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

## 4. RAG Pipeline

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

## 5. Memory and Agent Context

Automates history size management to prevent context window overflows and excessive token costs.

> **Runtime reference:** See the [history compaction flowchart](diagrams/flowchart.md) for the compaction algorithm, the [runtime sequence diagram](diagrams/sequence-diagram.md) for how context management interacts with the chat loop, and the [class diagram](diagrams/class-diagram.md) for `ContextManagedAgent` and `AgentContextExt`.

### `ContextManagedAgent<M, C, P>`
Wraps an `Agent<M, P>` (where `M: CompletionModel` and `P: PromptHook<M>`) and a compaction model `C: Prompt` to automatically summarize conversation history when it crosses a token threshold calculated accurately via the `cl100k_base` BPE tokenizer.

#### Methods
* **`async chat(&self, prompt: &str, history: &mut Vec<Message>) -> Result<String, PromptError>`**
  Executes an LLM chat turn. Summarizes conversation history in-place if threshold is crossed, then appends the current user prompt and assistant response.
* **`async chat_with_owned_history(&self, prompt: &str, history: Vec<Message>) -> Result<(String, Vec<Message>), PromptError>`**
  Executes an LLM chat turn using owned history, returning the updated history rather than mutating it in-place.
* **`async stream_chat(&self, prompt: &str, history: &[Message]) -> Result<(ContextManagedChatStream<impl Stream, M::StreamingResponse>, oneshot::Receiver<Vec<Message>>), PromptError>`**
  Executes a streaming LLM chat turn. Compacts history if needed, returns a stream wrapper yielding elements, and a Future (oneshot Receiver) that resolves to the updated history once the stream is fully consumed.
* **`async stream_chat_with_owned_history(&self, prompt: &str, history: Vec<Message>) -> Result<(ContextManagedChatStream<impl Stream, M::StreamingResponse>, oneshot::Receiver<Vec<Message>>), PromptError>`**
  Executes a streaming LLM chat turn using owned history.
* **`with_token_estimator(mut self, estimator: fn(&[Message]) -> usize) -> Self`**
  Registers a custom token estimator callback to replace the default `cl100k_base` token counting.
* **`with_compaction_prompt_formatter(mut self, formatter: fn(&str) -> String) -> Self`**
  Registers a custom prompt formatter to format the compaction request sent to the compaction model.
* **`agent(&self) -> &Agent<M, P>`**
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

## 6. Permission System

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

## 7. Agent Tools

Standard Rig `Tool` implementations available to agents.

> **Architecture reference:** See the [C4 component diagram](diagrams/c4-architecture.md) for how tools relate to the agent core, the [sandbox validation flowchart](diagrams/flowchart.md) for path security enforcement, and the [class diagram](diagrams/class-diagram.md) for tool type hierarchy.

All filesystem tools accept a `PermissionPolicy` and a `SandboxConfig` in their constructor. When the policy denies an operation, the tool returns `DocumentError::PermissionDenied` (see [§8 Domain Errors](#8-domain-errors)).

### `CompactTool`
Invokes a completion model to summarize conversation history.
- **Name**: `compact`
- **Arguments**: `CompactArgs { text: String }`

### `ReadDocumentTool`
Reads document contents from the filesystem. Access is restricted to a configurable sandbox root directory and an explicit set of allowed file extensions.
- **Name**: `read_document`
- **Constructor**: `ReadDocumentTool::new(sandbox: SandboxConfig, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `ReadDocumentArgs { path: String }` (resolved relative to the sandbox root)
- **Note**: When `"pdf"` is in the allowed set, PDF parsing is handled by `pdf-extract`; all other extensions are read as plain text.

### `WriteDocumentTool`
Writes or appends content to a text file. Access is restricted to a configurable sandbox root directory and an explicit set of allowed file extensions.
- **Name**: `write_document`
- **Constructor**: `WriteDocumentTool::new(sandbox: SandboxConfig, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `WriteDocumentArgs { path: String, content: String, append: Option<bool> }` (resolved relative to the sandbox root)

### `ListDirectoryTool`
Lists the contents of a directory within the sandbox root. Directories are prefixed with `[DIR]`, files with `[FILE]` (including byte size). Entries are sorted directories-first, then case-insensitively by name.
- **Name**: `list_directory`
- **Constructor**: `ListDirectoryTool::new(sandbox: SandboxConfig, policy: PermissionPolicy)`
- **Arguments**: `ListDirectoryArgs { path: Option<String> }` (defaults to sandbox root)

### `GlobSearchTool`
Finds files and directories matching a glob pattern within the sandbox root. Uses the [`glob`](https://crates.io/crates/glob) crate. Rejects absolute patterns and path traversals containing `..`. Returns up to 100 results.
- **Name**: `glob_search`
- **Constructor**: `GlobSearchTool::new(sandbox: SandboxConfig, policy: PermissionPolicy)`
- **Arguments**: `GlobSearchArgs { pattern: String, directory: Option<String> }` (pattern relative to sandbox root, e.g. `"src/**/*.rs"`; optional `directory` narrows the search to a specific subdirectory)

### `GrepSearchTool`
Searches for a substring pattern in workspace text files within the sandbox root. Only searches files whose extension is in the configured allowlist. Results are returned in `path:line: content` format, capped at 100 matches.
- **Name**: `grep_search`
- **Constructor**: `GrepSearchTool::new(sandbox: SandboxConfig, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `GrepSearchArgs { query: String, path: Option<String>, case_sensitive: Option<bool> }`

### `ManageRagTool`
Unified tool for managing RAG sources. Supports three actions via a string enum: add a file or directory, remove a source, or list all indexed sources. After add/remove, the consumer should rebuild the RAG pipeline from the updated registry.
- **Name**: `manage_rag`
- **Constructor**: `ManageRagTool::new(registry: Arc<Mutex<RagSourceRegistry>>, sandbox: SandboxConfig, policy: PermissionPolicy)`
- **Arguments**: `ManageRagArgs { action: String, path: Option<String> }`
  - `action`: One of `"add"`, `"remove"`, or `"list"`.
  - `path`: Path to the file or directory (relative to sandbox root). Required for `"add"` and `"remove"`.

### `RagSourceRegistry`
Thread-safe registry that tracks which files and directories are indexed for RAG. Does not rebuild the vector index itself — consumers read [`sources()`](RagSourceRegistry::sources) and rebuild the pipeline when needed. Intended to be wrapped in `Arc<Mutex<...>>` for shared ownership across tools.

#### Methods
* **`new(supported_extensions: HashSet<String>) -> Self`**
  Creates an empty registry. `supported_extensions` is the set of file extensions (without the dot) the consumer can load.
* **`add_source(&mut self, path: &Path, sandbox: &SandboxConfig) -> Result<String, DocumentError>`**
  Validates the path against the sandbox, checks the file extension, rejects duplicates, and registers the source.
* **`remove_source(&mut self, canonical_path: &Path) -> Result<String, DocumentError>`**
  Removes a source by its canonical path. Returns an error if no source matches.
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
pub use embeddings::EmbeddingService;
pub use memory::{AgentContextExt, ContextManagedAgent};
pub use rag::{
    Chunk, Document, DocumentLoader, PdfLoader, RagPipeline, RagSource, RagSourceType,
    TextLoader, TextSplitter, WordSplitter,
};
pub use tools::{
    CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ManageRagTool,
    RagSourceRegistry, ReadDocumentTool, WriteDocumentTool,
};
```

Module re-exports (`src/security/mod.rs`):

```rust
pub use sandbox::{SandboxConfig, find_containing_root, relative_display_path, validate_sandboxed_path};
```

---

## 8. Domain Errors

Robust, typed errors used across tools and modules.

> **Type reference:** See the [class diagram](diagrams/class-diagram.md) for error enum variants and their usage across the system.

### `DocumentError`
* `Io(std::io::Error)`: File read/write failures.
* `Pdf(String)`: PDF parsing and extraction failures.
* `UnsupportedExtension(String)`: Ingestion or write attempted on an unsupported file format.
* `SandboxEscape(String)`: Unauthorized path traversal attempt outside the configured sandbox root folder.
* `PermissionDenied(String)`: Tool execution denied by the configured `PermissionPolicy`.
* `Rag(String)`: RAG registry errors — duplicate source, source not found, invalid action, or missing required arguments.
