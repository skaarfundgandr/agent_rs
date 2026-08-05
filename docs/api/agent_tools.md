# Agent Tools

Standard Rig `Tool` implementations available to agents.

> **Architecture reference:** See the [C4 component diagram](../diagrams/c4-architecture.md) for how tools relate to the agent core, the [sandbox flowchart](../diagrams/flowchart.md) for path security enforcement, and the [class diagram](../diagrams/class-diagram.md) for tool type hierarchy.

**Filesystem tools** (`ReadDocumentTool`, `WriteDocumentTool`, `ListDirectoryTool`,
`GlobSearchTool`, `GrepSearchTool`) accept a `PermissionPolicy` and an
`Arc<SharedSandbox>` in their constructor. When the policy denies an operation,
the tool returns `DocumentError::PermissionDenied` (see the
[Domain Errors Reference](domain_errors.md)).

**Reasoning tools** (`ThinkTool`) are pure and require no sandbox or permission
policy.

**Context / RAG tools** (`CompactTool`, `ManageRagTool`) are documented in their
sections below.

---

## `CompactTool`

Invokes a completion model to summarize conversation history.
- **Name**: `compact`
- **Arguments**: `CompactArgs { text: String }`

### Constructor
```rust
CompactTool::new(model: M) -> Self
```

---

## `ThinkTool`

No-op reasoning tool for ReAct agents. Echoes the thought into the tool-result
channel so it appears in working memory. Performs no I/O and cannot fail
(`Tool::Error = std::convert::Infallible`).

- **Name**: `think`
- **Arguments**: `ThinkArgs { thought: String }`
- **Output**: `ThinkOutput { thought: String, acknowledged: bool }`
- **Import**: `agent_rs::agent::ThinkTool` or `agent_rs::agent::tools::ThinkTool`

### Constructor

Unit struct — use `ThinkTool` directly (no `::new`).

### Example

```rust
use agent_rs::agent::tools::{ThinkArgs, ThinkTool};
use rig_core::tool::Tool;

// async context
let out = ThinkTool
    .call(ThinkArgs {
        thought: "list open questions before calling web_search".into(),
    })
    .await
    .expect("Infallible");
assert!(out.acknowledged);
```

### Notes

- Prefer this over `rig_core::tool::builtin::ThinkTool`, which uses a phantom
  `ThinkError` and returns a bare `String`.
- Inspired by Anthropic's think-tool pattern (engineering blog; citation only).
- Not feature-gated; always available in the default build.

---

## `ReadDocumentTool`

Reads document contents from the filesystem. Access is restricted to a configurable sandbox and an explicit set of allowed file extensions.
- **Name**: `read_document`
- **Arguments**: `ReadDocumentArgs { path: String }` (resolved relative to the sandbox root)
- **Note**: When `"pdf"` is in the allowed set, PDF parsing is handled by `pdf-extract`; all other extensions are read as plain text.

### Constructor
```rust
ReadDocumentTool::new(
    sandbox: Arc<SharedSandbox>,
    allowed_extensions: HashSet<String>,
    policy: PermissionPolicy,
) -> Self
```

---

## `WriteDocumentTool`

Writes or appends content to a text file. Access is restricted to a configurable sandbox and an explicit set of allowed file extensions.
- **Name**: `write_document`
- **Arguments**: `WriteDocumentArgs { path: String, content: String, append: Option<bool> }` (resolved relative to the sandbox root)

### Constructor
```rust
WriteDocumentTool::new(
    sandbox: Arc<SharedSandbox>,
    allowed_extensions: HashSet<String>,
    policy: PermissionPolicy,
) -> Self
```

---

## `ListDirectoryTool`

Lists the contents of a directory within the sandbox. Directories are prefixed with `[DIR]`, files with `[FILE]` (including byte size). Entries are sorted directories-first, then case-insensitively by name.
- **Name**: `list_directory`
- **Arguments**: `ListDirectoryArgs { path: Option<String> }` (defaults to sandbox root)

### Constructor
```rust
ListDirectoryTool::new(sandbox: Arc<SharedSandbox>, policy: PermissionPolicy) -> Self
```

---

## `GlobSearchTool`

Finds files and directories matching a glob pattern within the sandbox root(s). Uses the [`glob`](https://crates.io/crates/glob) crate. Rejects absolute patterns and path traversals containing parent directory components (`..`). When multiple roots are configured, searches all roots and deduplicates. Returns up to 100 results.
- **Name**: `glob_search`
- **Arguments**: `GlobSearchArgs { pattern: String, directory: Option<String> }` (pattern relative to sandbox root, e.g. `"src/**/*.rs"`; optional `directory` narrows the search to a specific subdirectory)

### Constructor
```rust
GlobSearchTool::new(sandbox: Arc<SharedSandbox>, policy: PermissionPolicy) -> Self
```

---

## `GrepSearchTool`

Searches for a substring pattern in workspace text files within the sandbox root. Only searches files whose extension is in the configured allowlist. Recursion depth is capped at a maximum of 10 directories. Results are returned in `path:line: content` format, capped at 100 matches. Supports case-insensitive (default) and case-sensitive modes.
- **Name**: `grep_search`
- **Arguments**: `GrepSearchArgs { query: String, path: Option<String>, case_sensitive: Option<bool> }`

### Constructor
```rust
GrepSearchTool::new(
    sandbox: Arc<SharedSandbox>,
    allowed_extensions: HashSet<String>,
    policy: PermissionPolicy,
) -> Self
```

---

## `ManageRagTool` (requires `rag` feature)

Unified tool for managing RAG sources. Supports four actions via a string enum: add a file or directory (optionally force-reindex), remove a source, list all indexed sources, or display status with per-source details and chunk counts.
- **Name**: `manage_rag`
- **Arguments**: `ManageRagArgs { action: String, path: Option<String>, force: Option<bool> }`
  - `action`: One of `"add"`, `"remove"`, `"list"`, or `"status"`.
  - `path`: Path to the file or directory (relative to sandbox root). Required for `"add"` and `"remove"`.
  - `force`: When `true` on `"add"`, re-indexes an already-registered source (refreshes modified files). Plain re-add stays a no-op returning `"indexed 0 chunks"`.
- **Output** (`status` action): Reports registered source count, per-source list (`[FILE]`/`[DIR]` + path), persisted chunk count, embedding dimensionality, and a hint when zero chunks are indexed.

### Construction

`ManageRagTool` has **no public constructor** — its `new` is `pub(crate)` and takes a `RagIndexer` plus a `PermissionPolicy`. The only public construction path is through the indexer:

```rust
let manage_tool = rag.indexer.tool(PermissionPolicy::AllowAll);
```

See [`RagIndexer::tool(policy)`](rag_pipeline.md) in the RAG pipeline docs.

---

## `SearchRagTool` (requires `rag` feature)

Read-only semantic search over the indexed RAG corpus. Performs a vector similarity search against embedding indexes. Ungated (no permission check required).
- **Name**: `rag_search`
- **Arguments**: `SearchRagArgs { query: String, samples: Option<u64>, threshold: Option<f64> }`
  - `query`: The search text (required).
  - `samples`: Maximum number of results (optional, default `4`).
  - `threshold`: Minimum similarity score — results below this are dropped (optional; scores are quantized inner-product estimates, approximately cosine similarity for normalized embeddings).
- **Output**: A formatted string of matching text excerpts with scores.
- **Created via**: `RagIndexer::search_tool()` which shares the pipeline's live store and index.

### Constructor
Not constructed manually — obtained from `RagIndexer::search_tool()`.

### Example
```rust
use agent_rs::agent::tools::SearchRagTool;
// Obtained via RagIndexer::search_tool() — see RAG Pipeline docs.
```

---

## `RagSourceRegistry` (requires `rag` feature)

Thread-safe registry that tracks which files and directories are indexed for RAG. Does not rebuild the vector index itself — consumers read [`sources()`](#methods) and rebuild the pipeline when needed. Intended to be wrapped in `Arc<Mutex<...>>` for shared ownership across tools.

### Methods
- **`new(supported_extensions: HashSet<String>) -> Self`**
  Creates an empty registry. `supported_extensions` is the set of file extensions (without the dot) the consumer can load.
- **`hydrate_from_store(pipeline: &RagPipeline, supported_extensions: HashSet<String>) -> Result<Self>`**
  Rebuilds the registry from sources persisted in the SQLite `rag_sources` table (used to restore the registry after a restart).
- **`add_source(&mut self, path: &Path, sandbox: &SharedSandbox) -> Result<String, DocumentError>`**
  Validates the path against the sandbox, checks the file extension, rejects duplicates, and registers the source.
- **`remove_source(&mut self, canonical_path: &Path) -> Result<String, DocumentError>`**
  Removes a source by its canonical path. Returns an error if no source matches.
- **`sources(&self) -> &[RagSource]`**
  Returns a read-only slice of registered sources for consumers to iterate when rebuilding the pipeline.
- **`list_sources(&self) -> String`**
  Returns a formatted string listing all registered sources with their type.
- **`is_empty(&self) -> bool`**
  Returns `true` if no sources are registered.

---

## `extract_pdf_text()`

A free function to extract text from PDF files (used internally by `ReadDocumentTool`).

```rust
pub fn extract_pdf_text<P: AsRef<Path>>(path: P) -> Result<String>
```

---

> **Migration from v0.1.0**: See the [Migration Guide](../migration-0.2.0.md) for breaking changes to tool constructors.

---

## Module Re-exports

Located in `src/agent/tools/mod.rs`:

```rust
pub use context::CompactTool;
pub use directory::ListDirectoryTool;
pub use document::{ReadDocumentTool, WriteDocumentTool};
pub use glob::GlobSearchTool;
pub use search::GrepSearchTool;
pub use registry::{RegisteredTool, ToolFactory, ToolRegistry, ToolRegistryBuilder};
pub use think::{ThinkArgs, ThinkOutput, ThinkTool};
#[cfg(feature = "rag")]
pub use rag::{ManageRagTool, RagSourceRegistry, SearchRagTool};
```

**Tool composition:** `ToolRegistry`, `ToolRegistryBuilder`, `RegisteredTool`, and `ToolFactory` are re-exported from `agent_rs::agent::tools` for composing tool sets into a shared registry — see `src/agent/tools/registry.rs`.

Crate-level re-exports (`src/agent/mod.rs`):

```rust
pub use tools::{
    CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ReadDocumentTool,
    ThinkTool, WriteDocumentTool,
};
#[cfg(feature = "rag")]
pub use tools::{ManageRagTool, RagSourceRegistry};
```

Note: `SearchRagTool` is **not** re-exported at the crate level — import it via `agent_rs::agent::tools::SearchRagTool`.
