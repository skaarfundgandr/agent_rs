# Agent Tools

Standard Rig `Tool` implementations available to agents.

> **Architecture reference:** See the [C4 component diagram](../diagrams/c4-architecture.md) for how tools relate to the agent core, the [sandbox flowchart](../diagrams/flowchart.md) for path security enforcement, and the [class diagram](../diagrams/class-diagram.md) for tool type hierarchy.

All filesystem tools accept a `PermissionPolicy` and a `SandboxConfig` in their constructor. When the policy denies an operation, the tool returns `DocumentError::PermissionDenied` (see the [Domain Errors Reference](domain_errors.md)).

## `CompactTool`
Invokes a completion model to summarize conversation history.
- **Name**: `compact`
- **Arguments**: `CompactArgs { text: String }`

---

## `ReadDocumentTool`
Reads document contents from the filesystem. Access is restricted to a configurable sandbox root directory and an explicit set of allowed file extensions.
- **Name**: `read_document`
- **Constructor**: `ReadDocumentTool::new(sandbox: SandboxConfig, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `ReadDocumentArgs { path: String }` (resolved relative to the sandbox root)
- **Note**: When `"pdf"` is in the allowed set, PDF parsing is handled by `pdf-extract`; all other extensions are read as plain text.

---

## `WriteDocumentTool`
Writes or appends content to a text file. Access is restricted to a configurable sandbox root directory and an explicit set of allowed file extensions.
- **Name**: `write_document`
- **Constructor**: `WriteDocumentTool::new(sandbox: SandboxConfig, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `WriteDocumentArgs { path: String, content: String, append: Option<bool> }` (resolved relative to the sandbox root)

---

## `ListDirectoryTool`
Lists the contents of a directory within the sandbox root. Directories are prefixed with `[DIR]`, files with `[FILE]` (including byte size). Entries are sorted directories-first, then case-insensitively by name.
- **Name**: `list_directory`
- **Constructor**: `ListDirectoryTool::new(sandbox: SandboxConfig, policy: PermissionPolicy)`
- **Arguments**: `ListDirectoryArgs { path: Option<String> }` (defaults to sandbox root)

---

## `GlobSearchTool`
Finds files and directories matching a glob pattern within the sandbox root. Uses the [`glob`](https://crates.io/crates/glob) crate. Rejects absolute patterns and path traversals containing `..`. Returns up to 100 results.
- **Name**: `glob_search`
- **Constructor**: `GlobSearchTool::new(sandbox: SandboxConfig, policy: PermissionPolicy)`
- **Arguments**: `GlobSearchArgs { pattern: String, directory: Option<String> }` (pattern relative to sandbox root, e.g. `"src/**/*.rs"`; optional `directory` narrows the search to a specific subdirectory)

---

## `GrepSearchTool`
Searches for a substring pattern in workspace text files within the sandbox root. Only searches files whose extension is in the configured allowlist. Results are returned in `path:line: content` format, capped at 100 matches.
- **Name**: `grep_search`
- **Constructor**: `GrepSearchTool::new(sandbox: SandboxConfig, allowed_extensions: HashSet<String>, policy: PermissionPolicy)`
- **Arguments**: `GrepSearchArgs { query: String, path: Option<String>, case_sensitive: Option<bool> }`

---

## `ManageRagTool`
Unified tool for managing RAG sources. Supports three actions via a string enum: add a file or directory, remove a source, or list all indexed sources. After add/remove, the consumer should rebuild the RAG pipeline from the updated registry.
- **Name**: `manage_rag`
- **Constructor**: `ManageRagTool::new(registry: Arc<Mutex<RagSourceRegistry>>, sandbox: SandboxConfig, policy: PermissionPolicy)`
- **Arguments**: `ManageRagArgs { action: String, path: Option<String> }`
  - `action`: One of `"add"`, `"remove"`, or `"list"`.
  - `path`: Path to the file or directory (relative to sandbox root). Required for `"add"` and `"remove"`.

---

## `RagSourceRegistry`
Thread-safe registry that tracks which files and directories are indexed for RAG. Does not rebuild the vector index itself — consumers read [`sources()`](#methods) and rebuild the pipeline when needed. Intended to be wrapped in `Arc<Mutex<...>>` for shared ownership across tools.

### Methods
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
pub use rag::{ManageRagTool, RagSourceRegistry};
```

Crate-level re-exports (`src/agent/mod.rs`):

```rust
pub mod agents;
pub mod embeddings;
pub mod memory;
pub mod model;
pub mod permission;
// pub mod react;
pub mod tools;

pub use agents::{AgentContextExt, ContextManagedAgent, strip_reasoning_from_history};
pub use embeddings::EmbeddingService;
pub use permission::{PermissionGate, PermissionPolicy};
pub use tools::{
    CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ManageRagTool,
    RagSourceRegistry, ReadDocumentTool, WriteDocumentTool,
};
```

RAG module (`src/rag/mod.rs`) re-exports at crate root (`src/lib.rs`):

```rust
pub mod rag;
```

Module re-exports (`src/security/mod.rs`):

```rust
pub use sandbox::{SandboxConfig, find_containing_root, relative_display_path, validate_sandboxed_path};
```
