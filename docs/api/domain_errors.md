# Domain Errors

Robust, typed errors used across tools and modules.

> **Type reference:** See the [class diagram](../diagrams/class-diagram.md) for error enum variants and their usage across the system.

## `DocumentError`

* `Io(std::io::Error)`: File read/write failures.
* `Pdf(String)`: PDF parsing and extraction failures.
* `UnsupportedExtension(String)`: Ingestion or write attempted on an unsupported file format.
* `SandboxEscape(String)`: Unauthorized path traversal attempt outside the configured sandbox root folder.
* `PermissionDenied(String)`: Tool execution denied by the configured `PermissionPolicy`.
* `Rag(String)`: RAG registry errors — duplicate source, source not found, invalid action, or missing required arguments.
