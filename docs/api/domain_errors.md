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
* `Sandbox(String)`: Sandbox invariant violation (e.g. attempting to remove the last remaining root from a `SandboxConfig`).

## `ReActError`

Returned from `BuiltReAct::prompt()` / `BuiltReAct::chat()`. Lives in `src/domain/errors.rs`.

* `MaxCyclesExceeded { cycles: usize }` — the loop reached `max_cycles` without a final answer.
* `ToolExecution { tool: String, source: Box<dyn Error + Send + Sync> }` — a tool call returned an error; the `result` field of the corresponding `Observation` will carry the error text (`is_error: true`).
* `Model(String)` — completion error (rig's `PromptError` formatted as a string).
* `NoToolCallsAndNoFinalAnswer { cycle: usize }` — the assistant response contained neither a tool call nor a "Final Answer:" sentinel.
