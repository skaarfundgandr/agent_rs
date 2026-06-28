# Migration Guide: v0.1.0 → v0.2.0

## Overview

v0.2.0 introduces path sandboxing and customizable file extension constraints to `ReadDocumentTool` and `WriteDocumentTool` to address critical security vulnerabilities (H1 & H2 path traversals). 

As a result, both tools have been changed from unit structs to structured types requiring explicit initialization parameters.

---

## 🚨 Breaking Changes

### 1. Tool Instantiation

In `v0.1.0`, `ReadDocumentTool` and `WriteDocumentTool` were unit structs without any fields or constructors. In `v0.2.0`, both tools are structured types and must be constructed via `::new(...)` with a sandbox configuration, allowed extensions, and a permission policy.

#### `ReadDocumentTool::new`

**Before (v0.1.0):**
```rust
let tool = ReadDocumentTool;
```

**After (v0.2.0):**
```rust
use std::collections::HashSet;
use agent_rs::security::SandboxConfig;
use agent_rs::agent::permission::PermissionPolicy;

let sandbox = SandboxConfig::single("./sandbox").unwrap();
let allowed_extensions = HashSet::from(["txt".to_string(), "md".to_string(), "pdf".to_string()]);
let policy = PermissionPolicy::AllowAll;
let tool = ReadDocumentTool::new(sandbox, allowed_extensions, policy);
```

#### `WriteDocumentTool::new`

**Before (v0.1.0):**
```rust
let tool = WriteDocumentTool;
```

**After (v0.2.0):**
```rust
use std::collections::HashSet;
use agent_rs::security::SandboxConfig;
use agent_rs::agent::permission::PermissionPolicy;

let sandbox = SandboxConfig::single("./sandbox").unwrap();
let allowed_extensions = HashSet::from(["txt".to_string(), "md".to_string()]);
let policy = PermissionPolicy::AllowAll;
let tool = WriteDocumentTool::new(sandbox, allowed_extensions, policy);
```

### 2. Permission System

`ReadDocumentTool`, `WriteDocumentTool`, `ListDirectoryTool`, `GlobSearchTool`, and `GrepSearchTool` now require a `PermissionPolicy` parameter in their constructors (typically `PermissionPolicy::AllowAll` for unrestricted use).

**Before (v0.1.0/v0.2.0-beta):**
```rust
let reader = ReadDocumentTool::new(sandbox, read_extensions);
let lister = ListDirectoryTool::new(sandbox);
```

**After (v0.2.0):**
```rust
use agent_rs::security::SandboxConfig;
use agent_rs::agent::permission::PermissionPolicy;

let sandbox = SandboxConfig::single("./sandbox").unwrap();
let policy = PermissionPolicy::AllowAll;
let reader = ReadDocumentTool::new(sandbox.clone(), read_extensions, policy.clone());
let lister = ListDirectoryTool::new(sandbox, policy);
```

The policy controls tool execution at runtime. Available variants:
- `AllowAll` — permits every call (backward-compatible default).
- `DenyAll` — denies every call.
- `CliPrompt` — interactively prompts the user via stderr/stdin.
- `Custom(Arc<dyn PermissionGate>)` — delegates to user-defined logic.

### 3. PermissionDenied Error Variant

A new error variant `DocumentError::PermissionDenied(String)` has been added. If you manually match on `DocumentError`, you must handle this new variant.

### 4. Sandbox Escape Error Variant

A new error variant `DocumentError::SandboxEscape` has been added to `DocumentError` representing unauthorized path traversal attempts outside the configured sandbox root folder. If you manually match on `DocumentError`, you must handle this new variant.

---

## 🛠️ Migration Steps

### Step 1: Add Imports

Ensure you import `HashSet`, `SandboxConfig`, `PermissionPolicy` and the new error variant if you pattern match on document errors:

```rust
use std::collections::HashSet;
use agent_rs::security::SandboxConfig;
use agent_rs::agent::permission::PermissionPolicy;
use agent_rs::agent::tools::{ReadDocumentTool, WriteDocumentTool};
use agent_rs::domain::errors::DocumentError;
```

### Step 2: Configure Sandbox and Allowed Extensions

Define your sandbox configuration and the set of allowed file extensions (without leading dots):

```rust
use agent_rs::security::SandboxConfig;
use agent_rs::agent::permission::PermissionPolicy;

let sandbox = SandboxConfig::single("./data").unwrap();
let policy = PermissionPolicy::AllowAll;

// Read: txt, md, and pdf
let read_extensions = HashSet::from(["txt", "md", "pdf"].map(String::from));
let reader = ReadDocumentTool::new(sandbox.clone(), read_extensions, policy.clone());

// Write: only txt and md
let write_extensions = HashSet::from(["txt", "md"].map(String::from));
let writer = WriteDocumentTool::new(sandbox, write_extensions, policy);
```

> **Note on PDF support**: PDF files are read-only and extracted using `pdf-extract`. Including `"pdf"` in write extensions will only perform a plain-text write and will not build a valid PDF file.

---

## 🔒 Behavioral Changes

- **Path Verification**: Target paths are canonicalized to resolve symlinks and relative directories (`../`). Any path resolving outside the canonicalized sandbox root returns `DocumentError::SandboxEscape`.
- **Extension Filtering**: File extensions are checked before execution. Any file access with an extension not present in `allowed_extensions` returns `DocumentError::UnsupportedExtension`.
- **Dynamic Tool Description**: The supported extensions list in the tool definitions is now dynamically generated from the configured `allowed_extensions` set.

> **Architecture reference:** See the [sandbox path validation flowchart](diagrams/flowchart.md) for a visual walkthrough of the canonicalization logic, and the [class diagram](diagrams/class-diagram.md) for the `DocumentError` enum definition.
