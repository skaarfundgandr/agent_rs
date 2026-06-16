# Security Sandbox

Multi-root filesystem sandboxing. Controls which paths tools can access and where new files are written.

> **Architecture reference:** See the [sandbox flowchart](../diagrams/flowchart.md) for the path resolution algorithm, and the [module dependency graph](../diagrams/module-dependency.md) for how `security::sandbox` fits into the crate.

## `SandboxConfig`

Stores an ordered list of allowed filesystem roots. The first root is the **primary** — used as the default target when creating new files. All roots participate equally in read/search/glob operations.

```rust
pub struct SandboxConfig {
    roots: Vec<PathBuf>,
    canonical_roots: Vec<PathBuf>,
}
```

### Methods
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

### Trait Implementations
* `Default` — uses `"."` as a single root.
* `TryFrom<PathBuf>`, `TryFrom<&Path>`, `TryFrom<&str>` — each creates a single-root sandbox. Returns an error if the root cannot be canonicalized.

---

## `validate_sandboxed_path()`

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

---

## `find_containing_root()`

```rust
pub fn find_containing_root<'a>(
    sandbox: &'a SandboxConfig,
    path: &Path,
) -> Option<&'a PathBuf>
```
Returns the original (non-canonicalized) root that contains `path`, or `None` if the path does not fall under any root. Uses original roots for comparison to avoid Windows `\\?\` prefix issues.

---

## `relative_display_path()`

```rust
pub fn relative_display_path(sandbox: &SandboxConfig, path: &Path) -> String
```
Computes a relative display path for a file, preferring the shortest prefix among the sandbox roots for readability. Falls back to the full path if no root matches.

---

## Environment Variables

* **`SANDBOX_ROOTS`** — comma-separated list of allowed filesystem paths. The first path is the primary root (default for writes). Example: `SANDBOX_ROOTS="./,/tmp/shared,/home/user/docs"`. Defaults to `"./"` if unset.
