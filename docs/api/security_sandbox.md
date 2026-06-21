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
* **`add_root(&mut self, root: impl AsRef<Path>) -> Result<(), DocumentError>`**
  Appends a root, canonicalizing it. If a root with the same canonical form already exists, the call is a no-op (idempotent). Returns `Io` if the path cannot be canonicalized.
* **`add_roots(&mut self, roots: impl IntoIterator<Item = impl AsRef<Path>>) -> Result<(), DocumentError>`**
  Batch append. Per-item atomicity: a successful `add_root` mutates before the next item is validated. A failing item after a successful one leaves partial state — callers wanting all-or-nothing must validate paths externally. Returns the first `Io` error encountered.
* **`remove_root(&mut self, root: impl AsRef<Path>) -> Result<(), DocumentError>`**
  Removes a root by canonical-form lookup. If not found, the call is a no-op (idempotent on miss). Removing the **last** root returns `DocumentError::Sandbox` (the sandbox invariant requires at least one root; use `set` to swap to a fresh config instead). Returns `Io` if the path cannot be canonicalized.
* **`contains_root(&self, root: impl AsRef<Path>) -> Result<bool, DocumentError>`**
  Strict canonical-form membership check. Returns `Ok(true)`/`Ok(false)` if the path canonicalizes, `Err(Io)` if it cannot.

### Trait Implementations
* `Default` — uses `"."` as a single root.
* `TryFrom<PathBuf>`, `TryFrom<&Path>`, `TryFrom<&str>` — each creates a single-root sandbox. Returns an error if the root cannot be canonicalized.

---

## `SharedSandbox`

Thread-safe, cheaply-cloneable handle to a `SandboxConfig` that supports runtime hot-swapping of sandbox roots. Wraps `Arc<RwLock<SandboxConfig>>`.

```rust
pub struct SharedSandbox { /* private */ }
```

### Methods
* **`new(initial: SandboxConfig) -> Self`** — wraps an initial config.
* **`snapshot(&self) -> SandboxConfig`** — clones the current config under a read lock. Cheap (two `Vec<PathBuf>`); held lock is brief.
* **`set(&self, new_config: SandboxConfig) -> Result<(), DocumentError>`** — replaces the inner config after re-canonicalizing all roots. Returns `Rag` if the new config has no roots, `Io` if any root cannot be canonicalized. **Full-replacement escape hatch** — prefer the incremental mutators below when you only want to add or remove a single root.
* **`add_root(&self, root: impl AsRef<Path>) -> Result<(), DocumentError>`** — appends a root under a write lock. Idempotent on canonical-form dedup. Returns `Io` if the path cannot be canonicalized.
* **`add_roots<I, P>(&self, roots: I) -> Result<(), DocumentError>`** — batch append under a single write lock. Per-item atomicity: a successful `add_root` mutates before the next item is validated. Holds the write lock for the whole iterator (block readers briefly; roots lists are small).
* **`remove_root(&self, root: impl AsRef<Path>) -> Result<(), DocumentError>`** — removes a root under a write lock. Idempotent on miss. Returns `Io` if the path cannot be canonicalized, `Sandbox` if it would leave the sandbox empty.
* **`contains_root(&self, root: impl AsRef<Path>) -> Result<bool, DocumentError>`** — strict canonical-form membership check under a read lock. Returns `Ok(true)`/`Ok(false)` if the path canonicalizes, `Err(Io)` otherwise.

### Trait Implementations
* `Clone` — clones the `Arc` (cheap), shares the same inner config.
* `Default` — uses `"."` as a single root, same as `SandboxConfig::default()`.
* `From<SandboxConfig>` and `From<&SandboxConfig>` — convenient conversion.

### When to use
Use `SharedSandbox` when sandbox roots may change after tools are constructed (e.g., an operator-facing reload mechanism). Tools that hold an `Arc<SharedSandbox>` automatically pick up the new roots on the next call. For one-shot sandboxes, plain `SandboxConfig` is simpler.

### Poisoning
Uses `std::sync::RwLock`. A panic while holding a write lock will poison the lock; subsequent `snapshot`/`set` calls panic. This is deliberate — a poisoned sandbox indicates a logic error that should be surfaced.

---

## `validate_sandboxed_path_shared()`

```rust
pub fn validate_sandboxed_path_shared(
    sandbox: &SharedSandbox,
    user_path: &Path,
) -> Result<PathBuf, DocumentError>
```

Snapshot-then-validate variant of `validate_sandboxed_path`. See the non-shared version for the algorithm.

---

## `find_containing_root_shared()`

```rust
pub fn find_containing_root_shared(
    sandbox: &SharedSandbox,
    path: &Path,
) -> Option<PathBuf>
```

Snapshot-then-find variant of `find_containing_root`. **Returns `Option<PathBuf>` (owned), not `Option<&PathBuf>`,** because the snapshot is a temporary that cannot be safely borrowed.

---

## `relative_display_path_shared()`

```rust
pub fn relative_display_path_shared(
    sandbox: &SharedSandbox,
    path: &Path,
) -> String
```

Snapshot-then-display variant of `relative_display_path`.

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
