# Permission System

> **Type reference:** See the [class diagram](../diagrams/class-diagram.md) for the `PermissionPolicy` type hierarchy.

## `PermissionPolicy`

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

---

## `PermissionGate` Trait

```rust
#[async_trait::async_trait]
pub trait PermissionGate: Send + Sync {
    async fn check_permission(&self, tool_name: &str, description: &str) -> bool;
}
```
