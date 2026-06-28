# Permission System

Controls whether tool execution is allowed, denied, or requires user confirmation. Supports mapping policies per tool, observing decisions for audit trails, and custom gates.

> **Type reference:** See the [class diagram](../diagrams/class-diagram.md) for the `PermissionPolicy` type hierarchy.

---

## `PermissionResult`

Represents the result of evaluating a permission check.

```rust
pub enum PermissionResult {
    /// The operation is allowed.
    Allow,
    /// The operation is denied with a reason.
    Deny { reason: String },
    /// The operation should be deferred to the user (e.g. interactive prompt).
    DeferToUser,
}
```

### Methods
* **`is_allow(&self) -> bool`**
  Returns `true` if the result is `PermissionResult::Allow`.

---

## `PermissionGate` Trait

Implement this trait to define dynamic, custom execution gates for tool invocations.

```rust
#[async_trait::async_trait]
pub trait PermissionGate: Send + Sync {
    /// Checks whether the tool with the given name is permitted to execute.
    ///
    /// # Arguments
    /// * `tool_name` - The unique identifier name of the tool requesting execution.
    /// * `description` - A description of the action the tool wants to perform.
    async fn check_permission(&self, tool_name: &str, description: &str) -> PermissionResult;
}
```

---

## `PermissionPolicy`

Standard execution policies that determine if a tool call should proceed.

```rust
pub enum PermissionPolicy {
    /// Automatically allows every tool execution.
    AllowAll,
    /// Automatically denies every tool execution.
    DenyAll,
    /// Prompts the user interactively on stderr/stdin for permission.
    CliPrompt,
    /// Delegates validation to a custom user-defined `PermissionGate`.
    Custom(Arc<dyn PermissionGate>),
}
```

### Methods
* **`async fn evaluate(&self, tool_name: &str, description: &str) -> PermissionResult`**
  Evaluates the policy for a given tool name and action description.
  * **`CliPrompt` details:** Prompts the user on standard error and reads standard input asynchronously. It is safe to use in async contexts and has a **30-second timeout**. If standard input is not provided within 30 seconds or the user denies the request, it returns a `PermissionResult::Deny`.

---

## `PolicyMap`

A manager that maps tool names to specific `PermissionPolicy` instances, falling back to a default policy when no override is configured. It can also dispatch evaluation results to a `PermissionObserver` for audit trails.

```rust
pub struct PolicyMap {
    default: PermissionPolicy,
    overrides: HashMap<String, PermissionPolicy>,
    observer: Option<Arc<dyn PermissionObserver>>,
}
```

### Methods
* **`new(default: PermissionPolicy) -> Self`**
  Creates a new policy map with a default fallback policy.
* **`tool(mut self, name: impl Into<String>, policy: PermissionPolicy) -> Self`**
  Registers a policy override for a specific tool name (builder pattern).
* **`with_observer(mut self, observer: Arc<dyn PermissionObserver>) -> Self`**
  Registers a permission observer to log or audit evaluations (builder pattern).
* **`async fn evaluate(&self, tool_name: &str, description: &str) -> PermissionResult`**
  Evaluates the policy for a tool name. Resolves to the tool-specific override if present, otherwise falls back to the default policy. Fires the observer if registered.

---

## Audit Trail / Observability

### `PermissionObserver` Trait
Implement this trait to log or record permission checks.

```rust
pub trait PermissionObserver: Send + Sync {
    /// Callback triggered immediately after a permission check has been evaluated.
    fn on_evaluation(&self, event: &PermissionEvent);
}
```

### `PermissionEvent`
Data payload describing a permission check result.

```rust
pub struct PermissionEvent {
    /// Name of the tool being evaluated.
    pub tool_name: String,
    /// Description of the action the tool wanted to perform.
    pub description: String,
    /// The resulting `PermissionResult` of the evaluation.
    pub result: PermissionResult,
    /// Name of the policy variant or `"Override"` that performed the evaluation.
    pub policy_variant: &'static str,
    /// The timestamp when the evaluation occurred.
    pub timestamp: Instant,
}
```

### `LoggingObserver`
A built-in observer that logs evaluations to the `tracing` framework.

```rust
pub struct LoggingObserver;
```
Logs are emitted at `info` level under the target `"permission"` containing structured fields: `tool_name`, `allowed` (boolean), and `policy_variant`.

---

## Example: Policy Map with Overrides & Logging

```rust
use std::sync::Arc;
use agent_rs::agent::permission::{
    PermissionPolicy, PermissionGate, PermissionResult, PolicyMap, LoggingObserver
};

struct MyCustomGate;

#[async_trait::async_trait]
impl PermissionGate for MyCustomGate {
    async fn check_permission(&self, tool_name: &str, _desc: &str) -> PermissionResult {
        if tool_name == "delete_file" {
            PermissionResult::Deny { reason: "Deletion is blocked by corporate safety guidelines".to_string() }
        } else {
            PermissionResult::Allow
        }
    }
}

fn setup_policies() -> PolicyMap {
    PolicyMap::new(PermissionPolicy::CliPrompt) // Default is prompt mode
        .tool("read_document", PermissionPolicy::AllowAll) // Read is always allowed
        .tool("write_document", PermissionPolicy::Custom(Arc::new(MyCustomGate))) // Custom validation for writes
        .with_observer(Arc::new(LoggingObserver)) // Log everything
}
```
