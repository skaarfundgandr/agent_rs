use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// The result of a permission evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionResult {
    /// The operation is allowed.
    Allow,
    /// The operation is denied with a reason.
    Deny { reason: String },
    /// The operation should be deferred to the user (not yet supported).
    DeferToUser,
}

impl PermissionResult {
    /// Returns `true` if this result is `Allow`.
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }
}

/// Trait representing an execution gate that dynamically checks permissions for tool invocations.
#[async_trait::async_trait]
pub trait PermissionGate: Send + Sync {
    /// Checks whether the tool with the given name is permitted to execute.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The identifier name of the tool requesting execution.
    /// * `description` - A description of the action the tool wants to perform.
    ///
    /// # Returns
    ///
    /// Returns a [`PermissionResult`] indicating whether the tool is allowed, denied, or deferred.
    async fn check_permission(&self, tool_name: &str, description: &str) -> PermissionResult;
}

/// Policy defining how tool execution permissions are evaluated.
#[derive(Clone)]
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

impl fmt::Debug for PermissionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PermissionPolicy::AllowAll => write!(f, "PermissionPolicy::AllowAll"),
            PermissionPolicy::DenyAll => write!(f, "PermissionPolicy::DenyAll"),
            PermissionPolicy::CliPrompt => write!(f, "PermissionPolicy::CliPrompt"),
            PermissionPolicy::Custom(_) => write!(f, "PermissionPolicy::Custom(...)"),
        }
    }
}

impl PermissionPolicy {
    /// Evaluates the policy for a given tool invocation.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The identifier name of the tool requesting execution.
    /// * `description` - A description of the action the tool wants to perform.
    ///
    /// # Returns
    ///
    /// Returns a [`PermissionResult`] indicating whether the execution is allowed, denied, or deferred.
    pub async fn evaluate(&self, tool_name: &str, description: &str) -> PermissionResult {
        match self {
            PermissionPolicy::AllowAll => PermissionResult::Allow,
            PermissionPolicy::DenyAll => PermissionResult::Deny {
                reason: "denied by DenyAll policy".to_string(),
            },
            PermissionPolicy::CliPrompt => {
                use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

                eprintln!("\n[Permission] Tool: {tool_name}");
                eprintln!("[Permission] Description: {description}");
                eprint!("[Permission] Allow? [y/N]: ");
                let _ = tokio::io::stderr().flush().await;

                let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
                    let stdin = tokio::io::stdin();
                    let mut reader = BufReader::new(stdin);
                    let mut input = String::new();
                    match reader.read_line(&mut input).await {
                        Ok(_) => Ok(input),
                        Err(e) => Err(e),
                    }
                })
                .await;

                match result {
                    Ok(Ok(input)) => {
                        let trimmed = input.trim().to_lowercase();
                        if trimmed == "y" || trimmed == "yes" {
                            PermissionResult::Allow
                        } else {
                            PermissionResult::Deny {
                                reason: "user denied prompt".to_string(),
                            }
                        }
                    }
                    _ => PermissionResult::Deny {
                        reason: "user denied prompt".to_string(),
                    },
                }
            }
            PermissionPolicy::Custom(gate) => {
                gate.check_permission(tool_name, description).await
            }
        }
    }
}

fn variant_name(p: &PermissionPolicy) -> &'static str {
    match p {
        PermissionPolicy::AllowAll => "AllowAll",
        PermissionPolicy::DenyAll => "DenyAll",
        PermissionPolicy::CliPrompt => "CliPrompt",
        PermissionPolicy::Custom(_) => "Custom",
    }
}

/// Observer trait for monitoring permission policy evaluations.
pub trait PermissionObserver: Send + Sync {
    /// Callback triggered immediately after a permission check has been evaluated.
    ///
    /// # Arguments
    ///
    /// * `event` - The evaluation event details.
    fn on_evaluation(&self, event: &PermissionEvent);
}

/// Event payload containing information about a single permission evaluation.
#[derive(Debug, Clone)]
pub struct PermissionEvent {
    /// The unique identifier name of the tool being evaluated.
    pub tool_name: String,
    /// A description of the action the tool is requesting to perform.
    pub description: String,
    /// The resulting `PermissionResult` of the evaluation.
    pub result: PermissionResult,
    /// The string name of the policy variant that decided this evaluation (e.g. `"AllowAll"`, `"Override"`).
    pub policy_variant: &'static str,
    /// The instant when this evaluation was completed.
    pub timestamp: Instant,
}

/// A policy engine that maps tool names to specific `PermissionPolicy` behaviors,
/// falling back to a default policy when no override is defined.
///
/// Optionally triggers a `PermissionObserver` after each evaluation.
#[derive(Clone)]
pub struct PolicyMap {
    default: PermissionPolicy,
    overrides: HashMap<String, PermissionPolicy>,
    observer: Option<Arc<dyn PermissionObserver>>,
}

impl fmt::Debug for PolicyMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolicyMap")
            .field("default", &self.default)
            .field("override_count", &self.overrides.len())
            .field("has_observer", &self.observer.is_some())
            .finish()
    }
}

impl PolicyMap {
    /// Creates a new `PolicyMap` with the specified default fallback policy.
    ///
    /// # Arguments
    ///
    /// * `default` - The `PermissionPolicy` to use when no override exists for a tool.
    ///
    /// # Returns
    ///
    /// Returns the initialized `PolicyMap` instance.
    pub fn new(default: PermissionPolicy) -> Self {
        Self {
            default,
            overrides: HashMap::new(),
            observer: None,
        }
    }

    /// Configures an explicit `PermissionPolicy` override for a given tool name.
    ///
    /// # Arguments
    ///
    /// * `name` - The unique identifier name of the tool to override.
    /// * `policy` - The `PermissionPolicy` to apply specifically to this tool.
    ///
    /// # Returns
    ///
    /// Returns the updated `PolicyMap` builder instance.
    pub fn tool(mut self, name: impl Into<String>, policy: PermissionPolicy) -> Self {
        self.overrides.insert(name.into(), policy);
        self
    }

    /// Registers a `PermissionObserver` to monitor all evaluations performed by this map.
    ///
    /// # Arguments
    ///
    /// * `observer` - The shared thread-safe observer implementation.
    ///
    /// # Returns
    ///
    /// Returns the updated `PolicyMap` builder instance.
    pub fn with_observer(mut self, observer: Arc<dyn PermissionObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Evaluates the permission policy for a specific tool and action description.
    ///
    /// Looks up any registered override for the tool; if none exists, falls back
    /// to the default policy. Dispatches the evaluation event to the registered
    /// observer, if any.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The identifier name of the tool requesting execution.
    /// * `description` - A description of the action the tool wants to perform.
    ///
    /// # Returns
    ///
    /// Returns a [`PermissionResult`] indicating whether the execution is allowed, denied, or deferred.
    pub async fn evaluate(&self, tool_name: &str, description: &str) -> PermissionResult {
        let (policy, policy_variant): (&PermissionPolicy, &'static str) =
            if let Some(p) = self.overrides.get(tool_name) {
                (p, "Override")
            } else {
                (&self.default, variant_name(&self.default))
            };

        let result = policy.evaluate(tool_name, description).await;

        if let Some(observer) = &self.observer {
            observer.on_evaluation(&PermissionEvent {
                tool_name: tool_name.to_string(),
                description: description.to_string(),
                result: result.clone(),
                policy_variant,
                timestamp: Instant::now(),
            });
        }

        result
    }
}

/// A standard observer implementation that logs evaluations to the `tracing` framework.
///
/// Logs are emitted at `info` level under the target `"permission"`.
pub struct LoggingObserver;

impl PermissionObserver for LoggingObserver {
    /// Logs the evaluation event details via `tracing::info`.
    ///
    /// # Arguments
    ///
    /// * `event` - The details of the evaluation event.
    fn on_evaluation(&self, event: &PermissionEvent) {
        tracing::info!(
            target: "permission",
            tool_name = %event.tool_name,
            allowed = event.result.is_allow(),
            policy_variant = event.policy_variant,
            "permission evaluation"
        );
    }
}
