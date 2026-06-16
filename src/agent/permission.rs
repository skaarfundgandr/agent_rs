use std::fmt;
use std::sync::Arc;

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
    /// Returns `true` if the tool is allowed to execute, or `false` otherwise.
    async fn check_permission(&self, tool_name: &str, description: &str) -> bool;
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
    /// Returns `true` if the policy allows the execution, or `false` otherwise.
    pub async fn evaluate(&self, tool_name: &str, description: &str) -> bool {
        match self {
            PermissionPolicy::AllowAll => true,
            PermissionPolicy::DenyAll => false,
            PermissionPolicy::CliPrompt => {
                eprintln!("\n[Permission] Tool: {tool_name}");
                eprintln!("[Permission] Description: {description}");
                eprint!("[Permission] Allow? [y/N]: ");
                use std::io::Write;
                std::io::stdout().flush().ok();
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).ok();
                let input = input.trim().to_lowercase();
                input == "y" || input == "yes"
            }
            PermissionPolicy::Custom(gate) => gate.check_permission(tool_name, description).await,
        }
    }
}
