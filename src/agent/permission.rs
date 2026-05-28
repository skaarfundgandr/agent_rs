use std::fmt;
use std::sync::Arc;

#[async_trait::async_trait]
pub trait PermissionGate: Send + Sync {
    async fn check_permission(&self, tool_name: &str, description: &str) -> bool;
}

#[derive(Clone)]
pub enum PermissionPolicy {
    AllowAll,
    DenyAll,
    CliPrompt,
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
