use rig_core::agent::{AgentHook, Flow, HookContext, StepEvent};
use rig_core::completion::CompletionModel;

/// Policy for handling tool calls that name an unknown or disallowed tool.
///
/// The hook inspects `StepEvent::InvalidToolCall` and responds according to
/// this policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvalidToolPolicy {
    /// Abort the run with an error.
    Fail,
    /// Inject a corrective feedback message into the model's next turn,
    /// telling it to pick from the allowed tool names.
    #[default]
    Skip,
    /// Re-run the model with the corrective feedback, up to the configured
    /// retry budget.
    Retry,
}

/// Build the corrective feedback text injected when a tool call names an
/// unknown or disallowed tool: an error line listing the exact allowed names
/// and an instruction to call one of them or produce a final answer.
pub fn invalid_tool_feedback(tool_name: &str, allowed_tools: &[String]) -> String {
    let allowed = allowed_tools.join(", ");
    format!(
        "Error: tool `{tool_name}` is unknown or not allowed this turn.\n\
         Use one of these exact names: [{allowed}].\n\
         Do not invent tool names. Call a listed tool or produce a final answer."
    )
}

/// [`AgentHook`] that recovers from `InvalidToolCall` events according to an
/// [`InvalidToolPolicy`], converting a hard failure into corrective feedback
/// for the model.
///
/// Installed automatically by the ReAct and managed builders; other events
/// pass through untouched (`Flow::cont`).
#[derive(Debug, Clone)]
pub struct InvalidToolRecoveryHook {
    policy: InvalidToolPolicy,
}

impl InvalidToolRecoveryHook {
    /// Create a hook with the given recovery policy.
    pub fn new(policy: InvalidToolPolicy) -> Self {
        Self { policy }
    }
}

impl<M: CompletionModel> AgentHook<M> for InvalidToolRecoveryHook {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        let StepEvent::InvalidToolCall(ctx) = event else {
            return Flow::cont();
        };
        let allowed = if !ctx.allowed_tools.is_empty() {
            &ctx.allowed_tools
        } else {
            &ctx.available_tools
        };
        match self.policy {
            InvalidToolPolicy::Skip => Flow::skip(invalid_tool_feedback(&ctx.tool_name, allowed)),
            InvalidToolPolicy::Fail => Flow::fail(),
            InvalidToolPolicy::Retry => Flow::retry(invalid_tool_feedback(&ctx.tool_name, allowed)),
        }
    }
}
