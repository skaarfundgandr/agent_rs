use rig_core::agent::{AgentHook, Flow, HookContext, StepEvent};
use rig_core::completion::CompletionModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvalidToolPolicy {
    Fail,
    #[default]
    Skip,
    Retry,
}

pub fn invalid_tool_feedback(tool_name: &str, allowed_tools: &[String]) -> String {
    let allowed = allowed_tools.join(", ");
    format!(
        "Error: tool `{tool_name}` is unknown or not allowed this turn.\n\
         Use one of these exact names: [{allowed}].\n\
         Do not invent tool names. Call a listed tool or produce a final answer."
    )
}

#[derive(Debug, Clone)]
pub struct InvalidToolRecoveryHook {
    policy: InvalidToolPolicy,
}

impl InvalidToolRecoveryHook {
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
