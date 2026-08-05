use std::future::Future;
use std::pin::Pin;

use crate::domain::agent::ReActTrace;

/// Input to an [`AgentDefinition::run`] call: the user prompt plus optional
/// free-form JSON context.
#[derive(Debug)]
pub struct AgentInput {
    pub prompt: String,
    pub context: Option<serde_json::Value>,
}

/// Output of an [`AgentDefinition::run`] call: the final answer text plus,
/// for ReAct agents, the full execution trace.
#[derive(Debug)]
pub struct AgentOutput {
    pub answer: String,
    pub trace: Option<ReActTrace>,
}

/// The execution style of an agent: ReAct (reasoning + acting loop) or
/// managed (single multi-turn prompt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    ReAct,
    Managed,
}

/// Describes an agent that can be registered with an
/// [`AgentDispatcher`](crate::agent::dispatch::AgentDispatcher) and invoked
/// with an [`AgentInput`].
///
/// Implementations are typically created by wrapping a
/// [`ReActAgentDefinition`](crate::agent::dispatch::ReActAgentDefinition) or
/// [`ManagedAgentDefinition`](crate::agent::dispatch::ManagedAgentDefinition)
/// around a built agent.
pub trait AgentDefinition: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> AgentKind;
    fn tool_groups(&self) -> &[String];
    fn description(&self) -> &str;
    fn max_retries(&self) -> u32;
    fn max_cycles(&self) -> Option<usize>;
    fn react_preamble(&self) -> Option<&str>;

    fn run<'a>(
        &'a self,
        input: AgentInput,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<AgentOutput>> + Send + 'a>>;
}
