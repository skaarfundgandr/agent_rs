use std::future::Future;
use std::pin::Pin;

use crate::domain::agent::ReActTrace;

#[derive(Debug)]
pub struct AgentInput {
    pub prompt: String,
    pub context: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct AgentOutput {
    pub answer: String,
    pub trace: Option<ReActTrace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    ReAct,
    Managed,
}

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
