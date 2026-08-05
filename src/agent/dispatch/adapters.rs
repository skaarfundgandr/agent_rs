use std::future::Future;
use std::pin::Pin;

use rig_core::completion::CompletionModel;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::managed::BuiltManagedAgent;
use crate::agent::react::BuiltReAct;

use super::definition::{AgentDefinition, AgentInput, AgentKind, AgentOutput};

/// [`AgentDefinition`] wrapping a built ReAct agent; `run` executes
/// [`BuiltReAct::prompt`](crate::agent::react::BuiltReAct::prompt) and
/// reports its trace in the [`AgentOutput`].
pub struct ReActAgentDefinition<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    name: String,
    tool_groups: Vec<String>,
    description: String,
    agent: BuiltReAct<M, ()>,
}

impl<M> ReActAgentDefinition<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Create a definition wrapping a ReAct agent under the given name, tool
    /// groups, and description.
    pub fn new(
        name: String,
        tool_groups: Vec<String>,
        description: String,
        agent: BuiltReAct<M, ()>,
    ) -> Self {
        Self {
            name,
            tool_groups,
            description,
            agent,
        }
    }
}

impl<M> AgentDefinition for ReActAgentDefinition<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> AgentKind {
        AgentKind::ReAct
    }
    fn tool_groups(&self) -> &[String] {
        &self.tool_groups
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn max_retries(&self) -> u32 {
        self.agent.max_retries()
    }
    fn max_cycles(&self) -> Option<usize> {
        Some(self.agent.max_cycles())
    }
    fn react_preamble(&self) -> Option<&str> {
        self.agent.react_preamble()
    }
    fn run<'a>(
        &'a self,
        input: AgentInput,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<AgentOutput>> + Send + 'a>> {
        Box::pin(async move {
            let trace = self
                .agent
                .prompt(input.prompt)
                .await
                .map_err(anyhow::Error::from)?;
            let answer = trace
                .final_answer
                .as_ref()
                .map(|fa| fa.text.clone())
                .unwrap_or_default();
            Ok(AgentOutput {
                answer,
                trace: Some(trace),
            })
        })
    }
}

/// [`AgentDefinition`] wrapping a built managed agent; `run` executes
/// [`BuiltManagedAgent::chat`](crate::agent::managed::BuiltManagedAgent::chat)
/// and reports no trace in the [`AgentOutput`].
pub struct ManagedAgentDefinition<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    name: String,
    tool_groups: Vec<String>,
    description: String,
    agent: BuiltManagedAgent<M>,
}

impl<M> ManagedAgentDefinition<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Create a definition wrapping a managed agent under the given name,
    /// tool groups, and description.
    pub fn new(
        name: String,
        tool_groups: Vec<String>,
        description: String,
        agent: BuiltManagedAgent<M>,
    ) -> Self {
        Self {
            name,
            tool_groups,
            description,
            agent,
        }
    }
}

impl<M> AgentDefinition for ManagedAgentDefinition<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> AgentKind {
        AgentKind::Managed
    }
    fn tool_groups(&self) -> &[String] {
        &self.tool_groups
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn max_retries(&self) -> u32 {
        self.agent.max_retries()
    }
    fn max_cycles(&self) -> Option<usize> {
        None
    }
    fn react_preamble(&self) -> Option<&str> {
        None
    }
    fn run<'a>(
        &'a self,
        input: AgentInput,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<AgentOutput>> + Send + 'a>> {
        Box::pin(async move {
            let mut history = Vec::new();
            let answer = self
                .agent
                .chat(input.prompt, &mut history)
                .await
                .map_err(anyhow::Error::from)?;
            Ok(AgentOutput {
                answer,
                trace: None,
            })
        })
    }
}
