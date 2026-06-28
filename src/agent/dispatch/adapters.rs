use std::future::Future;
use std::pin::Pin;

use rig_core::agent::PromptHook;
use rig_core::completion::CompletionModel;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::managed::BuiltManagedAgent;
use crate::agent::react::BuiltReAct;

use super::definition::{AgentDefinition, AgentInput, AgentKind, AgentOutput};

pub struct ReActAgentDefinition<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    name: String,
    tool_groups: Vec<String>,
    description: String,
    agent: BuiltReAct<M, P, ()>,
}

impl<M, P> ReActAgentDefinition<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new(
        name: String,
        tool_groups: Vec<String>,
        description: String,
        agent: BuiltReAct<M, P, ()>,
    ) -> Self {
        Self {
            name,
            tool_groups,
            description,
            agent,
        }
    }
}

impl<M, P> AgentDefinition for ReActAgentDefinition<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
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

pub struct ManagedAgentDefinition<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    name: String,
    tool_groups: Vec<String>,
    description: String,
    agent: BuiltManagedAgent<M, P>,
}

impl<M, P> ManagedAgentDefinition<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new(
        name: String,
        tool_groups: Vec<String>,
        description: String,
        agent: BuiltManagedAgent<M, P>,
    ) -> Self {
        Self {
            name,
            tool_groups,
            description,
            agent,
        }
    }
}

impl<M, P> AgentDefinition for ManagedAgentDefinition<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
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
            let answer = self
                .agent
                .chat(input.prompt)
                .await
                .map_err(anyhow::Error::from)?;
            Ok(AgentOutput {
                answer,
                trace: None,
            })
        })
    }
}
