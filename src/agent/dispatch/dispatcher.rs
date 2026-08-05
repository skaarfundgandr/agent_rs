use std::collections::HashMap;

use anyhow::Context;

use super::definition::{AgentDefinition, AgentInput, AgentOutput};

/// Registry of named [`AgentDefinition`]s that dispatches an [`AgentInput`]
/// to the agent matching a name.
///
/// Duplicate registration of the same name fails with an error.
pub struct AgentDispatcher {
    agents: HashMap<String, Box<dyn AgentDefinition>>,
}

impl Default for AgentDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentDispatcher {
    /// Create an empty dispatcher.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Register `def` under its [`AgentDefinition::name`].
    ///
    /// # Errors
    ///
    /// Returns an error if an agent with the same name is already registered.
    pub fn register(&mut self, def: Box<dyn AgentDefinition>) -> anyhow::Result<()> {
        let name = def.name().to_string();
        if self.agents.contains_key(&name) {
            anyhow::bail!("agent `{name}` is already registered");
        }
        self.agents.insert(name, def);
        Ok(())
    }

    /// Iterate over the names of all registered agents.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.agents.values().map(|d| d.name())
    }

    /// Look up a registered agent by name.
    pub fn get(&self, name: &str) -> Option<&dyn AgentDefinition> {
        self.agents.get(name).map(|b| b.as_ref())
    }

    /// Run the agent registered under `name` with `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if no agent is registered under `name`, or if the
    /// agent's run fails.
    pub async fn dispatch(&self, name: &str, input: AgentInput) -> anyhow::Result<AgentOutput> {
        let def = self
            .agents
            .get(name)
            .with_context(|| format!("no agent named `{name}`"))?;
        def.run(input).await
    }
}
