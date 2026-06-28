use std::collections::HashMap;

use anyhow::Context;

use super::definition::{AgentDefinition, AgentInput, AgentOutput};

pub struct AgentDispatcher {
    agents: HashMap<String, Box<dyn AgentDefinition>>,
}

impl Default for AgentDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentDispatcher {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    pub fn register(&mut self, def: Box<dyn AgentDefinition>) -> anyhow::Result<()> {
        let name = def.name().to_string();
        if self.agents.contains_key(&name) {
            anyhow::bail!("agent `{name}` is already registered");
        }
        self.agents.insert(name, def);
        Ok(())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.agents.values().map(|d| d.name())
    }

    pub fn get(&self, name: &str) -> Option<&dyn AgentDefinition> {
        self.agents.get(name).map(|b| b.as_ref())
    }

    pub async fn dispatch(&self, name: &str, input: AgentInput) -> anyhow::Result<AgentOutput> {
        let def = self
            .agents
            .get(name)
            .with_context(|| format!("no agent named `{name}`"))?;
        def.run(input).await
    }
}
