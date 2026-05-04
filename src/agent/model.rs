use rig::prelude::ProviderClient;
use rig::tool::Tool;
use crate::domain::agent::ModelResponse;

pub struct Model<P: ProviderClient, T: Tool> {
    pub provider: P,
    pub system_prompt: Option<String>,
    pub role: String,
    pub tools: Option<Vec<T>>,
    pub response: Option<ModelResponse>,
}

pub struct ModelBuilder<P: ProviderClient, T: Tool> {
    provider: P,
    system_prompt: Option<String>,
    role: String,
    tools: Option<Vec<T>>,
}