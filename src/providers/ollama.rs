use anyhow::*;
use rig::client::{Nothing, ProviderClient};
use rig::providers::ollama::Client;

pub struct OllamaProvider {
    pub client: Client,
}

impl OllamaProvider {
    /// Create a new Ollama provider with the specified base URL
    pub fn new(base_url: &str) -> Result<Self> {
        let client = Client::builder()
            .api_key(Nothing)
            .base_url(base_url)
            .build()?;
        Ok(Self { client })
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new("http://localhost:11434")
            .expect("Failed to create Ollama provider with default base URL")
    }
}

impl ProviderClient for OllamaProvider {
    type Input = Nothing;

    fn from_env() -> Self {
        let base_url = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self::new(&base_url).expect("Failed to create Ollama provider from environment variable")
    }

    fn from_val(input: Self::Input) -> Self {
        Self::default()
    }
}
