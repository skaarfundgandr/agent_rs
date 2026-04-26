use async_trait::async_trait;

pub mod ollama;
pub mod lmstudio;

#[async_trait]
pub trait LlmProvider {
    async fn completion(&self, system_prompt: &str, user_message: &str) -> Result<String, Box<dyn std::error::Error>>;
}