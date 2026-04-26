use rig::prelude::*;

pub struct LMStudioProvider {
    pub base_url: String,
    pub api_key: Option<String>,
    pub http_client: reqwest::Client,
}

impl LMStudioProvider {
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.to_string(),
            api_key,
            http_client: reqwest::Client::new(),
        }
    }
    pub fn builder() -> LMStudioBuilder {
        LMStudioBuilder::new()
    }
}
#[derive(Default)]
pub struct LMStudioBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
}

impl LMStudioBuilder {
    pub fn new() -> Self {
        Self {
            base_url: None,
            api_key: None,
        }
    }

    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = Some(url.to_string());
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    pub fn build(self) -> LMStudioProvider {
        let base_url = self.base_url.unwrap_or_else(|| "http://localhost:8000".to_string());
        LMStudioProvider::new(&base_url, self.api_key)
    }
}

impl ProviderClient for LMStudioProvider {
    type Input = (String, Option<String>); // (base_url, api_key)

    fn from_env() -> Self {
        let base_url = std::env::var("LMSTUDIO_BASE_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
        let api_key = std::env::var("LMSTUDIO_API_KEY").ok();
        Self::new(&base_url, api_key)
    }

    fn from_val(input: Self::Input) -> Self {
        let base_url = input.0;
        let api_key = input.1;
        
        Self::new(&base_url, api_key)
    }
}
