use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelResponse {
    pub content: String,
    pub action: String,
}
