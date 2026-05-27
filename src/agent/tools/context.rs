use crate::domain::errors::CompactError;
use rig::completion::{Prompt, ToolDefinition};
use rig::tool::Tool;
use rig::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, Serialize)]
pub struct CompactArgs {
    pub text: String,
}

pub struct CompactTool<M: Prompt + WasmCompatSend + WasmCompatSync + 'static> {
    model: M,
}

impl<M: Prompt + WasmCompatSend + WasmCompatSync + 'static> std::fmt::Debug for CompactTool<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactTool").field("model", &"M").finish()
    }
}

impl<M: Prompt + WasmCompatSend + WasmCompatSync + 'static> CompactTool<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }
}

impl<M: Prompt + WasmCompatSend + WasmCompatSync + 'static> Tool for CompactTool<M> {
    const NAME: &'static str = "compact";

    type Error = CompactError;
    type Args = CompactArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Summarize the current conversation history to save tokens while preserving key information.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The conversation history or text to summarize"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn call(&self, args: CompactArgs) -> Result<Self::Output, Self::Error> {
        let prompt_text = format!(
            "Summarize the following conversation history while preserving key information, \
            names, dates, and important technical details. Keep the summary concise:\n\n{}",
            args.text
        );

        let response = self
            .model
            .prompt(&prompt_text)
            .await
            .map_err(|e| CompactError::Model(e.to_string()))?;

        Ok(response)
    }
}
