//! A no-op reasoning tool for ReAct agents.
//!
//! Provides a dedicated space for structured thinking during complex
//! tool-use situations. Does not retrieve information or perform
//! side-effects — it simply echoes the thought back so it appears
//! in the agent's working memory / tool-result channel.
//!
//! Prefer this type over [`rig_core::tool::builtin::ThinkTool`], which uses a
//! phantom error type and returns a bare [`String`].
//!
//! Based on Anthropic's [Think tool](https://www.anthropic.com/engineering/claude-think-tool).
//!
//! # Examples
//!
//! ```rust,ignore
//! use agent_rs::agent::ThinkTool;
//!
//! // Unit struct — no constructor arguments.
//! let agent = model.agent("model-id").tool(ThinkTool).build();
//! ```

use std::future::Future;

use rig_core::tool::Tool;
use serde::{Deserialize, Serialize};

const NAME: &str = "think";

/// Arguments for the `think` tool.
#[derive(Debug, Deserialize)]
pub struct ThinkArgs {
    /// Reasoning, plan, or intermediate analysis to record in working memory.
    pub thought: String,
}

/// Output of a successful `think` invocation.
///
/// Always returns the input thought and `acknowledged: true`. The tool cannot
/// fail (`Tool::Error = Infallible`).
#[derive(Debug, Serialize)]
pub struct ThinkOutput {
    /// Echo of [`ThinkArgs::thought`].
    pub thought: String,
    /// Always `true` on success; confirms the thought was recorded.
    pub acknowledged: bool,
}

/// No-op reasoning tool (`name = "think"`).
///
/// Registers on any rig agent or ReAct loop like other [`Tool`] implementations.
/// Calling it performs no I/O and only returns a structured echo of the input.
///
/// # Errors
///
/// The [`Tool::Error`] associated type is [`std::convert::Infallible`].
/// `call` never returns `Err`.
///
/// # Examples
///
/// ```rust,no_run
/// use agent_rs::agent::tools::{ThinkArgs, ThinkTool};
/// use rig_core::tool::Tool;
///
/// # async fn demo() {
/// let out = ThinkTool
///     .call(ThinkArgs {
///         thought: "verify assumptions before the next search".into(),
///     })
///     .await
///     .expect("Infallible");
/// assert!(out.acknowledged);
/// assert_eq!(out.thought, "verify assumptions before the next search");
/// # }
/// ```
pub struct ThinkTool;

impl Tool for ThinkTool {
    const NAME: &'static str = NAME;
    type Error = std::convert::Infallible;
    type Args = ThinkArgs;
    type Output = ThinkOutput;

    /// Returns the tool description shown to the model in the tool schema.
    fn description(&self) -> String {
        "Use this tool to think through complex reasoning before acting. Records the thought in the agent's working memory.".to_string()
    }

    /// Returns the JSON Schema for [`ThinkArgs`].
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "thought": { "type": "string", "description": "The reasoning or plan to record." }
            },
            "required": ["thought"]
        })
    }

    /// Echoes `args.thought` and sets `acknowledged` to `true`.
    ///
    /// # Arguments
    ///
    /// * `args` - The thought payload from the model.
    ///
    /// # Returns
    ///
    /// A future resolving to [`Ok`] with [`ThinkOutput`].
    ///
    /// # Errors
    ///
    /// Never errors (`Infallible`).
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + rig_core::wasm_compat::WasmCompatSend
    {
        let thought = args.thought;
        std::future::ready(Ok(ThinkOutput {
            thought,
            acknowledged: true,
        }))
    }
}
