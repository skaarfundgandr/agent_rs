use std::str::FromStr;

use anyhow::{Context, Result};

use crate::domain::config::McpConfig;

use super::McpRegistry;

impl FromStr for McpRegistry {
    type Err = anyhow::Error;

    /// Parse and validate an MCP configuration, returning an initialized registry.
    ///
    /// # Errors
    ///
    /// This will return an error if the configuration JSON is invalid or if the validation fails.
    fn from_str(config: &str) -> Result<Self, Self::Err> {
        let parsed = McpConfig::from_str(config)?;
        parsed.validate()?;
        Ok(Self::new(parsed))
    }
}

impl FromStr for McpConfig {
    type Err = anyhow::Error;

    /// Parse an MCP configuration from a JSON string.
    ///
    /// Note: This parses the JSON structure but does not validate the server definitions.
    /// Call `.validate()` to validate the configurations.
    fn from_str(config: &str) -> Result<Self, Self::Err> {
        let parsed: Self =
            serde_json::from_str(config).context("failed to parse MCP config JSON")?;
        Ok(parsed)
    }
}
