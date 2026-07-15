#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use agent_rs::agent::tools::ToolRegistryBuilder;
use agent_rs::mcp::registry::McpRegistryRuntime;
use rig_core::tool::{Tool, ToolDyn};

// ---------------------------------------------------------------------------
// Mock tools
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MockToolA;

impl Tool for MockToolA {
    const NAME: &'static str = "tool_a";
    type Error = std::convert::Infallible;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "mock tool a".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("a".to_string())
    }
}

#[derive(Clone)]
struct MockToolB;

impl Tool for MockToolB {
    const NAME: &'static str = "tool_b";
    type Error = std::convert::Infallible;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "mock tool b".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("b".to_string())
    }
}

#[derive(Clone)]
struct MockToolC;

impl Tool for MockToolC {
    const NAME: &'static str = "tool_c";
    type Error = std::convert::Infallible;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "mock tool c".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("c".to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn group_enable_disable_toggles_active_tools() {
    let registry = ToolRegistryBuilder::new()
        .register("g1", || Box::new(MockToolA) as Box<dyn ToolDyn>)
        .unwrap()
        .register("g2", || Box::new(MockToolB) as Box<dyn ToolDyn>)
        .unwrap()
        .enable(&["g1"])
        .build();

    let tools = registry.active_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "tool_a");
}

#[test]
fn duplicate_tool_name_returns_err() {
    let result = ToolRegistryBuilder::new()
        .register("g1", || Box::new(MockToolA) as Box<dyn ToolDyn>)
        .unwrap()
        .register("g2", || Box::new(MockToolA) as Box<dyn ToolDyn>);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("duplicate tool `tool_a`"));
}

#[test]
fn register_mcp_borrows_without_consuming() {
    let runtime = McpRegistryRuntime::default();
    let registry = ToolRegistryBuilder::new()
        .register_mcp("mcp", &runtime)
        .unwrap()
        .build();

    // Runtime is still usable after register_mcp
    let _ = runtime.tools();
    let _ = runtime.tool_names();

    // Registry has no tools (empty runtime)
    assert!(registry.active_tools().is_empty());
}

#[test]
fn empty_registry_returns_empty_vec() {
    let registry = ToolRegistryBuilder::new().build();
    assert!(registry.active_tools().is_empty());
}

#[test]
fn groups_returns_dedup_sorted() {
    let registry = ToolRegistryBuilder::new()
        .register("beta", || Box::new(MockToolA) as Box<dyn ToolDyn>)
        .unwrap()
        .register("alpha", || Box::new(MockToolB) as Box<dyn ToolDyn>)
        .unwrap()
        .register("beta", || Box::new(MockToolC) as Box<dyn ToolDyn>)
        .unwrap()
        .build();

    let groups = registry.groups();
    assert_eq!(groups, vec!["alpha", "beta"]);
}

#[test]
fn disable_group_is_idempotent_on_missing() {
    let mut registry = ToolRegistryBuilder::new()
        .register("g1", || Box::new(MockToolA) as Box<dyn ToolDyn>)
        .unwrap()
        .enable(&["g1"])
        .build();

    registry.disable_group("g1");
    registry.disable_group("nonexistent"); // should not panic
    assert!(registry.active_tools().is_empty());
}

#[test]
fn re_enabling_group_restores_membership() {
    let mut registry = ToolRegistryBuilder::new()
        .register("g1", || Box::new(MockToolA) as Box<dyn ToolDyn>)
        .unwrap()
        .enable(&["g1"])
        .build();

    assert_eq!(registry.active_tools().len(), 1);
    registry.disable_group("g1");
    assert!(registry.active_tools().is_empty());
    registry.enable_group("g1");
    assert_eq!(registry.active_tools().len(), 1);
}

#[test]
fn active_tools_can_be_called_multiple_times() {
    let registry = ToolRegistryBuilder::new()
        .register("g1", || Box::new(MockToolA) as Box<dyn ToolDyn>)
        .unwrap()
        .enable(&["g1"])
        .build();

    let first = registry.active_tools();
    let second = registry.active_tools();
    assert_eq!(first.len(), second.len());
    assert_eq!(first[0].name(), second[0].name());
}
