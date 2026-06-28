use std::collections::HashSet;
use std::sync::Arc;

use rig_core::tool::ToolDyn;

use crate::mcp::registry::McpRegistryRuntime;

/// A factory closure that produces a fresh `Box<dyn ToolDyn>` on each call.
///
/// Used by [`ToolRegistry`] to support repeated `active_tools()` invocations
/// without consuming the registered tools.
pub type ToolFactory = Arc<dyn Fn() -> Box<dyn ToolDyn> + Send + Sync>;

/// A registered tool entry carrying its group assignment and factory.
pub struct RegisteredTool {
    /// The logical group this tool belongs to (e.g. `"mcp"`, `"filesystem"`).
    pub group: String,
    /// The tool's unique name (read from `ToolDyn::name()` at registration time).
    pub tool_name: String,
    /// Factory closure producing a fresh boxed tool instance.
    pub factory: ToolFactory,
}

/// Builder for constructing a [`ToolRegistry`] with named groups and duplicate detection.
pub struct ToolRegistryBuilder {
    entries: Vec<RegisteredTool>,
    enabled: HashSet<String>,
    seen_names: HashSet<String>,
}

impl std::fmt::Debug for ToolRegistryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistryBuilder")
            .field("entries", &self.entries.len())
            .field("enabled", &self.enabled)
            .field("seen_names", &self.seen_names)
            .finish()
    }
}

impl Default for ToolRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistryBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            enabled: HashSet::new(),
            seen_names: HashSet::new(),
        }
    }

    /// Register an internal tool under a named group.
    ///
    /// The factory is called once to probe the tool name; subsequent calls happen
    /// only via [`ToolRegistry::active_tools`]. Returns `Err` if a tool with the
    /// same name has already been registered (library-level collision detection).
    pub fn register(
        mut self,
        group: &str,
        tool_factory: impl Fn() -> Box<dyn ToolDyn> + Send + Sync + 'static,
    ) -> anyhow::Result<Self> {
        let probe = tool_factory();
        let name = probe.name();
        anyhow::ensure!(
            !self.seen_names.contains(&name),
            "duplicate tool `{name}` already registered"
        );
        self.seen_names.insert(name.clone());
        self.entries.push(RegisteredTool {
            group: group.into(),
            tool_name: name,
            factory: Arc::new(tool_factory),
        });
        Ok(self)
    }

    /// Register all tools from a connected MCP runtime under one group.
    ///
    /// Borrows the runtime without consuming it — uses
    /// [`McpRegistryRuntime::tools`] to iterate over registered tools.
    pub fn register_mcp(
        mut self,
        group: &str,
        runtime: &McpRegistryRuntime,
    ) -> anyhow::Result<Self> {
        for tool in runtime.tools() {
            let name = tool.tool_name().to_owned();
            anyhow::ensure!(
                !self.seen_names.contains(&name),
                "duplicate MCP tool `{name}` already registered"
            );
            self.seen_names.insert(name.clone());
            let tool_clone = tool.clone();
            self.entries.push(RegisteredTool {
                group: group.into(),
                tool_name: name,
                factory: Arc::new(move || Box::new(tool_clone.clone()) as Box<dyn ToolDyn>),
            });
        }
        Ok(self)
    }

    /// Mark the given groups as enabled.
    pub fn enable(mut self, groups: &[&str]) -> Self {
        self.enabled.extend(groups.iter().map(|s| s.to_string()));
        self
    }

    /// Consume the builder and produce a [`ToolRegistry`].
    pub fn build(self) -> ToolRegistry {
        ToolRegistry {
            entries: self.entries,
            enabled: self.enabled,
        }
    }
}

/// A registry of tools partitioned into named groups with runtime enable/disable.
pub struct ToolRegistry {
    entries: Vec<RegisteredTool>,
    enabled: HashSet<String>,
}

impl ToolRegistry {
    /// Return boxed tool instances for every entry whose group is currently enabled.
    ///
    /// Can be called repeatedly — factories produce fresh instances each time.
    pub fn active_tools(&self) -> Vec<Box<dyn ToolDyn>> {
        self.entries
            .iter()
            .filter(|e| self.enabled.contains(&e.group))
            .map(|e| (e.factory)())
            .collect()
    }

    /// Disable a group, removing its tools from [`active_tools`](Self::active_tools).
    ///
    /// Idempotent — disabling a missing group is a no-op.
    pub fn disable_group(&mut self, group: &str) {
        self.enabled.remove(group);
    }

    /// Enable a group, restoring its tools to [`active_tools`](Self::active_tools).
    pub fn enable_group(&mut self, group: &str) {
        self.enabled.insert(group.into());
    }

    /// Return the sorted, deduplicated list of group names present in the registry.
    pub fn groups(&self) -> Vec<&str> {
        let mut groups: Vec<&str> = self.entries.iter().map(|e| e.group.as_str()).collect();
        groups.sort_unstable();
        groups.dedup();
        groups
    }
}
