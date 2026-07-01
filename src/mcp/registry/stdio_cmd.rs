use anyhow::Result;

use crate::domain::mcp::McpStdioTransportSpec;

pub(crate) fn build_stdio_command(spec: &McpStdioTransportSpec) -> Result<std::process::Command> {
    let mut command = std::process::Command::new(&spec.command);
    command
        .args(&spec.args)
        .envs(&spec.env)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    if let Some(cwd) = &spec.cwd {
        command.current_dir(cwd);
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }

    Ok(command)
}
