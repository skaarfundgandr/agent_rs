use std::collections::HashSet;

use crate::domain::errors::DocumentError;

pub(crate) fn truncated_output(matches: &[String], max_results: usize) -> String {
    let count = matches.len();
    let mut output = matches.join("\n");
    if count >= max_results {
        output.push_str(&format!(
            "\n... [Truncated: reached limit of {} matches]",
            max_results
        ));
    }
    output
}

pub(crate) fn check_extension(ext: &str, allowed: &HashSet<String>) -> Result<(), DocumentError> {
    if !allowed.contains(ext) {
        return Err(DocumentError::UnsupportedExtension(ext.to_string()));
    }
    Ok(())
}
