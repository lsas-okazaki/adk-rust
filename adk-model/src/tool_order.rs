//! A deterministic order for the tool declarations of an `LlmRequest`.
//!
//! `LlmRequest::tools` is a [`HashMap`], whose iteration order differs from
//! one map to the next. Sent in that order, the same tools come out shuffled
//! on every request, and since providers render the tool list at the front of
//! the prompt, the shuffle defeats prefix caching (vLLM's automatic prefix
//! cache, Bedrock and Anthropic cache points) on every call. The converters
//! therefore send them sorted by name.

use std::collections::HashMap;

use serde_json::Value;

/// The declarations of `tools`, sorted by tool name.
pub(crate) fn by_name(tools: &HashMap<String, Value>) -> Vec<(&String, &Value)> {
    let mut entries: Vec<(&String, &Value)> = tools.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(names: &[&str]) -> HashMap<String, Value> {
        names.iter().map(|n| (n.to_string(), serde_json::json!({ "name": n }))).collect()
    }

    #[test]
    fn sorts_by_name() {
        let tools = tools(&["get_b", "create_a", "z", "a_tool"]);
        let names: Vec<&str> = by_name(&tools).into_iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["a_tool", "create_a", "get_b", "z"]);
    }

    #[test]
    fn same_tools_in_separate_maps_give_the_same_order() {
        let names: Vec<String> = (0..64).map(|i| format!("tool_{i:02}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let first: Vec<String> =
            by_name(&tools(&refs)).into_iter().map(|(n, _)| n.clone()).collect();
        for _ in 0..8 {
            let again: Vec<String> =
                by_name(&tools(&refs)).into_iter().map(|(n, _)| n.clone()).collect();
            assert_eq!(again, first);
        }
    }
}
