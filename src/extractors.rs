//! Pluggable structural extraction for `parse_structure`, keyed by file
//! extension. The tool's dispatch logic never changes when a new extraction
//! strategy shows up — a new `StructuralExtractor` impl gets registered in
//! `main()` and takes over for the extensions it declares.
//!
//! Whole-file (not line-by-line) granularity: the trait takes the entire
//! file's content and returns structural nodes directly, rather than being
//! asked "is this one line a declaration?" per line. That's a deliberate
//! choice — a future AST-based extractor doesn't work line-by-line at all,
//! so a line-grained trait would only fit the regex approach.

use std::collections::HashMap;
use std::sync::Arc;

/// Extracts structural nodes (functions, classes, imports, ...) from a
/// whole file's content. Each returned value is shaped
/// `{"line": N, "declaration": "..."}` — `parse_structure`'s existing
/// output contract.
pub trait StructuralExtractor: Send + Sync {
    /// Lowercase file extensions (without the dot) this extractor handles.
    fn extensions(&self) -> &[&str];
    fn extract(&self, ext: &str, content: &str) -> Vec<serde_json::Value>;
}

/// The original regex-based extractor (`lib.rs`'s `matches_structural_line`),
/// wrapped behind the trait. `lib.rs`'s functions are untouched — this is a
/// pure move of `ParseStructureTool::execute`'s former inline loop, not a
/// behavior change.
pub struct RegexExtractor;

impl StructuralExtractor for RegexExtractor {
    fn extensions(&self) -> &[&str] {
        &["rs", "py", "cs"]
    }

    fn extract(&self, ext: &str, content: &str) -> Vec<serde_json::Value> {
        let mut structural_nodes = Vec::new();
        for (idx, line) in content.lines().enumerate() {
            let line_trimmed = line.trim();
            if line_trimmed.is_empty() || line_trimmed.starts_with("//") || line_trimmed.starts_with("#") {
                continue;
            }
            if mcp_native_core::matches_structural_line(ext, line) {
                structural_nodes.push(serde_json::json!({
                    "line": idx + 1,
                    "declaration": line_trimmed
                }));
            }
        }
        structural_nodes
    }
}

/// Maps a file extension to the extractor registered for it. Registering a
/// new extractor for an extension that already has one overwrites it —
/// last-registered-wins, no separate priority mechanism needed for v1.
pub struct ExtractorRegistry {
    by_extension: HashMap<String, Arc<dyn StructuralExtractor>>,
}

impl ExtractorRegistry {
    pub fn new() -> Self {
        Self { by_extension: HashMap::new() }
    }

    pub fn register(&mut self, extractor: Arc<dyn StructuralExtractor>) {
        for ext in extractor.extensions() {
            self.by_extension.insert(ext.to_string(), extractor.clone());
        }
    }

    pub fn get(&self, ext: &str) -> Option<Arc<dyn StructuralExtractor>> {
        self.by_extension.get(ext).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_extractor_matches_prior_inline_behavior() {
        let extractor = RegexExtractor;
        let content = "\
// a comment, skipped
pub fn foo(x: i32) -> i32 {
    x + 1
}

struct Bar {
    field: i32,
}
";
        let nodes = extractor.extract("rs", content);
        assert_eq!(
            nodes,
            vec![
                serde_json::json!({"line": 2, "declaration": "pub fn foo(x: i32) -> i32 {"}),
                serde_json::json!({"line": 6, "declaration": "struct Bar {"}),
            ]
        );
    }

    #[test]
    fn regex_extractor_returns_empty_for_unknown_extension() {
        let extractor = RegexExtractor;
        assert!(extractor.extract("txt", "fn foo() {}").is_empty());
    }

    #[test]
    fn registry_returns_registered_extractor() {
        let mut registry = ExtractorRegistry::new();
        registry.register(Arc::new(RegexExtractor));

        assert!(registry.get("rs").is_some());
        assert!(registry.get("py").is_some());
        assert!(registry.get("cs").is_some());
        assert!(registry.get("txt").is_none());
    }

    struct StubExtractor;
    impl StructuralExtractor for StubExtractor {
        fn extensions(&self) -> &[&str] {
            &["rs"]
        }
        fn extract(&self, _ext: &str, _content: &str) -> Vec<serde_json::Value> {
            vec![serde_json::json!({"line": 1, "declaration": "stub"})]
        }
    }

    #[test]
    fn registering_a_new_extractor_for_the_same_extension_overwrites_the_old_one() {
        let mut registry = ExtractorRegistry::new();
        registry.register(Arc::new(RegexExtractor));
        registry.register(Arc::new(StubExtractor));

        let nodes = registry.get("rs").unwrap().extract("rs", "pub fn foo() {}");
        assert_eq!(nodes, vec![serde_json::json!({"line": 1, "declaration": "stub"})]);
    }
}
