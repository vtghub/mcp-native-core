//! Pluggable content-matching for `fast_search`, selected by an optional
//! `backend` request parameter (default `"regex"`). Two-level trait
//! (compile once, match many) mirrors the tool's existing shape: a query
//! gets compiled a single time per `fast_search` call and the compiled
//! matcher is then shared across every spawned per-file task — collapsing
//! this into one trait method would force recompiling per file, a real
//! performance regression for large trees.

use std::collections::HashMap;
use std::sync::Arc;

use regex::Regex;

/// A pluggable content-matching strategy, registered under `name()` and
/// selected via `fast_search`'s `backend` parameter.
pub trait SearchBackend: Send + Sync {
    fn name(&self) -> &'static str;
    /// Compile/validate `query` once per `fast_search` call.
    fn compile(&self, query: &str) -> Result<Arc<dyn CompiledQuery>, String>;
}

/// A compiled, reusable matcher — shared (via `Arc`) across every spawned
/// per-file task for one `fast_search` call.
pub trait CompiledQuery: Send + Sync {
    /// Scan file content, returning `{"line": N, "text": "..."}` matches —
    /// `fast_search`'s existing per-file match shape.
    fn find_matches(&self, content: &str) -> Vec<serde_json::Value>;
}

/// The original regex-based backend, wrapped behind the trait. A pure move
/// of `FastSearchTool::execute`'s former `Regex::new(...)` call and the
/// per-file `content.lines()` matching loop — not a behavior change.
pub struct RegexSearchBackend;

impl SearchBackend for RegexSearchBackend {
    fn name(&self) -> &'static str {
        "regex"
    }

    fn compile(&self, query: &str) -> Result<Arc<dyn CompiledQuery>, String> {
        let re = Regex::new(query).map_err(|e| format!("Invalid Regex Compilation Error: {}", e))?;
        Ok(Arc::new(CompiledRegex(re)))
    }
}

struct CompiledRegex(Regex);

impl CompiledQuery for CompiledRegex {
    fn find_matches(&self, content: &str) -> Vec<serde_json::Value> {
        let mut line_matches = Vec::new();
        for (idx, line) in content.lines().enumerate() {
            if self.0.is_match(line) {
                line_matches.push(serde_json::json!({
                    "line": idx + 1,
                    "text": line.trim()
                }));
            }
        }
        line_matches
    }
}

/// Maps a backend name to the `SearchBackend` registered for it. Registering
/// a new backend under a name that's already taken overwrites it —
/// last-registered-wins, matching `ExtractorRegistry`'s policy.
pub struct SearchBackendRegistry {
    by_name: HashMap<String, Arc<dyn SearchBackend>>,
}

impl SearchBackendRegistry {
    pub fn new() -> Self {
        Self { by_name: HashMap::new() }
    }

    pub fn register(&mut self, backend: Arc<dyn SearchBackend>) {
        self.by_name.insert(backend.name().to_string(), backend);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn SearchBackend>> {
        self.by_name.get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_backend_matches_prior_inline_behavior() {
        let backend = RegexSearchBackend;
        let compiled = backend.compile("pub fn").unwrap();
        let content = "pub fn foo() {}\nfn bar() {}\n  pub fn baz() {}\n";

        let matches = compiled.find_matches(content);
        assert_eq!(
            matches,
            vec![
                serde_json::json!({"line": 1, "text": "pub fn foo() {}"}),
                serde_json::json!({"line": 3, "text": "pub fn baz() {}"}),
            ]
        );
    }

    #[test]
    fn regex_backend_surfaces_the_original_error_message_on_bad_pattern() {
        // Not unwrap_err(): the Ok type (Arc<dyn CompiledQuery>) doesn't
        // implement Debug, which unwrap_err() requires for its panic message.
        let backend = RegexSearchBackend;
        match backend.compile("(unclosed") {
            Err(err) => assert!(err.starts_with("Invalid Regex Compilation Error:"), "unexpected error: {err}"),
            Ok(_) => panic!("expected compile to fail on an invalid regex pattern"),
        }
    }

    #[test]
    fn registry_returns_registered_backend_and_none_for_unknown_name() {
        let mut registry = SearchBackendRegistry::new();
        registry.register(Arc::new(RegexSearchBackend));

        assert!(registry.get("regex").is_some());
        assert!(registry.get("fuzzy").is_none());
    }

    struct StubBackend;
    struct StubQuery;
    impl CompiledQuery for StubQuery {
        fn find_matches(&self, _content: &str) -> Vec<serde_json::Value> {
            vec![serde_json::json!({"line": 1, "text": "stub"})]
        }
    }
    impl SearchBackend for StubBackend {
        fn name(&self) -> &'static str {
            "regex"
        }
        fn compile(&self, _query: &str) -> Result<Arc<dyn CompiledQuery>, String> {
            Ok(Arc::new(StubQuery))
        }
    }

    #[test]
    fn registering_a_new_backend_under_the_same_name_overwrites_the_old_one() {
        let mut registry = SearchBackendRegistry::new();
        registry.register(Arc::new(RegexSearchBackend));
        registry.register(Arc::new(StubBackend));

        let compiled = registry.get("regex").unwrap().compile("anything").unwrap();
        assert_eq!(compiled.find_matches("irrelevant"), vec![serde_json::json!({"line": 1, "text": "stub"})]);
    }
}
