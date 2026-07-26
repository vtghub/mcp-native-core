//! `semantic_search`: finds code conceptually related to a natural-language
//! query even when the exact keywords don't match, complementing
//! `fast_search`'s exact/regex matching. Each file is split into fixed-size
//! line chunks, embedded locally (see `embeddings.rs`), and ranked by cosine
//! similarity against the query's embedding — compressed through the
//! TurboQuant-inspired quantizer in `quantizer.rs` so the resident index
//! stays a fraction of what raw f32 embeddings for a whole repo would cost.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::cache::{ContentCache, DirCache, EmbeddedChunk, VectorCache};
use crate::embeddings::{load_default_embedder, Embedder};
use crate::quantizer::{score, QuantizedVector, Rotation};
use crate::watcher::RepoWatcher;
use crate::McpTool;

// Lines per chunk. Small enough that a match points the caller at a focused
// snippet rather than a whole file, while comfortably fitting embeddings.rs's
// MAX_TOKENS budget for typical source-code line lengths.
const CHUNK_LINES: usize = 40;
const DEFAULT_TOP_K: usize = 10;

/// Splits `content` into non-overlapping `[start_line, end_line]` (1-based,
/// inclusive) windows of up to `chunk_size` lines each, skipping windows
/// that are entirely whitespace. A pure function so chunk boundaries can be
/// tested without a filesystem, a model, or a cache.
fn chunk_lines(content: &str, chunk_size: usize) -> Vec<(usize, usize, String)> {
    let lines: Vec<&str> = content.lines().collect();
    lines
        .chunks(chunk_size)
        .enumerate()
        .filter_map(|(i, window)| {
            let text = window.join("\n");
            if text.trim().is_empty() {
                return None;
            }
            let start = i * chunk_size + 1;
            let end = start + window.len() - 1;
            Some((start, end, text))
        })
        .collect()
}

/// The embedder and its matching rotation, resolved together on first use:
/// the rotation's dimension has to equal whatever model actually loaded
/// (`MCP_EMBEDDING_MODEL_ID` can point at a model with a different hidden
/// size than the default), so it can't be built any earlier than the
/// embedder itself.
struct SemanticResources {
    embedder: Arc<dyn Embedder>,
    rotation: Rotation,
}

pub struct SemanticSearchTool {
    dir_cache: Arc<DirCache>,
    content_cache: Arc<ContentCache>,
    vector_cache: Arc<VectorCache>,
    watcher: Arc<RepoWatcher>,
    resources: tokio::sync::OnceCell<Result<Arc<SemanticResources>, String>>,
}

impl SemanticSearchTool {
    pub fn new(
        dir_cache: Arc<DirCache>,
        content_cache: Arc<ContentCache>,
        vector_cache: Arc<VectorCache>,
        watcher: Arc<RepoWatcher>,
    ) -> Self {
        Self { dir_cache, content_cache, vector_cache, watcher, resources: tokio::sync::OnceCell::new() }
    }

    /// Lazily loads the embedding model (and builds its matching rotation)
    /// on first use rather than blocking server startup on a model
    /// download/load that may not even be needed this run, and memoizes the
    /// result — including a load failure — so a broken or offline setup
    /// fails fast on every subsequent call instead of re-attempting an
    /// expensive load each time.
    async fn resources(&self) -> Result<Arc<SemanticResources>, String> {
        self.resources
            .get_or_init(|| async {
                let embedder = load_default_embedder().await?;
                let rotation = Rotation::new(embedder.dim());
                Ok(Arc::new(SemanticResources { embedder, rotation }))
            })
            .await
            .clone()
    }

    /// Best-effort snippet text for one result, re-sliced from `ContentCache`
    /// by line range. Only called for the final top-k results, not every
    /// indexed chunk, so the vector index itself never needs to hold raw
    /// text — just compact quantized vectors and line ranges.
    fn snippet_for(&self, path: &Path, start_line: usize, end_line: usize) -> Option<String> {
        let metadata = std::fs::metadata(path).ok()?;
        let mtime = metadata.modified().ok()?;
        let len = metadata.len();
        let content = self.content_cache.get_if_fresh(path, mtime, len)?;
        let lines: Vec<&str> = content.lines().collect();
        let slice = lines.get(start_line.saturating_sub(1)..end_line.min(lines.len()))?;
        Some(slice.join("\n"))
    }
}

#[async_trait::async_trait]
impl McpTool for SemanticSearchTool {
    fn name(&self) -> &'static str {
        "semantic_search"
    }

    fn description(&self) -> &'static str {
        "Finds code conceptually related to a natural-language query using local sentence embeddings, even when exact keywords don't match. Complements fast_search's exact/regex matching; slower per-call (embeds every indexed file once) but far better at 'find the code that handles X' when you don't know the exact identifiers."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "root_dir": { "type": "string", "description": "The target codebase absolute directory path" },
                "query": { "type": "string", "description": "Natural-language description of the code you're looking for" },
                "top_k": { "type": "integer", "description": "Maximum number of results to return (default 10)" },
                "extensions": { "type": "array", "items": { "type": "string" }, "description": "Optional filters, e.g. ['rs', 'cs', 'py']" }
            },
            "required": ["root_dir", "query"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let root_str = params.get("root_dir").and_then(|v| v.as_str()).ok_or("Missing root_dir")?;
        let query = params.get("query").and_then(|v| v.as_str()).ok_or("Missing query")?;
        let top_k = params.get("top_k").and_then(|v| v.as_u64()).map(|n| n as usize).unwrap_or(DEFAULT_TOP_K);

        let mut extensions = Vec::new();
        if let Some(arr) = params.get("extensions").and_then(|v| v.as_array()) {
            for ext in arr {
                if let Some(s) = ext.as_str() {
                    extensions.push(s.to_string());
                }
            }
        }

        let resources = self.resources().await?;
        let embedder = Arc::clone(&resources.embedder);

        let root_path = PathBuf::from(root_str);
        self.watcher.ensure_watching(&root_path);

        let mut target_files = Vec::new();
        self.dir_cache.crawl(&root_path, &extensions, &mut target_files);

        let query_rotated = {
            let embedder = Arc::clone(&embedder);
            let query_owned = query.to_string();
            let embedding = tokio::task::spawn_blocking(move || embedder.embed(&query_owned))
                .await
                .map_err(|e| format!("Query embedding task panicked: {e}"))??;
            resources.rotation.apply(&embedding)
        };

        let chunks_indexed = Arc::new(AtomicUsize::new(0));
        let chunks_reused = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for file_path in target_files {
            let content_cache = Arc::clone(&self.content_cache);
            let vector_cache = Arc::clone(&self.vector_cache);
            let embedder = Arc::clone(&embedder);
            let resources = Arc::clone(&resources);
            let chunks_indexed = Arc::clone(&chunks_indexed);
            let chunks_reused = Arc::clone(&chunks_reused);

            let task = tokio::task::spawn_blocking(move || -> Option<(PathBuf, Arc<Vec<EmbeddedChunk>>)> {
                let metadata = std::fs::metadata(&file_path).ok()?;
                let mtime = metadata.modified().ok()?;
                let len = metadata.len();

                if let Some(cached) = vector_cache.get_if_fresh(&file_path, mtime, len) {
                    chunks_reused.fetch_add(cached.len(), Ordering::Relaxed);
                    return Some((file_path, cached));
                }

                // Same stat-first / mmap-on-miss content lookup fast_search
                // uses, sharing the same `ContentCache` so indexing a repo
                // for semantic_search doesn't cost a second read of every
                // file fast_search already warmed.
                let content: Arc<str> = if let Some(cached) = content_cache.get_if_fresh(&file_path, mtime, len) {
                    cached
                } else {
                    let file = std::fs::File::open(&file_path).ok()?;
                    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
                    let text = std::str::from_utf8(&mmap).ok()?;
                    let owned: Arc<str> = Arc::from(text);
                    content_cache.store(file_path.clone(), mtime, len, owned.clone());
                    owned
                };

                // A single chunk failing to embed (a model error on one
                // odd span of text) should only drop that chunk, not the
                // rest of an otherwise-good file's index.
                let mut chunks = Vec::new();
                for (start_line, end_line, text) in chunk_lines(&content, CHUNK_LINES) {
                    let Ok(embedding) = embedder.embed(&text) else { continue };
                    let vector = QuantizedVector::encode(&embedding, &resources.rotation);
                    chunks.push(EmbeddedChunk { start_line, end_line, vector });
                }

                chunks_indexed.fetch_add(chunks.len(), Ordering::Relaxed);
                let chunks = Arc::new(chunks);
                vector_cache.store(file_path.clone(), mtime, len, chunks.clone());
                Some((file_path, chunks))
            });
            tasks.push(task);
        }

        let mut scored: Vec<(f32, PathBuf, usize, usize)> = Vec::new();
        for task in tasks {
            if let Ok(Some((file_path, chunks))) = task.await {
                for chunk in chunks.iter() {
                    let similarity = score(&query_rotated, &chunk.vector);
                    scored.push((similarity, file_path.clone(), chunk.start_line, chunk.end_line));
                }
            }
        }

        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let results: Vec<serde_json::Value> = scored
            .into_iter()
            .map(|(similarity, file_path, start_line, end_line)| {
                let snippet = self.snippet_for(&file_path, start_line, end_line);
                serde_json::json!({
                    "file": file_path.to_string_lossy(),
                    "start_line": start_line,
                    "end_line": end_line,
                    "similarity": similarity,
                    "snippet": snippet,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "status": "success",
            "query": query,
            "chunks_indexed": chunks_indexed.load(Ordering::Relaxed),
            "chunks_reused_from_cache": chunks_reused.load(Ordering::Relaxed),
            "results": results,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_lines_splits_into_fixed_size_non_overlapping_windows() {
        let content = (1..=95).map(|n| format!("line {n}")).collect::<Vec<_>>().join("\n");
        let chunks = chunk_lines(&content, 40);

        assert_eq!(chunks.len(), 3);
        assert_eq!((chunks[0].0, chunks[0].1), (1, 40));
        assert_eq!((chunks[1].0, chunks[1].1), (41, 80));
        assert_eq!((chunks[2].0, chunks[2].1), (81, 95));
        assert!(chunks[0].2.starts_with("line 1\n"));
        assert!(chunks[2].2.ends_with("line 95"));
    }

    #[test]
    fn chunk_lines_skips_whitespace_only_windows() {
        let content = "\n\n   \n\t\n";
        let chunks = chunk_lines(content, 40);
        assert!(chunks.is_empty(), "an all-whitespace file shouldn't produce any chunks to embed");
    }

    #[test]
    fn chunk_lines_handles_empty_content() {
        assert!(chunk_lines("", 40).is_empty());
    }

    #[test]
    fn chunk_lines_handles_content_shorter_than_one_chunk() {
        let chunks = chunk_lines("fn a() {}\nfn b() {}", 40);
        assert_eq!(chunks, vec![(1, 2, "fn a() {}\nfn b() {}".to_string())]);
    }
}
