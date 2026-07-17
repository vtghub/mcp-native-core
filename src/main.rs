use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs::File;
use tokio::io::{stdin, stdout, BufReader, BufWriter, AsyncBufReadExt, AsyncWriteExt};
use serde::{Deserialize, Serialize};

mod cache;
mod extractors;
mod search_backend;
mod watcher;
use cache::{ContentCache, DirCache, FileCache};
use extractors::{ExtractorRegistry, RegexExtractor};
use search_backend::{RegexSearchBackend, SearchBackendRegistry};
use watcher::RepoWatcher;

#[derive(Deserialize, Serialize, Debug)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
    #[serde(default)]
    id: Option<serde_json::Value>, // Made optional because MCP sends notifications without IDs
}

#[derive(Serialize, Debug)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<serde_json::Value>,
    id: Option<serde_json::Value>,
}

#[async_trait::async_trait]
pub trait McpTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn input_schema(&self) -> serde_json::Value;
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value, String>;
}

pub struct McpServerState {
    pub tools: HashMap<String, Arc<dyn McpTool>>,
}

impl McpServerState {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }
    pub fn register_tool(&mut self, tool: Box<dyn McpTool>) {
        self.tools.insert(tool.name().to_string(), Arc::from(tool));
    }
}

pub struct FastSearchTool {
    dir_cache: Arc<DirCache>,
    content_cache: Arc<ContentCache>,
    watcher: Arc<RepoWatcher>,
    search_registry: Arc<SearchBackendRegistry>,
}
impl FastSearchTool {
    pub fn new(
        dir_cache: Arc<DirCache>,
        content_cache: Arc<ContentCache>,
        watcher: Arc<RepoWatcher>,
        search_registry: Arc<SearchBackendRegistry>,
    ) -> Self {
        Self { dir_cache, content_cache, watcher, search_registry }
    }
}

#[async_trait::async_trait]
impl McpTool for FastSearchTool {
    fn name(&self) -> &'static str { "fast_search" }
    
    fn description(&self) -> &'static str {
        "Scans workspace files using zero-copy memory mapping and multi-threaded regex compilation."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "root_dir": { "type": "string", "description": "The target codebase absolute directory path" },
                "query": { "type": "string", "description": "Regex matching pattern string" },
                "extensions": { "type": "array", "items": { "type": "string" }, "description": "Optional filters, e.g. ['rs', 'cs', 'py']" },
                "backend": { "type": "string", "description": "Search backend to use, e.g. 'regex' (default)" }
            },
            "required": ["root_dir", "query"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let root_str = params.get("root_dir").and_then(|v| v.as_str()).ok_or("Missing root_dir")?;
        let query_str = params.get("query").and_then(|v| v.as_str()).ok_or("Missing query pattern")?;

        let mut extensions = Vec::new();
        if let Some(arr) = params.get("extensions").and_then(|v| v.as_array()) {
            for ext in arr {
                if let Some(s) = ext.as_str() { extensions.push(s.to_string()); }
            }
        }

        let backend_name = params.get("backend").and_then(|v| v.as_str()).unwrap_or("regex");
        let backend = self.search_registry.get(backend_name)
            .ok_or_else(|| format!("Unknown search backend: {}", backend_name))?;
        let compiled = backend.compile(query_str)?;
        let root_path = PathBuf::from(root_str);
        self.watcher.ensure_watching(&root_path);

        let mut target_files = Vec::new();
        let crawl_stats = self.dir_cache.crawl(&root_path, &extensions, &mut target_files);

        let start_time = std::time::Instant::now();
        let mut tasks = Vec::new();
        let content_hits = Arc::new(AtomicUsize::new(0));
        let content_misses = Arc::new(AtomicUsize::new(0));

        for file_path in target_files {
            let compiled = Arc::clone(&compiled);
            let content_cache = Arc::clone(&self.content_cache);
            let content_hits = Arc::clone(&content_hits);
            let content_misses = Arc::clone(&content_misses);

            let task = tokio::task::spawn_blocking(move || -> Option<serde_json::Value> {
                // Stat first (cheap) so an unchanged file can skip the
                // open+mmap+utf8-decode below entirely on a cache hit —
                // only the regex scan has to rerun, since the query differs
                // per call. Falls straight through to a full read on a stat
                // failure (e.g. a race with a delete) or a cache miss.
                let content: Arc<str> = 'content: {
                    if let Ok(metadata) = std::fs::metadata(&file_path) {
                        if let Ok(mtime) = metadata.modified() {
                            let len = metadata.len();
                            if let Some(cached) = content_cache.get_if_fresh(&file_path, mtime, len) {
                                content_hits.fetch_add(1, Ordering::Relaxed);
                                break 'content cached;
                            }
                            content_misses.fetch_add(1, Ordering::Relaxed);
                            let file = File::open(&file_path).ok()?;
                            let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
                            let text = std::str::from_utf8(&mmap).ok()?;
                            let owned: Arc<str> = Arc::from(text);
                            content_cache.store(file_path.clone(), mtime, len, owned.clone());
                            break 'content owned;
                        }
                    }
                    content_misses.fetch_add(1, Ordering::Relaxed);
                    let file = File::open(&file_path).ok()?;
                    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
                    Arc::from(std::str::from_utf8(&mmap).ok()?)
                };

                let line_matches = compiled.find_matches(&content);

                if !line_matches.is_empty() {
                    Some(serde_json::json!({
                        "file": file_path.to_string_lossy(),
                        "matches": line_matches
                    }))
                } else {
                    None
                }
            });
            tasks.push(task);
        }

        let mut final_results = Vec::new();
        for task in tasks {
            if let Ok(Some(file_match_block)) = task.await {
                final_results.push(file_match_block);
            }
        }

        Ok(serde_json::json!({
            "status": "success",
            "search_latency_ms": start_time.elapsed().as_millis(),
            "directories_scanned": crawl_stats.dirs_visited,
            "directories_rescanned": crawl_stats.dirs_rescanned,
            "content_cache_hits": content_hits.load(Ordering::Relaxed),
            "content_cache_misses": content_misses.load(Ordering::Relaxed),
            "matches": final_results
        }))
    }
}

pub struct ParseStructureTool {
    file_cache: Arc<FileCache>,
    watcher: Arc<RepoWatcher>,
    registry: Arc<ExtractorRegistry>,
}
impl ParseStructureTool {
    pub fn new(file_cache: Arc<FileCache>, watcher: Arc<RepoWatcher>, registry: Arc<ExtractorRegistry>) -> Self {
        Self { file_cache, watcher, registry }
    }
}

#[async_trait::async_trait]
impl McpTool for ParseStructureTool {
    fn name(&self) -> &'static str { "parse_structure" }

    fn description(&self) -> &'static str {
        "Extracts structural definitions (classes, functions, methods) from a file to optimize token usage."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "Absolute path to the target source code file" }
            },
            "required": ["file_path"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let path_str = params.get("file_path").and_then(|v| v.as_str()).ok_or("Missing file_path")?;
        let path = PathBuf::from(path_str);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        if let Some(parent) = path.parent() {
            self.watcher.ensure_watching(parent);
        }

        let metadata = std::fs::metadata(&path).map_err(|e| format!("Failed to stat file: {}", e))?;
        let mtime = metadata.modified().map_err(|e| format!("Failed to read mtime: {}", e))?;
        let len = metadata.len();

        if let Some(cached_nodes) = self.file_cache.get_if_fresh(&path, mtime, len) {
            return Ok(serde_json::json!({
                "status": "success",
                "file": path_str,
                "detected_language": ext,
                "structural_skeleton": &*cached_nodes,
                "cache_hit": true
            }));
        }

        let file = File::open(&path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| format!("Mmap failed: {}", e))? };
        let content = std::str::from_utf8(&mmap).map_err(|e| format!("Invalid UTF-8 sequence: {}", e))?;

        let structural_nodes = match self.registry.get(ext) {
            Some(extractor) => extractor.extract(ext, content),
            None => Vec::new(),
        };

        let nodes = Arc::new(structural_nodes);
        self.file_cache.store(path.clone(), mtime, len, nodes.clone());

        Ok(serde_json::json!({
            "status": "success",
            "file": path_str,
            "detected_language": ext,
            "structural_skeleton": &*nodes,
            "cache_hit": false
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut state = McpServerState::new();

    let dir_cache = Arc::new(DirCache::new());
    let file_cache = Arc::new(FileCache::new());
    let content_cache = Arc::new(ContentCache::new());
    let watcher = RepoWatcher::spawn(dir_cache.clone(), file_cache.clone(), content_cache.clone());

    let mut extractor_registry = ExtractorRegistry::new();
    extractor_registry.register(Arc::new(RegexExtractor));
    let extractor_registry = Arc::new(extractor_registry);

    let mut search_registry = SearchBackendRegistry::new();
    search_registry.register(Arc::new(RegexSearchBackend));
    let search_registry = Arc::new(search_registry);

    state.register_tool(Box::new(FastSearchTool::new(dir_cache, content_cache, watcher.clone(), search_registry)));
    state.register_tool(Box::new(ParseStructureTool::new(file_cache, watcher, extractor_registry)));

    let shared_state = Arc::new(state);

    let mut reader = BufReader::new(stdin());
    let mut line = String::new();

    // tokio::io::stdout() is not safe to write to concurrently from multiple tasks
    // (its blocking-thread wrapper panics with "JoinHandle polled after completion"
    // under concurrent access). Route all responses through a single dedicated
    // writer task instead, so stdout is only ever touched sequentially.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        let mut writer = BufWriter::new(stdout());
        while let Some(serialized) = rx.recv().await {
            let _ = writer.write_all(format!("{}\n", serialized).as_bytes()).await;
            let _ = writer.flush().await;
        }
    });

    while reader.read_line(&mut line).await? > 0 {
        let current_line = line.trim().to_string();
        line.clear();

        if current_line.is_empty() { continue; }

        let state_clone = Arc::clone(&shared_state);
        let tx_clone = tx.clone();

        tokio::spawn(async move {
            if let Ok(req) = serde_json::from_str::<JsonRpcRequest>(&current_line) {
                // MCP sends background notifications (like "notifications/initialized").
                // We should safely ignore requests with no ID to prevent crashing.
                if req.id.is_none() { return; }

                let response = match req.method.as_str() {
                    "initialize" => handle_initialize(req.id),
                    "tools/list" => handle_list_tools(&state_clone, req.id),
                    "tools/call" => handle_call_tool(&state_clone, req.params, req.id).await,
                    _ => handle_unknown_method(req.id),
                };

                if let Ok(serialized) = serde_json::to_string(&response) {
                    let _ = tx_clone.send(serialized);
                }
            }
        });
    }
    Ok(())
}

fn handle_initialize(id: Option<serde_json::Value>) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        result: Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "mcp-native-core",
                "version": "0.1.0"
            }
        })),
        error: None,
        id,
    }
}

fn handle_list_tools(state: &McpServerState, id: Option<serde_json::Value>) -> JsonRpcResponse {
    let tools_list: Vec<serde_json::Value> = state.tools.values().map(|tool| {
        serde_json::json!({
            "name": tool.name(),
            "description": tool.description(),
            "inputSchema": tool.input_schema()
        })
    }).collect();

    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        result: Some(serde_json::json!({ "tools": tools_list })),
        error: None,
        id,
    }
}

async fn handle_call_tool(state: &McpServerState, params: serde_json::Value, id: Option<serde_json::Value>) -> JsonRpcResponse {
    let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(tool) = state.tools.get(tool_name) {
        let args = params.get("arguments").unwrap_or(&serde_json::Value::Null).clone();
        match tool.execute(args).await {
            Ok(res) => JsonRpcResponse { 
                jsonrpc: "2.0".to_string(), 
                // MCP strictly requires successful tool results to be formatted as an array of 'content' blocks
                result: Some(serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string_pretty(&res).unwrap_or_default()
                    }]
                })), 
                error: None, 
                id 
            },
            Err(e) => JsonRpcResponse { 
                jsonrpc: "2.0".to_string(), 
                result: None, 
                error: Some(serde_json::json!({ "code": -32603, "message": e })), 
                id 
            },
        }
    } else {
        JsonRpcResponse { 
            jsonrpc: "2.0".to_string(), 
            result: None, 
            error: Some(serde_json::json!({ "code": -32601, "message": "Tool not found" })), 
            id 
        }
    }
}

fn handle_unknown_method(id: Option<serde_json::Value>) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        result: None,
        error: Some(serde_json::json!({ "code": -32601, "message": "Method Not Found" })),
        id,
    }
}