use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

const SKIPPED_DIR_NAMES: [&str; 4] = [".git", "node_modules", "target", "bin"];

#[derive(Clone)]
struct DirCacheEntry {
    mtime: SystemTime,
    files: Arc<Vec<PathBuf>>,
    subdirs: Arc<Vec<PathBuf>>,
}

#[derive(Default)]
pub struct CrawlStats {
    pub dirs_visited: usize,
    pub dirs_rescanned: usize,
}

/// Caches directory listings keyed by absolute path. A directory's mtime changes
/// whenever an entry is added, removed, or renamed inside it, so comparing mtimes
/// lets repeat crawls skip re-listing (and re-recursing past) any subtree that
/// hasn't changed since it was last scanned.
pub struct DirCache {
    entries: DashMap<PathBuf, DirCacheEntry>,
}

impl DirCache {
    pub fn new() -> Self {
        Self { entries: DashMap::new() }
    }

    pub fn crawl(&self, dir: &Path, extensions: &[String], out: &mut Vec<PathBuf>) -> CrawlStats {
        let mut stats = CrawlStats::default();
        self.crawl_inner(dir, extensions, out, &mut stats);
        stats
    }

    fn crawl_inner(
        &self,
        dir: &Path,
        extensions: &[String],
        out: &mut Vec<PathBuf>,
        stats: &mut CrawlStats,
    ) {
        let mtime = match std::fs::metadata(dir).and_then(|m| m.modified()) {
            Ok(m) => m,
            Err(_) => return,
        };
        stats.dirs_visited += 1;

        // The DashMap `Ref` guard from `.get()` holds a read lock on its shard.
        // It must be dropped before `rescan()` can `insert()` into that same
        // shard, so the freshness check is resolved to a plain value first
        // (dropping the guard at the end of this statement) rather than being
        // matched on directly, which would keep the guard alive into the
        // `rescan()` call and deadlock on the shard's lock.
        let fresh = self.entries.get(dir).and_then(|cached| {
            if cached.mtime == mtime { Some(cached.clone()) } else { None }
        });

        let entry = match fresh {
            Some(cached) => cached,
            None => {
                stats.dirs_rescanned += 1;
                self.rescan(dir, mtime)
            }
        };

        for file in entry.files.iter() {
            if let Some(ext) = file.extension().and_then(|e| e.to_str()) {
                if extensions.is_empty() || extensions.iter().any(|e| e == ext) {
                    out.push(file.clone());
                }
            }
        }

        for subdir in entry.subdirs.iter() {
            self.crawl_inner(subdir, extensions, out, stats);
        }
    }

    fn rescan(&self, dir: &Path, mtime: SystemTime) -> DirCacheEntry {
        let mut files = Vec::new();
        let mut subdirs = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let skip = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| SKIPPED_DIR_NAMES.contains(&n))
                        .unwrap_or(false);
                    if !skip {
                        subdirs.push(path);
                    }
                } else {
                    files.push(path);
                }
            }
        }

        let entry = DirCacheEntry {
            mtime,
            files: Arc::new(files),
            subdirs: Arc::new(subdirs),
        };
        self.entries.insert(dir.to_path_buf(), entry.clone());
        entry
    }

    /// Evict a directory's cached listing, e.g. in response to a filesystem
    /// watch event. The next crawl through `dir` will re-list it instead of
    /// trusting a listing that may now be stale. Safe to call for a path with
    /// no entry (removing a non-existent key is a no-op).
    pub fn invalidate(&self, dir: &Path) {
        self.entries.remove(dir);
    }
}

struct CachedFileEntry {
    mtime: SystemTime,
    len: u64,
    structural_nodes: Arc<Vec<serde_json::Value>>,
}

/// Caches parsed structural nodes per file, keyed by absolute path. A cached
/// result is only served when the file's current (mtime, len) still match what
/// was recorded when it was parsed; any change invalidates the entry.
pub struct FileCache {
    entries: DashMap<PathBuf, CachedFileEntry>,
}

impl FileCache {
    pub fn new() -> Self {
        Self { entries: DashMap::new() }
    }

    pub fn get_if_fresh(
        &self,
        path: &Path,
        mtime: SystemTime,
        len: u64,
    ) -> Option<Arc<Vec<serde_json::Value>>> {
        self.entries.get(path).and_then(|entry| {
            if entry.mtime == mtime && entry.len == len {
                Some(entry.structural_nodes.clone())
            } else {
                None
            }
        })
    }

    pub fn store(&self, path: PathBuf, mtime: SystemTime, len: u64, nodes: Arc<Vec<serde_json::Value>>) {
        self.entries.insert(
            path,
            CachedFileEntry { mtime, len, structural_nodes: nodes },
        );
    }

    /// Evict a file's cached structural nodes, e.g. in response to a
    /// filesystem watch event. Safe to call for a path with no entry.
    pub fn invalidate(&self, path: &Path) {
        self.entries.remove(path);
    }
}

// 256 MiB by default; override with MCP_CONTENT_CACHE_MAX_BYTES (parsed once
// per ContentCache::new() call, so it can be set per-process).
const DEFAULT_CONTENT_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024;
// When a store() pushes the cache over budget, evict down to this fraction
// of the cap rather than just enough to cross back under it — otherwise an
// insert sitting right at the boundary would re-trigger a full eviction scan
// on nearly every subsequent store().
const EVICT_TARGET_RATIO: f64 = 0.9;

fn env_max_bytes() -> usize {
    std::env::var("MCP_CONTENT_CACHE_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_CONTENT_CACHE_MAX_BYTES)
}

struct CachedContentEntry {
    mtime: SystemTime,
    len: u64,
    content: Arc<str>,
    // Monotonic "clock" tick set on insert and bumped on every hit; used to
    // approximate least-recently-used ordering without needing a real
    // linked-list LRU (which would need a lock shared across every access,
    // undermining DashMap's per-shard concurrency). A plain atomic field
    // updatable through DashMap's shared read-guard is enough for this.
    last_used: AtomicU64,
}

/// Caches whole-file text content, keyed by absolute path, so a repeat
/// `fast_search` query over an unchanged file skips re-opening, re-mmapping,
/// and re-decoding it — only the regex scan itself needs to rerun, since the
/// query differs per call. Freshness follows the same (mtime, len) contract
/// as `FileCache`.
///
/// Bounded by a total byte budget (`max_bytes`, default 256 MiB, override via
/// `MCP_CONTENT_CACHE_MAX_BYTES`): once `store()` pushes the running total
/// over budget, the least-recently-used entries are evicted until back under
/// `EVICT_TARGET_RATIO` of the cap. Eviction is approximate, not a strict
/// global LRU: a snapshot-sort-evict pass runs concurrently with other
/// threads' stores/evictions without a shared lock, so the byte total it
/// evicts against can drift slightly during the pass. That's an acceptable
/// trade-off for a cache this size (never unsafe, just approximate — the
/// same trade-off production concurrent caches like Caffeine/moka make).
pub struct ContentCache {
    entries: DashMap<PathBuf, CachedContentEntry>,
    total_bytes: AtomicUsize,
    max_bytes: usize,
    clock: AtomicU64,
}

impl ContentCache {
    pub fn new() -> Self {
        Self::with_max_bytes(env_max_bytes())
    }

    /// Construct with an explicit byte budget instead of the environment
    /// default — mainly for tests that need eviction to trigger
    /// deterministically without allocating hundreds of megabytes.
    pub fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            entries: DashMap::new(),
            total_bytes: AtomicUsize::new(0),
            max_bytes,
            clock: AtomicU64::new(0),
        }
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    pub fn get_if_fresh(&self, path: &Path, mtime: SystemTime, len: u64) -> Option<Arc<str>> {
        let tick = self.tick();
        self.entries.get(path).and_then(|entry| {
            if entry.mtime == mtime && entry.len == len {
                // Updating through the shared read-guard is safe: last_used
                // is the only mutated field, and it's atomic.
                entry.last_used.store(tick, Ordering::Relaxed);
                Some(entry.content.clone())
            } else {
                None
            }
        })
    }

    pub fn store(&self, path: PathBuf, mtime: SystemTime, len: u64, content: Arc<str>) {
        let new_size = content.len();
        let entry = CachedContentEntry { mtime, len, content, last_used: AtomicU64::new(self.tick()) };
        let old = self.entries.insert(path, entry);
        let old_size = old.map(|e| e.content.len()).unwrap_or(0);
        if new_size >= old_size {
            self.total_bytes.fetch_add(new_size - old_size, Ordering::Relaxed);
        } else {
            self.total_bytes.fetch_sub(old_size - new_size, Ordering::Relaxed);
        }

        if self.total_bytes.load(Ordering::Relaxed) > self.max_bytes {
            self.evict_to_target();
        }
    }

    /// Evict least-recently-used entries until the cache is back under
    /// `EVICT_TARGET_RATIO * max_bytes`. Snapshots (path, last_used, size)
    /// for every entry first — dropping all DashMap iterator guards — before
    /// removing anything, so this can never hold a shard guard across a
    /// `remove()` call on that same shard (the class of self-deadlock fixed
    /// in DirCache during phase 1).
    fn evict_to_target(&self) {
        let target = (self.max_bytes as f64 * EVICT_TARGET_RATIO) as usize;

        let mut snapshot: Vec<(PathBuf, u64, usize)> = self
            .entries
            .iter()
            .map(|e| (e.key().clone(), e.value().last_used.load(Ordering::Relaxed), e.value().content.len()))
            .collect();
        snapshot.sort_unstable_by_key(|&(_, last_used, _)| last_used);

        let mut current = self.total_bytes.load(Ordering::Relaxed);
        let mut freed = 0usize;
        for (path, _, size) in snapshot {
            if current <= target {
                break;
            }
            if self.entries.remove(&path).is_some() {
                current = current.saturating_sub(size);
                freed += size;
            }
        }
        if freed > 0 {
            self.total_bytes.fetch_sub(freed, Ordering::Relaxed);
        }
    }

    /// Evict a file's cached content, e.g. in response to a filesystem watch
    /// event. Safe to call for a path with no entry.
    pub fn invalidate(&self, path: &Path) {
        if let Some((_, entry)) = self.entries.remove(path) {
            self.total_bytes.fetch_sub(entry.content.len(), Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::thread::sleep;
    use std::time::Duration;

    // Some filesystems have coarse mtime resolution; sleep past it so a
    // deliberate modification is guaranteed to bump the recorded mtime.
    fn settle() {
        sleep(Duration::from_millis(1100));
    }

    /// A directory under the OS temp dir that removes itself on drop, so tests
    /// don't need an external tempdir crate.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mcp-native-core-test-{}-{}",
                std::process::id(),
                n
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn dir_cache_reuses_listing_when_untouched() {
        let dir = TempDir::new();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();

        let cache = DirCache::new();
        let extensions = vec!["rs".to_string()];

        let mut out = Vec::new();
        let first = cache.crawl(dir.path(), &extensions, &mut out);
        assert_eq!(first.dirs_rescanned, 1);
        assert_eq!(out.len(), 1);

        let mut out2 = Vec::new();
        let second = cache.crawl(dir.path(), &extensions, &mut out2);
        assert_eq!(second.dirs_rescanned, 0, "unchanged directory should not be rescanned");
        assert_eq!(out2.len(), 1);
    }

    #[test]
    fn dir_cache_rescans_after_new_file_added() {
        let dir = TempDir::new();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();

        let cache = DirCache::new();
        let extensions = vec!["rs".to_string()];

        let mut out = Vec::new();
        cache.crawl(dir.path(), &extensions, &mut out);

        settle();
        fs::write(dir.path().join("b.rs"), "fn b() {}").unwrap();

        let mut out2 = Vec::new();
        let stats = cache.crawl(dir.path(), &extensions, &mut out2);
        assert_eq!(stats.dirs_rescanned, 1, "directory mtime changed, should rescan");
        assert_eq!(out2.len(), 2);
    }

    #[test]
    fn dir_cache_skips_ignored_directories() {
        let dir = TempDir::new();
        fs::create_dir(dir.path().join(".git")).unwrap();
        fs::write(dir.path().join(".git").join("config"), "x").unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();

        let cache = DirCache::new();
        let mut out = Vec::new();
        let stats = cache.crawl(dir.path(), &[], &mut out);
        assert_eq!(stats.dirs_visited, 1, ".git should not be recursed into");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn file_cache_serves_fresh_entry_and_rejects_stale() {
        let dir = TempDir::new();
        let file_path = dir.path().join("a.rs");
        fs::write(&file_path, "fn a() {}").unwrap();

        let meta = fs::metadata(&file_path).unwrap();
        let mtime = meta.modified().unwrap();
        let len = meta.len();

        let cache = FileCache::new();
        assert!(cache.get_if_fresh(&file_path, mtime, len).is_none());

        let nodes = Arc::new(vec![serde_json::json!({"line": 1, "declaration": "fn a() {}"})]);
        cache.store(file_path.clone(), mtime, len, nodes.clone());

        let hit = cache.get_if_fresh(&file_path, mtime, len);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().len(), 1);

        settle();
        fs::write(&file_path, "fn a() {}\nfn b() {}").unwrap();
        let new_meta = fs::metadata(&file_path).unwrap();

        let stale = cache.get_if_fresh(&file_path, new_meta.modified().unwrap(), new_meta.len());
        assert!(stale.is_none(), "changed file should invalidate cache");
    }

    #[test]
    fn content_cache_serves_fresh_entry_and_rejects_stale() {
        let dir = TempDir::new();
        let file_path = dir.path().join("a.rs");
        fs::write(&file_path, "fn a() {}").unwrap();

        let meta = fs::metadata(&file_path).unwrap();
        let mtime = meta.modified().unwrap();
        let len = meta.len();

        let cache = ContentCache::new();
        assert!(cache.get_if_fresh(&file_path, mtime, len).is_none());

        let content: Arc<str> = Arc::from("fn a() {}");
        cache.store(file_path.clone(), mtime, len, content.clone());

        let hit = cache.get_if_fresh(&file_path, mtime, len);
        assert!(hit.is_some());
        assert_eq!(&*hit.unwrap(), "fn a() {}");

        settle();
        fs::write(&file_path, "fn a() {}\nfn b() {}").unwrap();
        let new_meta = fs::metadata(&file_path).unwrap();

        let stale = cache.get_if_fresh(&file_path, new_meta.modified().unwrap(), new_meta.len());
        assert!(stale.is_none(), "changed file should invalidate cache");
    }

    #[test]
    fn content_cache_invalidate_removes_entry_regardless_of_freshness_check() {
        let dir = TempDir::new();
        let file_path = dir.path().join("a.rs");
        fs::write(&file_path, "fn a() {}").unwrap();
        let meta = fs::metadata(&file_path).unwrap();
        let (mtime, len) = (meta.modified().unwrap(), meta.len());

        let cache = ContentCache::new();
        cache.store(file_path.clone(), mtime, len, Arc::from("fn a() {}"));
        assert!(cache.get_if_fresh(&file_path, mtime, len).is_some());

        cache.invalidate(&file_path);
        assert!(cache.get_if_fresh(&file_path, mtime, len).is_none());
    }

    #[test]
    fn content_cache_evicts_least_recently_used_when_over_budget() {
        // 250-byte budget, evicting down to 90% (225) on overflow. Three
        // 100-byte entries: after touching "a" via get_if_fresh, "b" becomes
        // the least-recently-used entry, so storing "c" (pushing the total to
        // 300, over budget) should evict exactly "b".
        let cache = ContentCache::with_max_bytes(250);
        let mtime = SystemTime::now();
        let content = |n: usize| -> Arc<str> { Arc::from("x".repeat(n).as_str()) };

        let path_a = PathBuf::from("/fake/a.rs");
        let path_b = PathBuf::from("/fake/b.rs");
        let path_c = PathBuf::from("/fake/c.rs");

        cache.store(path_a.clone(), mtime, 100, content(100));
        cache.store(path_b.clone(), mtime, 100, content(100));
        assert!(cache.get_if_fresh(&path_a, mtime, 100).is_some(), "bump a's recency");

        cache.store(path_c.clone(), mtime, 100, content(100));

        assert!(cache.get_if_fresh(&path_b, mtime, 100).is_none(), "b was least-recently-used and should be evicted");
        assert!(cache.get_if_fresh(&path_a, mtime, 100).is_some(), "recently-touched a should survive");
        assert!(cache.get_if_fresh(&path_c, mtime, 100).is_some(), "just-inserted c should survive");
    }

    #[test]
    fn content_cache_replacing_an_entry_adjusts_total_bytes_correctly() {
        // Regression guard for the old_size/new_size delta accounting in
        // store(): replacing a 500-byte entry with a 10-byte one must shrink
        // total_bytes by 490, not just add 10 on top of the stale 500.
        //
        // Budget is tuned so the two behaviors diverge observably: true
        // total after replace+one more 480-byte store is 490 (under the
        // 520 cap — no eviction, both entries survive). A version that
        // forgets to subtract the old size would compute 990 (over budget),
        // triggering eviction that empties the cache entirely — caught by
        // the assertions below.
        let cache = ContentCache::with_max_bytes(520);
        let mtime = SystemTime::now();
        let path_a = PathBuf::from("/fake/a.rs");
        let path_b = PathBuf::from("/fake/b.rs");

        cache.store(path_a.clone(), mtime, 500, Arc::from("x".repeat(500).as_str()));
        cache.store(path_a.clone(), mtime, 10, Arc::from("y".repeat(10).as_str()));
        cache.store(path_b.clone(), mtime, 480, Arc::from("z".repeat(480).as_str()));

        assert!(cache.get_if_fresh(&path_a, mtime, 10).is_some(), "replaced entry should survive under correct accounting");
        assert!(cache.get_if_fresh(&path_b, mtime, 480).is_some(), "second entry should survive under correct accounting");
    }
}
