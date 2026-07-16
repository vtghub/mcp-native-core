//! Push-based cache invalidation: watch pointed-at repos for filesystem
//! changes and evict stale `DirCache`/`FileCache` entries proactively,
//! instead of only catching staleness the next time a tool call happens to
//! stat that exact path.
//!
//! This is additive, not a replacement for the stat-based checks in
//! `cache.rs`: a missed or coalesced OS event (or a root that failed to
//! register a watch, e.g. an exhausted inotify watch limit) still gets
//! caught correctly on the next read, just without the eager-eviction
//! benefit. The watcher only ever makes stale entries disappear sooner; it
//! never lets one through.
//!
//! Event handling is modeled on graphify's watch.py (debounce + a queue that
//! survives a busy consumer), adapted for a single long-running process
//! rather than independent OS processes racing over a rebuild: there,
//! concurrent git-hook processes need an `flock` plus a `.pending_changes`
//! file so a losing process doesn't drop its change set. Here there is only
//! ever one consumer task draining the event channel, so that task's
//! sequential loop *is* the lock — no cross-process coordination needed.
//! The channel's own buffering plays the role of the pending-changes queue.

use dashmap::DashSet;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

use crate::cache::{DirCache, FileCache};

const DEBOUNCE: Duration = Duration::from_millis(300);
// Caps how many extra drain passes absorb events that arrive *while* a batch
// is being applied, so a continuous event storm (e.g. a large `git checkout`)
// can't livelock this task forever — mirrors graphify's
// _PENDING_DRAIN_MAX_PASSES.
const MAX_DRAIN_PASSES: usize = 20;

pub struct RepoWatcher {
    watched_roots: DashSet<PathBuf>,
    watcher: Mutex<RecommendedWatcher>,
}

impl RepoWatcher {
    /// Spawn the background drain task and return a handle tools can register
    /// watch roots with. `dir_cache`/`file_cache` are the caches events get
    /// applied to.
    pub fn spawn(dir_cache: Arc<DirCache>, file_cache: Arc<FileCache>) -> Arc<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<PathBuf>();

        let event_tx = tx.clone();
        let notify_watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            let Ok(event) = res else { return };
            if !is_relevant(&event.kind) {
                return;
            }
            for path in event.paths {
                // Sync send from notify's own callback thread; unbounded so
                // it never blocks that thread on a full channel.
                let _ = event_tx.send(path);
            }
        })
        .expect("failed to create filesystem watcher");

        let this = Arc::new(Self {
            watched_roots: DashSet::new(),
            watcher: Mutex::new(notify_watcher),
        });

        tokio::spawn(drain_loop(rx, dir_cache, file_cache));
        this
    }

    /// Start watching `root` (recursively) if it isn't already covered by a
    /// previously registered watch. Best-effort: a registration failure
    /// (nonexistent path, inotify watch-limit exhaustion, ...) is swallowed —
    /// correctness still holds via the stat-based checks, this only forgoes
    /// eager eviction for that root.
    pub fn ensure_watching(&self, root: &Path) {
        let root = match root.canonicalize() {
            Ok(p) => p,
            Err(_) => return,
        };
        if self.watched_roots.iter().any(|w| root.starts_with(w.as_path())) {
            return;
        }
        let Ok(mut guard) = self.watcher.lock() else { return };
        if guard.watch(&root, RecursiveMode::Recursive).is_ok() {
            self.watched_roots.insert(root);
        }
    }
}

fn is_relevant(kind: &EventKind) -> bool {
    matches!(kind, EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_))
}

async fn drain_loop(
    mut rx: mpsc::UnboundedReceiver<PathBuf>,
    dir_cache: Arc<DirCache>,
    file_cache: Arc<FileCache>,
) {
    loop {
        // Block until at least one event arrives.
        let Some(first) = rx.recv().await else { return }; // sender dropped -> shut down
        let mut batch = vec![first];

        // Debounce: keep absorbing events until DEBOUNCE has passed with no
        // new arrivals, so a burst of writes to the same file/tree collapses
        // into one invalidation pass instead of one per event.
        loop {
            match tokio::time::timeout(DEBOUNCE, rx.recv()).await {
                Ok(Some(path)) => batch.push(path),
                Ok(None) => break, // sender dropped
                Err(_elapsed) => break, // quiet period reached
            }
        }

        apply_batch(&batch, &dir_cache, &file_cache);

        // Late-arrival drain: events that landed while we were applying the
        // batch above are already sitting in the channel (its buffering is
        // the "pending queue"). Keep draining+applying until it's empty or
        // we hit the pass cap, so a continuous storm eventually quiesces
        // instead of trailing unapplied events indefinitely.
        for _ in 0..MAX_DRAIN_PASSES {
            let mut late = Vec::new();
            while let Ok(path) = rx.try_recv() {
                late.push(path);
            }
            if late.is_empty() {
                break;
            }
            apply_batch(&late, &dir_cache, &file_cache);
        }
    }
}

fn apply_batch(paths: &[PathBuf], dir_cache: &DirCache, file_cache: &FileCache) {
    for path in paths {
        file_cache.invalidate(path);
        // A changed path invalidates its parent's directory listing (an
        // add/remove/rename changes the parent's mtime) and, if the changed
        // path is itself a directory, its own listing too.
        if let Some(parent) = path.parent() {
            dir_cache.invalidate(parent);
        }
        dir_cache.invalidate(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread::sleep;
    use std::time::{Duration as StdDuration, SystemTime};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "mcp-native-core-watcher-test-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    async fn wait_until<F: Fn() -> bool>(cond: F, timeout: StdDuration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if cond() {
                return true;
            }
            tokio::time::sleep(StdDuration::from_millis(50)).await;
        }
        cond()
    }

    #[tokio::test]
    async fn watcher_evicts_file_cache_entry_on_modify() {
        let dir = TempDir::new();
        let file_path = dir.0.join("a.rs");
        fs::write(&file_path, "fn a() {}").unwrap();
        let meta = fs::metadata(&file_path).unwrap();
        let (original_mtime, original_len) = (meta.modified().unwrap(), meta.len());

        let dir_cache = Arc::new(DirCache::new());
        let file_cache = Arc::new(FileCache::new());
        let watcher = RepoWatcher::spawn(dir_cache.clone(), file_cache.clone());
        watcher.ensure_watching(&dir.0);

        let nodes = Arc::new(vec![serde_json::json!({"line": 1, "declaration": "fn a() {}"})]);
        file_cache.store(file_path.clone(), original_mtime, original_len, nodes);
        // Sanity check: querying with the exact stored (mtime, len) hits.
        assert!(file_cache.get_if_fresh(&file_path, original_mtime, original_len).is_some());

        sleep(StdDuration::from_millis(1100));
        fs::write(&file_path, "fn a() {}\nfn b() {}").unwrap();

        // Re-querying with the ORIGINAL (mtime, len) — not the new ones —
        // isolates proactive eviction from the (already-tested) lazy
        // staleness check: a lazy-only cache would still return Some here
        // since those are exactly the values it was stored under.
        let evicted = wait_until(
            || file_cache.get_if_fresh(&file_path, original_mtime, original_len).is_none(),
            StdDuration::from_secs(3),
        )
        .await;
        assert!(evicted, "watcher should have evicted the changed file's cache entry");
    }

    #[tokio::test]
    async fn watcher_evicts_dir_cache_entry_on_new_file() {
        let dir = TempDir::new();
        fs::write(dir.0.join("a.rs"), "fn a() {}").unwrap();

        let dir_cache = Arc::new(DirCache::new());
        let file_cache = Arc::new(FileCache::new());
        let watcher = RepoWatcher::spawn(dir_cache.clone(), file_cache.clone());
        watcher.ensure_watching(&dir.0);

        let mut out = Vec::new();
        dir_cache.crawl(&dir.0, &[], &mut out);
        assert_eq!(out.len(), 1);

        sleep(StdDuration::from_millis(1100));
        fs::write(dir.0.join("b.rs"), "fn b() {}").unwrap();

        let evicted = wait_until(
            || {
                let mut probe = Vec::new();
                dir_cache.crawl(&dir.0, &[], &mut probe).dirs_rescanned > 0
            },
            StdDuration::from_secs(3),
        )
        .await;
        assert!(evicted, "watcher should have evicted the directory's cached listing");
    }

    #[tokio::test]
    async fn ensure_watching_skips_already_covered_subdirectory() {
        let dir = TempDir::new();
        let sub = dir.0.join("sub");
        fs::create_dir_all(&sub).unwrap();

        let dir_cache = Arc::new(DirCache::new());
        let file_cache = Arc::new(FileCache::new());
        let watcher = RepoWatcher::spawn(dir_cache, file_cache);

        watcher.ensure_watching(&dir.0);
        watcher.ensure_watching(&sub);

        assert_eq!(watcher.watched_roots.len(), 1, "a subdirectory of an already-watched root should not get its own watch");
    }
}
