# mcp-native-core

A native Rust MCP (Model Context Protocol) server, speaking newline-delimited JSON-RPC 2.0 over stdio. Three tools: `fast_search` (mmap'd multi-threaded regex search), `parse_structure` (regex-based structural skeleton extraction), and `semantic_search` (local embedding-based conceptual search). See `README.md` for the user-facing pitch.

## Build & test

```bash
cargo build --release          # full build, all three tools
cargo test --release           # unit tests (src/*.rs) + tests/regex_matching.rs

cargo build --release --no-default-features   # drops semantic_search + candle/hf-hub/tokenizers/rand
cargo test --release --no-default-features
```

Both configurations must build and pass their full test suite — CI-worthy changes touch both. The `--no-default-features` build is ~4 MiB vs. ~22 MiB with the `semantic-search` feature on.

Manual smoke test (the binary is a stdio server, so pipe a JSON-RPC line in):
```bash
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n' | ./target/release/mcp-native-core
```

## Module map

- `main.rs` — `McpTool` trait, `FastSearchTool`/`ParseStructureTool` (semantic_search's tool lives in its own module), JSON-RPC loop, tool registration.
- `cache.rs` — `DirCache`/`FileCache`/`ContentCache`/`VectorCache`. All keyed by absolute path; freshness is a plain `(mtime, len)` check, no content hashing. `ContentCache`/`VectorCache` are additionally bounded by a byte-budget LRU (see their doc comments for the eviction algorithm and its documented approximation trade-offs).
- `watcher.rs` — `RepoWatcher`: one `notify` filesystem watcher + a debounced drain loop that proactively evicts stale cache entries. This is a proactive optimization layered on top of the stat-based lazy checks in `cache.rs`, never a substitute for them — a missed/coalesced event or a failed watch registration just forgoes eager eviction, correctness still holds.
- `extractors.rs` / `search_backend.rs` — trait+registry pluggable strategies for `parse_structure`/`fast_search` respectively (`ExtractorRegistry`, `SearchBackendRegistry`). Adding a language or search mode means implementing the trait and registering it in `main()`, not touching the tool's `execute()`.
- `embeddings.rs`, `quantizer.rs`, `semantic_search.rs` — the `semantic-search` feature (see below).

## The `semantic-search` feature

Gated in `Cargo.toml`, **on by default**; `candle-core`/`candle-nn`/`candle-transformers`/`hf-hub`/`tokenizers`/`rand`/`statrs` are all `optional = true` under it. When adding code that touches embeddings/quantization/the vector cache, gate it with `#[cfg(feature = "semantic-search")]` and verify the `--no-default-features` build still compiles — `cache.rs`'s `VectorCache` has a zero-field no-op stand-in (`new()`/`invalidate()` only) for exactly this reason, so `watcher.rs`/`main.rs` never need feature-conditional branching beyond the tool registration itself.

- `embeddings.rs`: `Embedder` trait + `BertEmbedder` (candle CPU BERT, mean-pooled, L2-normalized). Downloads `sentence-transformers/all-MiniLM-L6-v2` from the HF Hub on first use (lazily, memoized including failures); `MCP_EMBEDDING_MODEL_DIR` points at local `config.json`/`tokenizer.json`/`model.safetensors` instead for offline use, `MCP_EMBEDDING_MODEL_ID` picks a different Hub model.
- `quantizer.rs`: `Rotation` (fixed-seed random orthogonal matrix, dense O(d²) Gram-Schmidt — not turbovec's O(d log d) structured rotation) + `QuantizedVector` (8-bit/coordinate Lloyd-Max quantizer). The Lloyd-Max codebook (`codebook_for_dim`, memoized per `dim`) is built analytically against the *exact* `Beta((d-1)/2, (d-1)/2)` distribution a rotated unit-vector coordinate follows — via `statrs`'s `Beta` cdf/pdf plus a hand-rolled Lloyd-Max loop and Simpson's-rule integration, not a data-fit affine range. `QuantizedVector` also stores a length-renormalization scalar (`1 / ⟨u, x̂⟩`, computed once at encode time) that corrects the systematic inner-product-shrinkage bias scalar quantization introduces — applied at `score()`, not baked into `decode()`. Both pieces were adapted after reading [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)'s `codebook.rs`/`encode.rs` (own implementation, credited in the module's doc comment, not copied). Still a **simplified, documented stand-in** for the full TurboQuant paper (arXiv:2504.19874) and for turbovec itself: no bit-packing below 1 byte/coordinate, no SIMD scoring kernels, no per-coordinate (TQ+) calibration fit from the actual corpus (turbovec's own tuning needs ~1,000+ vectors before that fit is meaningful — most single-repo `semantic_search` calls won't index that many chunks). Don't present it as a literal reproduction in docs/comments; it's sized for a single-repo code index, not a multi-million-vector store. If revisiting this file: `build_lloyd_max_codebook`'s initial centroids come from `Beta::inverse_cdf` (quantile-based init) rather than a uniform grid — a uniform grid leaves most cells of a concentrated Beta essentially empty, which freezes their centroids and breaks the strictly-increasing ordering `quantize_coordinate`'s binary search relies on; and the `pdf` closure clamps non-finite values to 0 because `Beta::pdf` is singular at the domain edges for shape parameter `a < 1` (only reachable via this function's own `dim < 2` clamp for degenerate test inputs, not by any real embedding model).
- `semantic_search.rs`: chunks files into 40-line windows (`chunk_lines`, a pure function — test chunk-boundary logic there without touching a model), embeds/quantizes/scores. A single chunk failing to embed must only drop that chunk, not the whole file (there was a real bug here once — an errant `?` inside the per-chunk loop nuked an entire file's index on one bad chunk).

### Testing without network access

This sandbox's proxy blocks `huggingface.co` (crates.io is allowlisted, HF Hub is not), so `BertEmbedder::load_from_hub` can't be exercised for real here. Two ways tests/verification work around that:
1. `embeddings.rs`'s unit tests build a tiny untrained BERT + WordLevel tokenizer entirely in-process via `candle_nn::VarMap` (fabricates randomly-initialized tensors on demand) — no network, no bundled files, but a real tokenize→forward→pool→normalize pass.
2. `embeddings.rs::tests::write_tiny_model_fixture_to_disk` (`#[ignore]`d) persists that same fixture to disk (`config.json`/`tokenizer.json`/`model.safetensors`, via `VarMap::save`) so you can point the *actual compiled binary* at it with `MCP_EMBEDDING_MODEL_DIR` and drive a real `tools/call` end-to-end when you need to verify the full pipeline (not just unit tests) without network:
   ```bash
   MCP_TEST_FIXTURE_DIR=/tmp/fixture cargo test --release -- --ignored write_tiny_model_fixture_to_disk
   MCP_EMBEDDING_MODEL_DIR=/tmp/fixture ./target/release/mcp-native-core
   ```
   Rankings from this fixture are meaningless (random weights) — it only proves the plumbing, not embedding quality.

## Conventions

- CPU-bound synchronous work (mmap+regex, candle forward passes) goes through `tokio::task::spawn_blocking`, never runs directly on an async task.
- `stdout` is written from a single dedicated writer task fed by an mpsc channel — `tokio::io::stdout()` is not safe to write to concurrently. On stdin EOF, `main()` drains in-flight request tasks and awaits the writer task before returning, so a response that was still being computed/flushed isn't dropped (this was a real, previously-shipped bug).
- Tests favor a real temp-directory-and-filesystem style (see `TempDir` helpers in `cache.rs`/`watcher.rs`) over mocking, and a fake/stub implementation of a trait (`StubExtractor`, `StubBackend`) over mocking frameworks.
- Git branches: `develop` is the integration branch; `main` is released from `develop` via a separate promotion PR (see git history — features merge to `develop` first, then get their own PR from `develop` into `main`).
