# ⚡ mcp-native-core

A blazing-fast, ultra-low-latency local server for the **Model Context Protocol (MCP)**, written entirely in Rust. 

Designed specifically to optimize context windows and reduce I/O latency for local AI agents (like Claude Code), `mcp-native-core` bypasses the heavy footprint of Node.js/Python servers by utilizing zero-copy memory mapping (`memmap2`) and multi-threaded native execution.

## 🧠 The Problem it Solves
Standard MCP file-system servers rely on traditional buffered file reading and transmit entire files to the LLM. For massive codebases, this results in:
1. **High Latency:** Crawling thousands of files blocks the agent's reasoning loop.
2. **Context Bloat:** Pumping thousands of lines of code into a context window wastes tokens and degrades the LLM's attention mechanism.

## 🚀 Native Architecture
`mcp-native-core` solves this with three high-performance tools:

* **`fast_search`**: An embedded, multi-threaded regex crawler. By memory-mapping files directly to RAM, it executes codebase-wide searches in microseconds, feeding only the exact relevant lines back to the agent.
* **`parse_structure`**: An AST-lite structural tokenizer. Instead of reading a 2,000-line source file, the agent can call this tool to extract just the structural skeleton (structs, classes, functions, and interfaces), understanding the file's entire architecture in under 50 tokens.
* **`semantic_search`**: Finds code conceptually related to a natural-language query even when the exact keywords don't match — complementing `fast_search`'s exact/regex matching for "find the code that handles X" queries. Runs a local BERT-family sentence encoder (via [`candle`](https://github.com/huggingface/candle), no Python/ONNX runtime) over line-chunked files, then ranks chunks by cosine similarity. Each embedding is compressed with a [TurboQuant](https://arxiv.org/abs/2504.19874)-inspired quantizer — a fixed random rotation followed by a per-vector int8 scalar quantizer — shrinking the resident vector index to roughly a quarter of raw `f32` storage (a deliberately simplified stand-in for the paper's structured fast-Hadamard rotation and Beta-optimal quantizer, not a literal reproduction).

  On first use it downloads `sentence-transformers/all-MiniLM-L6-v2` from the Hugging Face Hub and caches it locally — every call after that (in this process or a later one) is fully offline. For air-gapped setups, or to use a different model, point these environment variables at your own files/model instead:
  * `MCP_EMBEDDING_MODEL_DIR` — a local directory with `config.json`, `tokenizer.json`, and `model.safetensors`, skipping the Hub entirely.
  * `MCP_EMBEDDING_MODEL_ID` — a different Hugging Face model id to download (default `sentence-transformers/all-MiniLM-L6-v2`).
  * `MCP_VECTOR_CACHE_MAX_BYTES` — byte budget for the in-memory embedding index (default 128 MiB, LRU-evicted like the other caches).

## ⚙️ Installation & Usage

### 1. Build the binary
Ensure you have Rust installed (`rustup`), then compile for release:
```bash
cargo build --release