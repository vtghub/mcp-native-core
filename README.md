# ⚡ mcp-native-core

A blazing-fast, ultra-low-latency local server for the **Model Context Protocol (MCP)**, written entirely in Rust. 

Designed specifically to optimize context windows and reduce I/O latency for local AI agents (like Claude Code), `mcp-native-core` bypasses the heavy footprint of Node.js/Python servers by utilizing zero-copy memory mapping (`memmap2`) and multi-threaded native execution.

## 🧠 The Problem it Solves
Standard MCP file-system servers rely on traditional buffered file reading and transmit entire files to the LLM. For massive codebases, this results in:
1. **High Latency:** Crawling thousands of files blocks the agent's reasoning loop.
2. **Context Bloat:** Pumping thousands of lines of code into a context window wastes tokens and degrades the LLM's attention mechanism.

## 🚀 Native Architecture
`mcp-native-core` solves this with two high-performance tools:

* **`fast_search`**: An embedded, multi-threaded regex crawler. By memory-mapping files directly to RAM, it executes codebase-wide searches in microseconds, feeding only the exact relevant lines back to the agent.
* **`parse_structure`**: An AST-lite structural tokenizer. Instead of reading a 2,000-line source file, the agent can call this tool to extract just the structural skeleton (structs, classes, functions, and interfaces), understanding the file's entire architecture in under 50 tokens.

## ⚙️ Installation & Usage

### 1. Build the binary
Ensure you have Rust installed (`rustup`), then compile for release:
```bash
cargo build --release