use mcp_native_core::matches_structural_line;

#[test]
fn detects_rust_fn() {
    assert!(matches_structural_line("rs", "fn execute() {"));
    assert!(matches_structural_line("rs", "pub fn new() -> Self {"));
    assert!(matches_structural_line("rs", "async fn execute(&self) {"));
    assert!(matches_structural_line("rs", "pub async fn run() {"));
}

#[test]
fn detects_rust_type_definitions() {
    assert!(matches_structural_line("rs", "struct JsonRpcRequest {"));
    assert!(matches_structural_line("rs", "pub struct McpServerState {"));
    assert!(matches_structural_line("rs", "enum Status {"));
    assert!(matches_structural_line("rs", "trait McpTool: Send + Sync {"));
    assert!(matches_structural_line("rs", "impl McpServerState {"));
}

#[test]
fn ignores_non_declaration_rust_lines() {
    assert!(!matches_structural_line("rs", "let x = 5;"));
    assert!(!matches_structural_line("rs", "// fn commented_out() {}"));
    assert!(!matches_structural_line("rs", "    self.crawl_directory(&path);"));
}

#[test]
fn detects_python_def_and_class() {
    assert!(matches_structural_line("py", "def handle_request():"));
    assert!(matches_structural_line("py", "class Server:"));
    assert!(matches_structural_line("py", "    def nested(self):"));
}

#[test]
fn ignores_non_declaration_python_lines() {
    assert!(!matches_structural_line("py", "x = 5"));
    assert!(!matches_structural_line("py", "return self.value"));
}

#[test]
fn detects_csharp_members() {
    assert!(matches_structural_line("cs", "void Run() {"));
    assert!(matches_structural_line("cs", "internal void Run() {"));
}

#[test]
fn returns_false_for_unknown_extensions() {
    assert!(!matches_structural_line("txt", "fn execute() {"));
    assert!(!matches_structural_line("", "class Foo {"));
}
