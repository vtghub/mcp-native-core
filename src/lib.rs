use regex::Regex;

pub fn rust_pattern() -> Regex {
    Regex::new(r"^\s*(pub\s+)?(async\s+)?(fn|struct|enum|trait|impl)\s+([a-zA-Z0-9_<>]+)").unwrap()
}

pub fn python_pattern() -> Regex {
    Regex::new(r"^\s*(def|class)\s+([a-zA-Z0-9_]+)").unwrap()
}

pub fn csharp_pattern() -> Regex {
    Regex::new(r"^\s*(public|private|protected|internal\s+)?(class|struct|interface|enum|void|[a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_<>]+)\s*\(").unwrap()
}

pub fn matches_structural_line(ext: &str, line: &str) -> bool {
    match ext {
        "rs" => rust_pattern().is_match(line),
        "py" => python_pattern().is_match(line),
        "cs" => csharp_pattern().is_match(line),
        _ => false,
    }
}
