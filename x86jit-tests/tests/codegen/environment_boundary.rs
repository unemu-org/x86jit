//! Source tripwire: environment and filesystem diagnostics stay in opt-in modules.

use std::path::Path;

// Preserve byte positions while hiding comments and quoted literals, including
// raw strings used by test fixtures. This keeps braces in fixtures out of scope
// matching without needing a parser dependency in the core crate.
fn code_only(source: &str) -> Vec<u8> {
    let mut code = source.as_bytes().to_vec();
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        if bytes[i..].starts_with(b"//") {
            i += bytes[i..]
                .iter()
                .position(|&b| b == b'\n')
                .unwrap_or(bytes.len() - i);
        } else if bytes[i..].starts_with(b"/*") {
            i += 2;
            let mut depth = 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
        } else if bytes[i] == b'r' && bytes.get(i + 1).is_some_and(|b| *b == b'#' || *b == b'"') {
            let mut quote = i + 1;
            while bytes.get(quote) == Some(&b'#') {
                quote += 1;
            }
            if bytes.get(quote) != Some(&b'"') {
                i += 1;
                continue;
            }
            let hashes = quote - i - 1;
            i = quote + 1;
            while i < bytes.len() {
                if bytes[i] == b'"'
                    && bytes
                        .get(i + 1..i + 1 + hashes)
                        .is_some_and(|s| s.iter().all(|&b| b == b'#'))
                {
                    i += 1 + hashes;
                    break;
                }
                i += 1;
            }
        } else if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
        } else if bytes[i] == b'\'' && bytes.get(i + 2) == Some(&b'\'') {
            i += 3;
        } else if bytes[i..].starts_with(b"'\\") && bytes.get(i + 3) == Some(&b'\'') {
            i += 4;
        } else {
            i += 1;
            continue;
        }
        code[start..i].fill(b' ');
    }
    code
}

fn production_source(source: &str) -> String {
    let mut code = code_only(source);
    for (start, _) in source.match_indices("#[cfg(") {
        let Some(attr_end) = source[start..].find(")]") else {
            continue;
        };
        let attr: String = source[start..start + attr_end + 2]
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        if attr != "#[cfg(test)]" && !attr.starts_with("#[cfg(all(test,") {
            continue;
        }
        if code[start] != b'#' {
            continue;
        }
        let rest = &code[start..];
        let open = rest.iter().position(|&b| b == b'{');
        let semi = rest.iter().position(|&b| b == b';');
        let Some(open) = open.filter(|o| semi.is_none_or(|s| *o < s)) else {
            continue;
        };
        let mut depth = 0;
        let mut end = start + open;
        for &byte in &code[start + open..] {
            end += 1;
            if byte == b'{' {
                depth += 1;
            }
            if byte == b'}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
        code[start..end].fill(b' ');
    }
    String::from_utf8(code).unwrap()
}

fn check_tree(dir: &Path, environment: &Path) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            check_tree(&path, environment);
            continue;
        }
        if path.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        if path == environment {
            continue;
        }
        // External test-only modules are checked at their declarations below.
        if path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("_tests.rs")
        {
            let mut parent = path.parent().unwrap().join("mod.rs");
            if !parent.exists() {
                parent = path.parent().unwrap().join("lib.rs");
            }
            let declaration = format!(
                "#[cfg(test)]\nmod {};",
                path.file_stem().unwrap().to_string_lossy()
            );
            assert!(
                std::fs::read_to_string(parent)
                    .unwrap()
                    .contains(&declaration),
                "{} must be test-only",
                path.display()
            );
            continue;
        }
        let production = production_source(&source);
        for forbidden in [
            "std::env",
            "std::fs",
            "File::open",
            "File::create",
            "OpenOptions",
        ] {
            assert!(
                !production.contains(forbidden),
                "{}: production access {forbidden} must live in environment.rs",
                path.display()
            );
        }
    }
}

#[test]
fn embedder_environment_and_files_stay_feature_gated() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    for name in ["x86jit-core", "x86jit-cranelift"] {
        let src = root.join(name).join("src");
        let lib = std::fs::read_to_string(src.join("lib.rs")).unwrap();
        assert!(lib.contains("#[cfg(feature = \"environment\")]\nmod environment;"));
        check_tree(&src, &src.join("environment.rs"));
    }
    // Pin the scope matcher: neither inline test fixtures nor multiple test
    // modules may hide production accesses later in a source file.
    let sample = "#[cfg(test)] mod tests { let _ = \"}\"; std::env::var(\"X\"); }\nfn production() { std::env::var(\"Y\"); }";
    let production = production_source(sample);
    assert_eq!(production.matches("std::env").count(), 1);
}
