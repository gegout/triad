use std::fs;
use std::path::Path;

fn main() {
    let count_file = Path::new("build_count.txt");
    let count: u64 = if count_file.exists() {
        fs::read_to_string(count_file)
            .unwrap_or_default()
            .trim()
            .parse()
            .unwrap_or(0) + 1
    } else {
        1
    };
    let _ = fs::write(count_file, count.to_string());

    let base_version = env!("CARGO_PKG_VERSION");
    let dynamic_version = format!("{}+build.{}", base_version, count);

    println!("cargo:rustc-env=TRIAD_BUILD_VERSION={}", dynamic_version);
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
