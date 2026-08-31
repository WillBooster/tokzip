//! Prints the detected segments and the per-language sizes of one file.
//!
//!   cargo run --release --example segments -- <file>
fn main() {
    let path = std::env::args().nth(1).expect("file");
    let doc = std::fs::read(&path).expect("read");
    let frame = tokzip::compress(&doc, false);
    println!("{} bytes -> {} bytes (auto)", doc.len(), frame.len());
    for (lang, name) in [
        "text",
        "en-US",
        "ja-JP",
        "html",
        "css",
        "javascript",
        "typescript",
    ]
    .iter()
    .enumerate()
    {
        println!(
            "  forced {name:<11} {}",
            tokzip::compress_with_language(&doc, lang).len()
        );
    }
    for seg in tokzip::segments(&doc) {
        println!("  segment end {:>6} lang {}", seg.0, seg.1);
    }
}
