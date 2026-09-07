use rustc_version::{Channel, version_meta};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(RUSTC_IS_NIGHTLY)");
    if version_meta().expect("failed to inspect the Rust compiler").channel == Channel::Nightly {
        println!("cargo::rustc-cfg=RUSTC_IS_NIGHTLY");
        println!("cargo::warning=Using Rust Nightly Build");
    }
}
