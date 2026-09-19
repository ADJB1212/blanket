use rustc_version::{Channel, version_meta};

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Wheel repair lengthens dylib install names. Reserve space so those
        // load commands cannot overwrite the first instructions in __text.
        println!("cargo::rustc-link-arg-cdylib=-Wl,-headerpad_max_install_names");
    }
    println!("cargo::rustc-check-cfg=cfg(RUSTC_IS_NIGHTLY)");
    if version_meta().expect("failed to inspect the Rust compiler").channel == Channel::Nightly {
        println!("cargo::rustc-cfg=RUSTC_IS_NIGHTLY");
        println!("cargo::warning=Using Rust Nightly Build");
    }
}
