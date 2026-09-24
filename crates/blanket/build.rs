fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Wheel repair lengthens dylib install names. Reserve space so those
        // load commands cannot overwrite the first instructions in __text.
        println!("cargo::rustc-link-arg-cdylib=-Wl,-headerpad_max_install_names");
    }
}
