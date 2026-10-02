fn main() {
    let lock_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
    println!("cargo::rerun-if-changed={}", lock_path.display());
    let lock: toml::Table = std::fs::read_to_string(lock_path)
        .expect("read Cargo.lock")
        .parse()
        .expect("parse Cargo.lock");
    let packages = lock["package"].as_array().expect("Cargo.lock packages");
    for name in ["png", "turbojpeg", "tiff", "gif", "image", "ravif", "dav1d"] {
        let package = packages
            .iter()
            .find(|package| package["name"].as_str() == Some(name))
            .expect("codec backend in Cargo.lock");
        let version = package["version"].as_str().expect("codec backend version");
        println!("cargo::rustc-env=BLANKET_BACKEND_{}={version}", name.to_ascii_uppercase());
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Wheel repair lengthens dylib install names. Reserve space so those
        // load commands cannot overwrite the first instructions in __text.
        println!("cargo::rustc-link-arg-cdylib=-Wl,-headerpad_max_install_names");
    }
}
