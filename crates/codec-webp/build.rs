fn main() {
    if std::env::var_os("CARGO_FEATURE_NO_VENDORED_WEBP").is_some() {
        for library in ["libwebpmux", "libwebpdemux"] {
            pkg_config::Config::new().probe(library).expect("WebP animation library");
        }
    }
}
