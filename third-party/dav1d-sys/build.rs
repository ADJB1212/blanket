fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    pkg_config::Config::new()
        .atleast_version("1.3.0")
        .probe("dav1d")
        .expect("dav1d >= 1.3.0 must be available through pkg-config");
}
