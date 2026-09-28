# Codec build layout

`blanket-codecs` preserves the public dispatch API and compressor helper
re-exports. Format implementations compile in separate crates:

| Crate suffix | Formats |
| --- | --- |
| `png` | PNG |
| `jpeg` | JPEG |
| `jxl` | JPEG XL |
| `tiff` | TIFF |
| `webp` | WebP |
| `heif` | HEIC/HEIF |
| `avif` | AVIF |
| `bmp-ico` | BMP and ICO |
| `gif` | GIF |
| `pdf` | PDF output |

All names use the `blanket-codec-` prefix. `blanket-codec-common` holds save
options, dimension limits, and error conversion without codec dependencies.
`blanket-codec-image` adapts the Rust `image` crate's decoded images.
BMP/ICO also uses the PNG backend for embedded PNG payloads.

JPEG, JPEG XL, HEIF, and PDF do not depend on `image`, allowing their backends
to compile while its dependencies are building. Cargo unifies `image`
features for the remaining backends in a workspace build. Splitting Blanket's
code does not parallelize compilation within upstream libraries or final
linking. Use `cargo build --timings` to inspect overlap; compare revisions with
the same profile, features, toolchain, and dependency-cache state.

The extension forwards `jxl-vendored`, `embedded-libheif`, and
`no-vendored-webp` through this crate to the JXL, HEIF, and WebP crates.
Format-specific unit tests live beside their implementations; dispatch tests
remain here. For example:

```sh
cargo test -p blanket-codecs -p blanket-codec-jxl -p blanket-codec-heif -p blanket-codec-tiff --features no-vendored-webp
```
