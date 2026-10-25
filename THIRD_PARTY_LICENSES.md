# Distribution licenses and third-party components

Blanket's original source files remain available under the MIT license in
`LICENSE`. Official binary wheels include x265 and are distributed as a
combined work under GPL-3.0-or-later, retaining the notices and permissions of
each component. Package metadata uses `MIT AND GPL-3.0-or-later` to describe
these combined distribution requirements. This does not replace the MIT
license on Blanket's original source. The GPL and LGPL texts are included in
`licenses/libheif-COPYING`.

The binary wheels include complete third-party license and notice files under
their `.dist-info/licenses/licenses/third-party/` directory. This includes
the Rust dependency notices and notices for nested native dependencies.

Matching complete source archives named `pyblanket-sources-*.tar.bz2` are
provided alongside the wheels at
<https://github.com/ADJB1212/blanket/releases>. Select the release matching
the installed package version. These archives contain Blanket's source,
the locked Cargo dependency sources, the native dependency sources (including
submodules), provenance, licenses, and build scripts. See `RELEASE_SOURCES.md`
in the archive's project directory for rebuild instructions. The publishing
workflow requires these archives to be publicly available before PyPI upload.

Pre-built binary release wheels bundle or link against the following native
codec dependencies:

- aom: BSD-2-Clause
- dav1d: BSD-2-Clause
- libde265: LGPL-3.0-or-later
- libheif: LGPL-3.0-or-later
- libjxl: BSD-3-Clause
- libwebp: BSD-3-Clause
- LittleCMS: MIT
- libjpeg-turbo: IJG, BSD-3-Clause, and Zlib
- OpenJPEG (via openjpeg-sys): BSD-2-Clause
- x265: GPL-2.0-or-later

This software is based in part on the work of the Independent JPEG Group.
JPEG XL's nested dependencies include Brotli (MIT), Highway (Apache-2.0),
and LittleCMS (MIT). AVIF encoding also uses rav1e (BSD-2-Clause), and AVIF
container parsing uses mp4parse (MPL-2.0). The generated notice directory and
Cargo metadata in the corresponding-source archive provide the full inventory.

Outline font rendering and layout also use these Rust dependencies:

- ab_glyph and ab_glyph_rasterizer: Apache-2.0
- rustybuzz: MIT
- ttf-parser and owned_ttf_parser: MIT OR Apache-2.0
- unicode-bidi and unicode-script: MIT OR Apache-2.0

Their full license texts accompany the wheels and corresponding-source archives.
