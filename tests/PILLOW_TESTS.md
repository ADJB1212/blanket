# Copied Pillow tests

The tests below come from Pillow 13.0.0, commit
`293218ad059d994baf1777d882911bed6a49704a`, in its `Tests/` directory.
The source checkout is available locally as `pil_source/`; running these copied
tests does not require that checkout.

Pillow's MIT-CMU license and copyright notices are retained in
[`licenses/Pillow-LICENSE`](../licenses/Pillow-LICENSE). The image fixtures in
`pillow_images/` are copied from the same revision's `Tests/images/` directory.

| Local module | Upstream module |
| --- | --- |
| `test_imagecolor.py` | `test_imagecolor.py` |
| `test_imagestat.py` | `test_imagestat.py` |
| `test_imagechops.py` | `test_imagechops.py` |
| `test_imageenhance.py` | `test_imageenhance.py` |
| `test_image_getbbox.py` | `test_image_getbbox.py` |
| `test_image_crop.py` | `test_image_crop.py` |
| `test_image_transpose.py` | `test_image_transpose.py` |
| `test_image_histogram.py` | `test_image_histogram.py` |
| `test_image_split.py` | `test_image_split.py` |
| `test_image_getextrema.py` | `test_image_getextrema.py` |
| `test_image_frombytes.py` | `test_image_frombytes.py` |
| `test_image_filter.py` | `test_image_filter.py` |
| `test_image_resize.py` | `test_image_resize.py` |
| `test_jpeg_options.py` | `test_file_jpeg.py` |
| `test_imageops.py` | `test_imageops.py` |
| `test_apng.py` | `test_file_apng.py` |
| `test_image_convert.py` | `test_image_convert.py` |
| `test_imagedraw.py` | `test_imagedraw.py` |
| `test_image_copy.py` | `test_image_copy.py` |
| `test_image_putalpha.py` | `test_image_putalpha.py` |
| `test_image_array.py` | `test_image_array.py` |

Retained copied tests preserve upstream assertions and tolerances. Redundant
smoke cases may be omitted or consolidated with stronger regressions, and
filter parameterization excludes combinations whose bodies performed no checks.
Imports target `blanket`, helper imports target
`pillow_helpers`, and fixture paths target `pillow_images`. Import cleanup and
lint suppressions accommodate this repository. `pillow_helpers.py` supplies
the required upstream assertion and hopper helpers without Pillow at runtime;
diagnostic image uploads are disabled.

Selection is limited to passing upstream contracts. JPEG includes the completed
save-option contracts in `TestFileJpeg`; resize includes `TestImageResize`,
`TestReducingGapResize`, and the passing cross-platform fixture contract.
Drawing, APNG, and conversion include wholly passing test groups from the
recorded compatibility runs. Tests for unavailable private internals, unsupported
features, and intentional differences were not copied. In particular, this
excludes Pillow's default-font assertions and APNG 16-bit mode assertions.
The upstream runner continues to track failures and report intentional
deselections separately.

Overlapping handwritten tests were removed, including 53 redundant test
functions across color, statistics, drawing, ImageOps, enhancement, filters,
resize, and JPEG modules. Shared crop, channel, split, and transpose tests were
narrowed to the contracts absent from the copied upstream tests. Tests with additional
coverage remain alongside the copied tests, including SIMD boundaries, native
sample depths, metadata ownership, larger inputs, extended color syntax, and
the public-domain font. These regressions are not claimed to be upstream tests.

Band-name checks are covered by `test_mode_metadata.py`; raw byte return types
are covered by `test_raw_buffers.py`. The standalone getbands and tobytes smoke
modules were removed. Split shape assertions now share the split/merge tests.

Run any module normally, for example:

```sh
uv run --no-sync pytest tests/test_image_crop.py tests/test_imageops.py
```
