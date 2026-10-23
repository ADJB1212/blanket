from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError, features


def patterned(mode: str, size: tuple[int, int] = (13, 7)) -> Image.Image:
    if mode == "1":
        result = Image.new(mode, size)
        result.putdata([255 if i % 3 else 0 for i in range(size[0] * size[1])])
        return result
    if mode in ("I", "F"):
        code = "i" if mode == "I" else "f"
        return Image.frombytes(mode, size, struct.pack("<" + code * (size[0] * size[1]), *[i - 50 for i in range(size[0] * size[1])]))
    result = Image.frombytes(mode, size, bytes((i * 37 + 13) % 256 for i in range(size[0] * size[1] * len(mode))))
    if mode == "P":
        result.putpalette(bytes((i * 19 + 71) % 256 for i in range(768)))
    return result


@pytest.mark.parametrize(("format", "mode"), [("PCX", "1"), ("PCX", "L"), ("PCX", "P"), ("PCX", "RGB"), ("SGI", "L"), ("SGI", "RGB"), ("SGI", "RGBA"), ("XBM", "1"), ("MSP", "1")])
@pytest.mark.parametrize("size", [(13, 7), (1, 3), (8, 3), (17, 3), (33, 5), (129, 7)])
def test_legacy_interoperability(format: str, mode: str, size: tuple[int, int]) -> None:
    source = patterned(mode, size)
    for writer in (source, source.to_pillow()):
        stream = BytesIO()
        writer.save(stream, format)
        actual = Image.open(stream, formats=[format])
        expected = PillowImage.open(stream)
        assert (actual.format, actual.mode, actual.size) == (format, mode, source.size)
        assert actual.tobytes() == source.tobytes() == expected.tobytes()
        assert actual.convert("RGB").tobytes() == expected.convert("RGB").tobytes()
        with pytest.raises(UnidentifiedImageError, match="not in formats"):
            Image.open(stream, formats=["PNG"])


@pytest.mark.parametrize("big", [False, True])
def test_spider_float_and_first_stack_image(big: bool) -> None:
    for stack in (False, True):
        header = [0.0] * 256
        for key, value in {1: 1, 2: 3, 5: 1, 12: 4, 13: 64, 22: 1024, 23: 16, 24: int(stack), 26: 2 if stack else 0}.items():
            header[key - 1] = value
        packed = struct.pack((">" if big else "<") + "256f", *header)
        values = [i * 1.25 - 7 for i in range(12)]
        data = packed * (2 if stack else 1) + struct.pack((">" if big else "<") + "12f", *values)
        actual = Image.open(BytesIO(data), formats=["SPIDER"])
        assert (actual.mode, actual.size, actual.format) == ("F", (4, 3), "SPIDER")
        assert actual.tobytes() == struct.pack("<12f", *values)


@pytest.mark.parametrize("bits", [8, 16, 32, -32, -64])
@pytest.mark.parametrize("width", [3, 15, 16, 17, 33, 129])
def test_fits_sample_depth_and_orientation(bits: int, width: int) -> None:
    values = [0, 1, 17, 255, 31, 42] * width
    cards = ["SIMPLE  =                    T", f"BITPIX  = {bits:20}", "NAXIS   =                    2", f"NAXIS1  = {width:20}", "NAXIS2  =                    6", "END"]
    header = b"".join(card.encode().ljust(80) for card in cards).ljust(2880)
    code = {8: "B", 16: "H", 32: "i", -32: "f", -64: "d"}[bits]
    data = header + struct.pack(f">{len(values)}{code}", *values).ljust(2880, b"\0")
    actual = Image.open(BytesIO(data), formats=["FITS"])
    expected = PillowImage.open(BytesIO(data))
    assert (actual.mode, actual.size, actual.format) == (expected.mode, expected.size, "FITS")
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("predictor", [1, 2])
def test_tiff_lzw_strips_preserve_samples_and_reject_truncation(mode: str, predictor: int) -> None:
    source = patterned(mode, (513, 517))
    stream = BytesIO()
    source.to_pillow().save(stream, "TIFF", compression="tiff_lzw", tiffinfo={317: predictor}, strip_size=8192)
    data = stream.getvalue()
    expected = PillowImage.open(BytesIO(data))
    assert len(expected.tag_v2[273]) > 1
    assert Image.open(BytesIO(data)).tobytes() == expected.tobytes() == source.tobytes()
    offset = expected.tag_v2[273][-1]
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data[: offset + 1]))


def test_pixar_rgb() -> None:
    header = bytearray(1024)
    header[:4] = b"\x80\xe8\0\0"
    struct.pack_into("<HH", header, 416, 3, 5)
    struct.pack_into("<HH", header, 424, 14, 2)
    raw = bytes(range(45))
    data = bytes(header) + raw
    actual = Image.open(BytesIO(data), formats=["PIXAR"])
    assert (actual.mode, actual.size, actual.format) == ("RGB", (5, 3), "PIXAR")
    assert actual.tobytes() == PillowImage.open(BytesIO(data)).tobytes() == raw


@pytest.mark.parametrize("depth", [1, 4, 8, 24, 32])
@pytest.mark.parametrize("kind", [1, 2, 3])
@pytest.mark.parametrize("w", [1, 7, 16, 17, 33, 129])
def test_sun_stride_channels_and_rle(depth: int, kind: int, w: int) -> None:
    h = 3
    stride = ((w * depth + 15) // 16) * 2
    raw = bytes((i * 37) % 256 for i in range(stride * h))
    payload = raw.replace(b"\x80", b"\x80\0") if kind == 2 else raw
    data = struct.pack(">8I", 0x59A66A95, w, h, depth, len(payload), kind, 0, 0) + payload
    actual = Image.open(BytesIO(data), formats=["SUN"])
    reference = struct.pack(">8I", 0x59A66A95, w, h, depth, len(raw), 1 if kind == 2 else kind, 0, 0) + raw
    expected = PillowImage.open(BytesIO(reference))
    assert (actual.mode, actual.size) == (expected.mode, expected.size)
    assert actual.tobytes() == expected.tobytes()


def test_sgi_rle_planes() -> None:
    header = bytearray(512)
    struct.pack_into(">HBB4H", header, 0, 474, 1, 1, 3, 3, 2, 3)
    rows = [bytes([0x83, i, i + 1, i + 2, 0]) for i in range(6)]
    data = bytes(header) + struct.pack(">6I", *[560 + i * 5 for i in range(6)]) + struct.pack(">6I", *[5] * 6) + b"".join(rows)
    actual = Image.open(BytesIO(data))
    assert actual.tobytes() == PillowImage.open(BytesIO(data)).tobytes()


def test_sgi_16bit_samples() -> None:
    header = bytearray(512)
    struct.pack_into(">HBB4H", header, 0, 474, 0, 2, 3, 3, 2, 3)
    values = [i * 3001 for i in range(18)]
    actual = Image.open(BytesIO(bytes(header) + struct.pack(">18H", *values)))
    expected = [values[c * 6 + y * 3 + x] for y in (1, 0) for x in range(3) for c in range(3)]
    assert (actual.mode, actual.bit_depth) == ("RGB", 16)
    assert actual.tobytes() == struct.pack("<18H", *expected)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_xpm_write_and_read(mode: str) -> None:
    source = patterned(mode)
    if mode == "RGBA":
        source.putalpha(255)
    stream = BytesIO()
    source.save(stream, "XPM")
    actual = Image.open(stream, formats=["XPM"])
    assert actual.convert("RGB").tobytes() == source.convert("RGB").tobytes() == PillowImage.open(stream).convert("RGB").tobytes()


def test_xpm_transparency_and_multichar_keys() -> None:
    data = b'/* XPM */\nstatic char *im[] = {\n"3 2 3 2",\n"aa c #ff0000",\n"bb c None",\n"cc c #0022ff",\n"aabbcc",\n"ccbbaa"};'
    actual = Image.open(BytesIO(data))
    assert actual.mode == "RGBA"
    assert actual.tobytes() == PillowImage.open(BytesIO(data)).convert("RGBA").tobytes()
    stream = BytesIO()
    actual.save(stream, "XPM")
    assert Image.open(stream).tobytes() == actual.tobytes()


@pytest.mark.parametrize("w", [9, 17, 33, 129])
def test_xbm_short_and_nonbyte_width(w: int) -> None:
    rows = [1 | (1 << (w - 1)), ((1 << (w - 2)) - 1) << 1]
    words = [f"0x{(row >> shift) & 65535:04x}" for row in rows for shift in range(0, w, 16)]
    data = (f"#define im_width {w}\n#define im_height 2\nstatic short im_bits[] = {{" + ",".join(words) + "};").encode()
    actual = Image.open(BytesIO(data))
    assert actual.convert("L").tobytes() == bytes(255 if row & (1 << x) else 0 for row in rows for x in range(w))


@pytest.mark.parametrize(("bits", "planes"), [(1, 1), (8, 1), (8, 3)])
def test_pcx_uncompressed_rows(bits: int, planes: int) -> None:
    w, h = 33, 3
    stride = ((w * bits + 7) // 8 + 1) // 2 * 2
    header = bytearray(128)
    header[:4] = bytes([10, 5, 0, bits])
    struct.pack_into("<HH", header, 8, w - 1, h - 1)
    header[65] = planes
    struct.pack_into("<H", header, 66, stride)
    raw = bytes((i * 37) % 256 for i in range(stride * planes * h))
    actual = Image.open(BytesIO(bytes(header) + raw))
    if bits == 1:
        expected = bytes(255 if raw[y * stride + x // 8] & (128 >> (x % 8)) else 0 for y in range(h) for x in range(w))
        assert (actual.mode, actual.size, actual.convert("L").tobytes()) == ("1", (w, h), expected)
    else:
        expected = bytes(raw[(y * planes + c) * stride + x] for y in range(h) for x in range(w) for c in range(planes))
        assert (actual.mode, actual.size, actual.tobytes()) == ("RGB" if planes == 3 else "L", (w, h), expected)


@pytest.mark.parametrize("run", [b"\xc0\x07", b"\xff\x07", b"\xc1"])
def test_pcx_invalid_run(run: bytes) -> None:
    header = bytearray(128)
    header[:4] = bytes([10, 5, 1, 8])
    header[65] = 1
    struct.pack_into("<H", header, 66, 2)
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(bytes(header) + run))


@pytest.mark.parametrize("mode", ["1", "L", "P"])
@pytest.mark.parametrize("size", [(13, 7), (1, 3), (8, 3), (17, 3), (33, 5), (129, 7)])
def test_palm_write(mode: str, size: tuple[int, int]) -> None:
    source = patterned(mode, size)
    stream = BytesIO()
    source.save(stream, "PALM")
    data = stream.getvalue()
    w, h, stride, flags = struct.unpack_from(">4H", data)
    assert (w, h, data[13]) == (*source.size, 255)
    assert len(data) == 16 + stride * h + (1026 if mode != "1" else 0)
    assert flags == (0 if mode == "1" else 0x4000)
    if mode == "1":
        other = BytesIO()
        source.to_pillow().save(other, "PALM")
        assert data == other.getvalue()


@pytest.mark.parametrize("compression", ["tiff_lzw", "tiff_adobe_deflate", "jpeg"])
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_tiff_compression_interoperability(compression: str, mode: str) -> None:
    source = patterned(mode)
    for writer in (source, source.to_pillow()):
        stream = BytesIO()
        writer.save(stream, "TIFF", compression=compression)
        actual = Image.open(stream)
        expected = PillowImage.open(stream)
        assert expected.tag_v2[259] == {"tiff_lzw": 5, "tiff_adobe_deflate": 8, "jpeg": 7}[compression]
        assert (actual.mode, actual.size) == (mode, source.size)
        assert actual.tobytes() == expected.tobytes()
        if compression != "jpeg":
            assert actual.tobytes() == source.tobytes()


@pytest.mark.parametrize("compression", ["tiff_lzw", "tiff_adobe_deflate"])
@pytest.mark.parametrize("mode", ["RGBA", "CMYK", "LAB", "F"])
def test_tiff_compression_special_modes(compression: str, mode: str) -> None:
    source = patterned(mode)
    stream = BytesIO()
    source.save(stream, "TIFF", compression=compression)
    assert Image.open(stream).tobytes() == source.tobytes() == PillowImage.open(stream).tobytes()


@pytest.mark.parametrize("compression", ["tiff_lzw", "tiff_adobe_deflate"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_tiff_compression_wide(compression: str, mode: str) -> None:
    raw = struct.pack("<" + "H" * (15 * len(mode)), *[(i * 3001) % 65536 for i in range(15 * len(mode))])
    source = Image.frombytes(mode, (5, 3), raw, bit_depth=16)
    stream = BytesIO()
    source.save(stream, "TIFF", compression=compression)
    actual = Image.open(stream)
    assert actual.bit_depth == 16
    assert actual.tobytes() == raw


@pytest.mark.parametrize(("ext", "mode"), [("pcx", "RGB"), ("sgi", "RGBA"), ("rgb", "RGB"), ("bw", "L"), ("xbm", "1"), ("xpm", "RGB"), ("msp", "1"), ("palm", "1")])
def test_legacy_extension(tmp_path: Path, ext: str, mode: str) -> None:
    target = tmp_path / f"image.{ext}"
    patterned(mode).save(target)
    assert target.stat().st_size > 0


@pytest.mark.parametrize("format", ["PCX", "SGI", "XBM", "XPM", "MSP", "PALM", "FITS", "SPIDER", "PIXAR", "SUN"])
def test_legacy_feature(format: str) -> None:
    assert features.check(format.lower())


def test_deprecated_im_is_unsupported() -> None:
    header = b"Image type: Greyscale image\r\nImage size (x*y): 1*1\r\n"
    data = header.ljust(511, b"\0") + b"\x1a\xff"
    for formats in (None, ["IM"]):
        with pytest.raises(UnidentifiedImageError):
            Image.open(BytesIO(data), formats=formats)
    with pytest.raises(ValueError, match="unsupported image format"):
        Image.new("L", (1, 1)).save(BytesIO(), "IM")
    assert "im" not in features.get_supported()
    with pytest.warns(UserWarning, match="Unknown feature 'im'"):
        assert features.check("im") is False


@pytest.mark.parametrize("signature", [b"\x0a\x05", b"\x01\xda", b"DanM", b"LinS", b"\x59\xa6\x6a\x95", b"\x80\xe8\0\0", b"SIMPLE  =", b"/* XPM */", b"#define im_width 3\n#define im_height 2\n"])
def test_legacy_truncated_header(signature: bytes) -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(signature))
