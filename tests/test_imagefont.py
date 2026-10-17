from __future__ import annotations

import struct
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from math import ceil, floor
from pathlib import Path

import pytest
from PIL import Image as PILImage, ImageFont as PILFont
from PIL.FontFile import FontFile

from blanket import Image, ImageDraw, ImageFont


def _u16(*values: int) -> bytes:
    return struct.pack(">" + "H" * len(values), *values)


@pytest.fixture(scope="module")
def font_data() -> bytes:
    """Generate an outline font with Unicode mappings, kerning, and a ligature."""
    chars = " AVfiéΩЖאב\u0633\u0644\u0627\u0645क\u0301😀"
    mapping = {ord(char): i + 1 for i, char in enumerate(chars)}
    count = len(chars) + 2
    advances = [600] * count
    advances[mapping[ord(" ")]] = 300
    advances[mapping[ord("i")]] = 250
    advances[-1] = 700
    outlines = []
    offsets = [0]
    for glyph in range(count):
        width = advances[glyph] - 120 - glyph * 5
        data = struct.pack(">5h", 1, 0, 0, width, 700) + _u16(3, 0) + bytes([1] * 4)
        data += struct.pack(">4h", 0, width, 0, -width) + struct.pack(">4h", 0, 0, 700, 0)
        if glyph == mapping[ord(" ")]:
            data = b""
        outlines.append(data)
        offsets.append(offsets[-1] + len(data))
    head = struct.pack(">IIIIHHqqhhhhHHhhh", 0x10000, 0x10000, 0, 0x5F0F3CF5, 0, 1000, 0, 0, 0, -200, 700, 800, 0, 8, 2, 1, 0)
    hhea = struct.pack(">IhhhH" + "h" * 11 + "H", 0x10000, 800, -200, 0, 700, 0, 0, 700, 1, 0, 0, 0, 0, 0, 0, 0, count)
    groups = b"".join(struct.pack(">III", char, char, glyph) for char, glyph in sorted(mapping.items()))
    cmap = _u16(0, 1, 3, 10) + struct.pack(">I", 12) + _u16(12, 0) + struct.pack(">III", 16 + len(groups), 0, len(mapping)) + groups
    names = ["Blanket Test", "Regular"]
    strings = [name.encode("utf-16-be") for name in names]
    name = _u16(0, 2, 30)
    name += _u16(3, 1, 0x409, 1, len(strings[0]), 0) + _u16(3, 1, 0x409, 2, len(strings[1]), len(strings[0])) + b"".join(strings)
    kern = _u16(0, 1, 0, 20, 1, 1, 6, 0, 0, mapping[ord("A")], mapping[ord("V")]) + struct.pack(">h", -120)
    script_list = _u16(1) + b"DFLT" + _u16(8, 4, 0, 0, 0xFFFF, 1, 0)
    feature_list = _u16(1) + b"liga" + _u16(8, 0, 1, 0)
    ligature = _u16(count - 1, 2, mapping[ord("i")])
    ligature_set = _u16(1, 4) + ligature
    substitution = _u16(1, 8 + len(ligature_set), 1, 8) + ligature_set + _u16(1, 1, mapping[ord("f")])
    lookup_list = _u16(1, 4, 4, 0, 1, 8) + substitution
    gsub = struct.pack(">IHHH", 0x10000, 10, 10 + len(script_list), 10 + len(script_list) + len(feature_list)) + script_list + feature_list + lookup_list
    tables = {
        b"GSUB": gsub,
        b"cmap": cmap,
        b"glyf": b"".join(outlines),
        b"head": head,
        b"hhea": hhea,
        b"hmtx": b"".join(_u16(advance, 0) for advance in advances),
        b"kern": kern,
        b"loca": b"".join(struct.pack(">I", offset) for offset in offsets),
        b"maxp": struct.pack(">IH", 0x10000, count) + _u16(*([4] * 13)),
        b"name": name,
        b"post": struct.pack(">I", 0x30000) + bytes(28),
    }
    offset = 12 + 16 * len(tables)
    directory = struct.pack(">IHHHH", 0x10000, len(tables), 0, 0, 0)
    payload = b""
    for tag, data in sorted(tables.items()):
        padded = data + bytes(-len(data) % 4)
        checksum = sum(struct.unpack(">" + "I" * (len(padded) // 4), padded)) & 0xFFFFFFFF
        directory += tag + struct.pack(">III", checksum, offset, len(data))
        payload += padded
        offset += len(padded)
    return directory + payload


@pytest.fixture
def font(font_data: bytes) -> ImageFont.TrueTypeFont:
    return ImageFont.truetype(BytesIO(font_data), 20)


def test_truetype_paths_streams_metrics_and_variants(font_data: bytes, tmp_path: Path) -> None:
    path = tmp_path / "font.ttf"
    path.write_bytes(font_data)
    font = ImageFont.truetype(path, 20)
    reference = PILFont.truetype(BytesIO(font_data), 20)
    assert font.getname() == reference.getname() == ("Blanket Test", "Regular")
    assert font.getmetrics() == reference.getmetrics() == (16, 4)
    assert font.getlength("AV", features=["-kern"]) == pytest.approx(24)
    assert font.getlength("AV") == pytest.approx(21.6)
    assert font.getlength("AV") < font.getlength("A") + font.getlength("V")
    assert font.getlength("fi") == pytest.approx(14)
    assert font.getlength("fi", features=["-liga"]) == pytest.approx(17)
    assert font.font_variant(size=40).getlength("AV") == pytest.approx(2 * font.getlength("AV"))
    assert ImageFont.truetype(bytes(path), 20).getlength("AV") == font.getlength("AV")
    assert ImageFont.truetype(BytesIO(font_data), 20, layout_engine=ImageFont.Layout.BASIC).getlength("AV") == font.getlength("AV")


def test_otf_cff_loading() -> None:
    data = PILFont.load_default(size=20).font_bytes
    font = ImageFont.truetype(BytesIO(data), 20)
    assert font.getname() == ("Aileron", "Regular")
    assert any(bytes(font.getmask("Hello")))
    assert font.getlength("Hello") > 0


def test_font_collection_face_selection(font_data: bytes) -> None:
    offset = 20
    face = bytearray(font_data)
    count = struct.unpack_from(">H", face, 4)[0]
    for entry in range(count):
        position = 12 + 16 * entry + 8
        table_offset = struct.unpack_from(">I", face, position)[0]
        struct.pack_into(">I", face, position, table_offset + offset)
    collection = b"ttcf" + struct.pack(">IIII", 0x10000, 2, offset, offset) + face
    for index in (0, 1):
        font = ImageFont.truetype(BytesIO(collection), 20, index=index)
        assert font.getname() == ("Blanket Test", "Regular")
        assert any(bytes(font.getmask("AV")))
    with pytest.raises(OSError):
        ImageFont.truetype(BytesIO(collection), index=2)


@pytest.mark.parametrize("text,direction", [("éΩЖ😀", None), ("אב", "rtl"), ("سلام", "rtl"), ("क", None), ("AאבV", "ltr"), ("AV", "ttb")])
def test_unicode_layout_and_bounds(font: ImageFont.TrueTypeFont, text: str, direction: str | None) -> None:
    assert font.getlength(text, direction=direction) > 0
    mask, offset = font.getmask2(text, direction=direction)
    box = font.getbbox(text, direction=direction)
    assert offset == box[:2]
    assert mask.size == (box[2] - box[0], box[3] - box[1])
    assert len(bytes(mask)) == mask.size[0] * mask.size[1]
    assert any(bytes(mask))


@pytest.mark.parametrize("text", ["é", "Ω", "Ж", "א", "س", "क", "😀"])
def test_unicode_uses_mapped_glyphs(font: ImageFont.TrueTypeFont, text: str) -> None:
    assert bytes(font.getmask(text)) != bytes(font.getmask("\U0010ffff"))


@pytest.mark.parametrize("text,direction,visual", [("אב", "rtl", "בא"), ("אב", None, "בא"), ("AאבV", "ltr", "AבאV")])
def test_bidirectional_visual_order(font: ImageFont.TrueTypeFont, text: str, direction: str | None, visual: str) -> None:
    actual = Image.new("L", (80, 30))
    expected = Image.new("L", actual.size)
    ImageDraw.Draw(actual).text((0, 0), text, font=font, direction=direction, fill=255)
    draw = ImageDraw.Draw(expected)
    for i, char in enumerate(visual):
        draw.text((i * 12, 0), char, font=font, fill=255)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("anchor", ["la", "lt", "lm", "ls", "lb", "ld", "ma", "mm", "rs", "rd"])
def test_horizontal_anchors(font: ImageFont.TrueTypeFont, anchor: str) -> None:
    text = "AV"
    box = font.getbbox(text, anchor=anchor)
    mask, offset = font.getmask2(text, anchor=anchor)
    assert offset == box[:2]
    assert mask.size == (box[2] - box[0], box[3] - box[1])
    if anchor[1] == "t":
        assert box[1] == 0
    if anchor[1] == "b":
        assert box[3] == 0
    if anchor[1] == "s":
        assert box[1] < 0 and box[3] == 0


@pytest.mark.parametrize("anchor", ["lt", "mt", "rt", "st", "lm", "mb", "rb"])
def test_vertical_anchors(font: ImageFont.TrueTypeFont, anchor: str) -> None:
    box = font.getbbox("AV", anchor=anchor, direction="ttb")
    mask, offset = font.getmask2("AV", anchor=anchor, direction="ttb")
    assert offset == box[:2]
    assert mask.size == (box[2] - box[0], box[3] - box[1])
    assert mask.size[1] > mask.size[0]


@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA", "CMYK", "HSV", "YCbCr", "LAB", "I", "I;16", "I;16L", "I;16B", "F", "P", "PA"])
def test_native_font_drawing_modes(font: ImageFont.TrueTypeFont, mode: str) -> None:
    image = Image.new(mode, (80, 60))
    draw = ImageDraw.Draw(image)
    draw.fontmode = "1"
    draw.text((3, 2), "AV", font=font, fill=1 if mode in ("1", "I", "I;16", "I;16L", "I;16B", "F", "P", "PA") else "white", stroke_width=1)
    assert any(image.tobytes())


def test_antialiasing_stroke_fractional_origin_and_multiline(font: ImageFont.TrueTypeFont) -> None:
    mask, offset = font.getmask2("AV", start=(0.5, 0.75))
    assert any(0 < value < 255 for value in bytes(mask))
    binary = font.getmask("AV", mode="1")
    assert set(bytes(binary)) <= {0, 255}
    stroked, stroked_offset = font.getmask2("AV", start=(0.5, 0.75), stroke_width=2)
    assert stroked.size == (mask.size[0] + 4, mask.size[1] + 4)
    assert stroked_offset == (offset[0] - 2, offset[1] - 2)
    assert sum(bytes(stroked)) > sum(bytes(mask))
    for align in ("left", "center", "right"):
        image = Image.new("L", (100, 100))
        draw = ImageDraw.Draw(image)
        draw.font = font
        draw.multiline_text((50.5, 40.75), "AV\nA", anchor="mm", align=align, fill=100, stroke_width=2, stroke_fill=255)
        box = draw.multiline_textbbox((50.5, 40.75), "AV\nA", anchor="mm", align=align, stroke_width=2)
        ink = image.getbbox()
        assert ink is not None
        assert floor(box[0]) <= ink[0] < ink[2] <= ceil(box[2]) + 1
        assert floor(box[1]) <= ink[1] < ink[3] <= ceil(box[3]) + 1


def test_default_font_public_api_and_no_pillow_import() -> None:
    import subprocess
    import sys

    font = ImageFont.load_default()
    assert isinstance(font, ImageFont.ImageFont)
    assert font.getmetrics() == (7, 0)
    assert font.getlength("Hello") == 30
    assert font.getbbox("Hello") == (0, 0, 30, 7)
    assert ImageFont.load_default(14).getmetrics() == (14, 0)
    image = Image.new("L", (60, 20))
    ImageDraw.Draw(image).text((0, 0), "Hello", font=font, fill=255)
    implicit = Image.new("L", image.size)
    ImageDraw.Draw(implicit).text((0, 0), "Hello", fill=255)
    assert image.tobytes() == implicit.tobytes()
    subprocess.run([sys.executable, "-c", "import sys; from blanket import ImageFont; ImageFont.load_default().getmask('abc'); assert 'PIL' not in sys.modules"], check=True)


@pytest.mark.parametrize("text", ["", " "])
def test_empty_and_whitespace(font: ImageFont.TrueTypeFont, text: str) -> None:
    for direction in (None, "ttb"):
        mask, offset = font.getmask2(text, direction=direction, stroke_width=2)
        box = font.getbbox(text, direction=direction, stroke_width=2)
        assert offset == box[:2]
        assert mask.size == (box[2] - box[0], box[3] - box[1])
        assert not any(bytes(mask))
        image = Image.new("L", (20, 20))
        ImageDraw.Draw(image).text((0, 0), text, font=font, fill=255, direction=direction, stroke_width=2)
        assert not any(image.tobytes())


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), 1_000_001])
def test_invalid_sizes(font_data: bytes, size: float) -> None:
    with pytest.raises(ValueError):
        ImageFont.truetype(BytesIO(font_data), size)


def test_invalid_options_and_input(font: ImageFont.TrueTypeFont, font_data: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(OSError):
        ImageFont.truetype(BytesIO(b"invalid"))
    with pytest.raises(OSError):
        ImageFont.truetype(BytesIO(font_data), index=1)
    for kwargs in ({"direction": "bad"}, {"features": ["invalid-feature-name"]}, {"anchor": "zz"}, {"stroke_width": -1}, {"stroke_width": float("nan")}):
        with pytest.raises(ValueError):
            font.getbbox("A", **kwargs)
    with pytest.raises(ValueError):
        font.getbbox("A", anchor="la", direction="ttb")
    with pytest.raises(ValueError):
        font.getmask("A", mode="RGBA")
    with pytest.raises(ValueError):
        font.getmask2("A", start=(float("nan"), 0))
    with pytest.raises(ValueError):
        font.getlength("A\nV")
    with pytest.raises(ValueError):
        ImageFont.truetype(BytesIO(font_data), encoding="symb")
    with pytest.raises(ValueError):
        ImageFont.truetype(BytesIO(font_data), index=-1)
    monkeypatch.setattr(ImageFont, "MAX_STRING_LENGTH", 2)
    with pytest.raises(ValueError):
        font.getlength("AVA")
    assert font.getlength(b"AV") == font.getlength("AV")


def test_font_shared_between_threads(font: ImageFont.TrueTypeFont) -> None:
    texts = ["AV", "fi", "éΩЖ", "אב"]

    def render(text: str) -> bytes:
        return bytes(font.getmask(text))

    expected = [render(text) for text in texts]
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(render, texts)) == expected


@pytest.mark.parametrize("text", ["AVfi" * 20, "AאבV ΩЖ", "سلام", "", "   "])
@pytest.mark.parametrize("direction", [None, "ltr", "rtl", "ttb"])
def test_advance_only_matches_full_layout(font: ImageFont.TrueTypeFont, text: str, direction: str | None) -> None:
    for features in ([], ["-kern", "-liga"]):
        advance, _, _ = font._layout(text, direction, features, None, True)
        assert font.getlength(text, direction=direction, features=features) == advance


@pytest.mark.parametrize("suffix", [".png", ".gif", ".pbm"])
def test_load_bitmap_font(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / "font.pil"
    glyphs = [(0,) * 10 for _ in range(256)]
    glyphs[ord("A")] = (4, 0, 0, 0, 3, 2, 0, 0, 3, 2)
    glyphs[ord(" ")] = (2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    path.write_bytes(b"PILfont\n; test font\nDATA\n" + b"".join(struct.pack(">10h", *glyph) for glyph in glyphs))
    atlas = PILImage.frombytes("1", (3, 2), bytes([0xA0, 0xE0]))
    atlas.save(path.with_suffix(suffix))
    font = ImageFont.load(path)
    reference = PILFont.load(path)
    assert font.getlength("A A") == reference.getlength("A A")
    assert font.getbbox("A A") == reference.getbbox("A A")
    assert bytes(font.getmask("A A")) == bytes(reference.getmask("A A"))
    image = Image.new("L", (20, 10))
    ImageDraw.Draw(image).text((1, 1), "A A", font=font, fill=255)
    assert any(image.tobytes())
    with pytest.raises(UnicodeEncodeError):
        font.getlength("Ω")


def test_corrupt_bitmap_font(tmp_path: Path) -> None:
    path = tmp_path / "font.pil"
    for data in (b"invalid", b"PILfont\n", b"PILfont\nDATA\n123"):
        path.write_bytes(data)
        with pytest.raises(OSError):
            ImageFont.load(path)


@pytest.mark.parametrize("bounds", [(0, 0, 3, 2), (-1, -1, 2, 1), (1, 2, 4, 4)])
def test_pillow_compiled_bitmap_font(tmp_path: Path, bounds: tuple[int, int, int, int]) -> None:
    path = tmp_path / "compiled.pil"
    compiled = FontFile()
    atlas = PILImage.frombytes("1", (3, 2), bytes([0xA0, 0xE0]))
    compiled.glyph[ord("A")] = ((4, 0), bounds, (0, 0, 3, 2), atlas)
    compiled.save(path)
    font = ImageFont.load(path)
    reference = PILFont.load(path)
    assert font.getbbox("AA") == reference.getbbox("AA")
    assert bytes(font.getmask("AA")) == bytes(reference.getmask("AA"))
