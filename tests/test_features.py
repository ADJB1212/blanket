from __future__ import annotations

from collections.abc import Callable
from io import BytesIO, StringIO

import pytest
from PIL import features as pillow_features

from blanket import Image, __version__, features


@pytest.mark.parametrize("feature", ["blanket", "png", "jpg", "jxl", "libtiff", "webp", "avif", "libheif", "bmp", "gif", "ico", "pdf", "littlecms2", "libjpeg_turbo"])
def test_available_features(feature: str) -> None:
    assert features.check(feature) is True
    assert feature in features.get_supported()
    if feature != "pdf":
        assert isinstance(features.version(feature), str)
        assert features.version(feature)


@pytest.mark.parametrize(("alias", "name"), [("jpeg", "jpg"), ("jpegxl", "jxl"), ("tiff", "libtiff"), ("heic", "heif")])
def test_aliases(alias: str, name: str) -> None:
    assert features.check(alias) == features.check(name)
    assert features.version(alias) == features.version(name)
    assert alias not in features.get_supported()


def test_package_and_backend_versions() -> None:
    assert features.version("blanket") == __version__
    assert features.version("pdf") is None
    assert features.version("jpg") == features.version("libjpeg_turbo")
    for name in ("jxl", "webp", "libheif", "littlecms2"):
        version = features.version(name)
        assert version is not None and all(part.isdecimal() for part in version.split("."))
    for name, prefix in (("png", "png "), ("jpg", "turbojpeg "), ("libtiff", "tiff "), ("gif", "gif "), ("bmp", "image "), ("avif", "ravif ")):
        assert features.version(name).startswith(prefix)


@pytest.mark.parametrize("feature", ["pil", "tkinter", "freetype2", "jpg_2000", "zlib", "raqm", "fribidi", "harfbuzz", "mozjpeg", "zlib_ng", "libimagequant", "xcb"])
def test_unavailable_pillow_features(feature: str) -> None:
    assert features.check(feature) is False
    assert features.version(feature) is None


def test_unknown_features_match_pillow() -> None:
    for module in (features, pillow_features):
        with pytest.warns(UserWarning, match="Unknown feature 'not_a_feature'"):
            assert module.check("not_a_feature") is False
        assert module.version("not_a_feature") is None


@pytest.mark.parametrize("value", [None, 1, [], {}])
@pytest.mark.parametrize("function", [features.check, features.version])
def test_feature_name_requires_string(function: Callable, value: object) -> None:
    with pytest.raises(TypeError, match="feature must be a string"):
        function(value)


@pytest.mark.parametrize(
    ("feature", "format"), [("png", "PNG"), ("jpg", "JPEG"), ("jxl", "JXL"), ("libtiff", "TIFF"), ("webp", "WEBP"), ("avif", "AVIF"), ("bmp", "BMP"), ("gif", "GIF"), ("ico", "ICO")]
)
def test_reported_codecs_can_save_and_open(feature: str, format: str) -> None:
    assert features.check(feature)
    stream = BytesIO()
    Image.new("RGB", (32, 32), (60, 120, 180)).save(stream, format)
    assert Image.open(BytesIO(stream.getvalue())).size == (32, 32)


def test_heif_runtime_capabilities() -> None:
    decode, encode = features.check("heif_decoder"), features.check("heif_encoder")
    assert features.check("heif") == (decode and encode)
    for feature, available in (("heif_decoder", decode), ("heif_encoder", encode), ("heif", decode and encode)):
        assert features.version(feature) == (features.version("libheif") if available else None)
    if encode:
        stream = BytesIO()
        Image.new("RGB", (32, 32), (60, 120, 180)).save(stream, "HEIF")
        if decode:
            assert Image.open(BytesIO(stream.getvalue())).size == (32, 32)


def test_blanketinfo_stream_and_default_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    report = StringIO()
    assert features.blanketinfo(report) is None
    output = report.getvalue()
    assert f"Blanket {__version__}" in output
    assert "Python " in output and "Platform " in output
    assert "png: available (png " in output
    assert "PNG: open, save" in output
    assert "PDF: save" in output
    assert "PDF: open" not in output
    features.blanketinfo(supported_formats=False)
    output = capsys.readouterr().out
    assert "Backend support:" in output
    assert "Formats " not in output


@pytest.mark.parametrize(("decode", "encode"), [(False, False), (True, False), (False, True), (True, True)])
def test_partial_heif_plugin_report(monkeypatch: pytest.MonkeyPatch, decode: bool, encode: bool) -> None:
    original = features._feature_info

    def info(feature: str) -> tuple[bool, str | None] | None:
        if feature in ("heif", "heif_decoder", "heif_encoder"):
            available = {"heif": decode and encode, "heif_decoder": decode, "heif_encoder": encode}[feature]
            return available, "1.2.3" if available else None
        return original(feature)

    monkeypatch.setattr(features, "_feature_info", info)
    assert features.check("heif") == (decode and encode)
    assert ("heif_decoder" in features.get_supported()) == decode
    assert ("heif_encoder" in features.get_supported()) == encode
    report = StringIO()
    features.blanketinfo(report)
    expected = ", ".join(name for name, available in (("open", decode), ("save", encode)) if available) or "unavailable"
    assert f"HEIF/HEIC: {expected}\n" in report.getvalue()
