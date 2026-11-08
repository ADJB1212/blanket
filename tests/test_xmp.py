from __future__ import annotations

from xml.etree.ElementTree import ParseError

import pytest
from helpers import assert_raises

from blanket import Image


@pytest.mark.parametrize("strip", [False, True])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"<root/>", None),
        (b'<root a="1">hello &amp; world</root>', {"a": "1", "text": "hello & world"}),
        (b'<root><item>one</item><item/><item a="2">three</item></root>', {"item": ["one", None, {"a": "2", "text": "three"}]}),
        (b'<root item="attribute"><item>child</item><item>again</item></root>', {"item": ["attribute", "child", "again"]}),
        (b"<root>ignored<item>value</item>tail</root>", {"item": "value"}),
        (b"<root>before<!-- comment -->after</root>", "beforeafter"),
        (b'<?xpacket begin=""?><root><![CDATA[<text>]]></root><?xpacket end="w"?>\x00 ', "<text>"),
        ("<root>caf\u00e9 &#x1f600;</root>".encode(), "caf\u00e9 \U0001f600"),
    ],
)
def test_getxmp_dictionary(raw: bytes, expected: object, strip: bool) -> None:
    image = Image.new("RGB", (1, 1))
    image.info["xmp"] = raw
    assert image.getxmp(strip_namespaces=strip) == {"root": expected}


@pytest.mark.parametrize("strip", [False, True])
def test_getxmp_namespaces(strip: bool) -> None:
    image = Image.new("RGB", (1, 1))
    image.info["xmp"] = b'<x:root xmlns:x="urn:x" xmlns:y="urn:y" x:a="1"><y:item>text</y:item></x:root>'
    expected = {"root": {"a": "1", "item": "text"}} if strip else {"{urn:x}root": {"{urn:x}a": "1", "{urn:y}item": "text"}}
    assert image.getxmp(strip_namespaces=strip) == expected


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"<root>",
        b"<root/><extra/>",
        b"<root>&unknown;</root>",
        b'<!DOCTYPE root [<!ENTITY secret SYSTEM "file:///etc/passwd">]><root>&secret;</root>',
        b'<!DOCTYPE root [<!ENTITY expanded "value">]><root>&expanded;</root>',
        b"<root>\xff</root>",
        b"<a>" * 258 + b"</a>" * 258,
    ],
)
def test_getxmp_rejects_invalid_or_unsafe_xml(raw: bytes) -> None:
    image = Image.new("RGB", (1, 1))
    image.info["xmp"] = raw
    assert_raises(ParseError, image.getxmp)


def test_getxmp_missing_metadata() -> None:
    assert Image.new("RGB", (1, 1)).getxmp() == {}


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-16-be", "utf-16-le"])
def test_getxmp_unicode_encoding(encoding: str) -> None:
    image = Image.new("RGB", (1, 1))
    image.info["xmp"] = "<root>caf\u00e9</root>".encode(encoding)
    assert image.getxmp() == {"root": "caf\u00e9"}


def test_getxmp_padded_packet() -> None:
    image = Image.new("RGB", (1, 1))
    image.info["xmp"] = b'prefix<x:xmpmeta xmlns:x="urn:x"><tag>value</tag></x:xmpmeta>suffix\x00 '
    assert image.getxmp() == {"xmpmeta": {"tag": "value"}}
