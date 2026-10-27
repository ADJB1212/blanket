"""Metadata retention without pixel processing."""

from __future__ import annotations

__lazy_modules__ = {"collections.abc"}

from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    from collections.abc import Mapping

from ._ifd import Exif

_POINTERS = (34665, 34853, 40965)
_LOCATION = frozenset({"location", "locationcreated", "locationshown", "sublocation", "city", "state", "province", "country", "countrycode", "countryname", "locationdetails"})
_XMP_ALIASES = ("xmp", "XML:com.adobe.xmp")
_EXIF_ALIASES = ("exif", "Raw profile type exif")


def _location(name: str) -> bool:
    local = name.rsplit("}", 1)[-1].casefold()
    return local.startswith("gps") or local in _LOCATION


def _filtered_exif(raw: bytes, policy: MetadataPolicy) -> bytes:
    source = Exif()
    source.load(raw)
    result = Exif()
    result.endian = source.endian

    def copy(values: Mapping[int, object], key: int, depth: int = 0) -> dict[int, object]:
        if depth > 16:
            raise ValueError("EXIF IFD nesting exceeds 16")
        output: dict[int, object] = {}
        for tag, value in values.items():
            if tag == 330 or (not policy.location and tag in (34853, 37500)):
                continue
            if tag in _POINTERS:
                nested = copy(source.get_ifd(tag), tag, depth + 1)
                if nested:
                    output[tag] = nested
            elif policy.exif_tags is None or tag in policy.exif_tags:
                output[tag] = value
        result._types[key] = source._types.get(key, {}).copy()
        return output

    result.update(copy(source, 0))
    if policy.previews and policy.location and policy.exif_tags is None:
        result.thumbnail = source.thumbnail
        result._ifds[1] = {k: v for k, v in source.get_ifd(1).items() if k not in (513, 514)}
        result._types[1] = source._types.get(1, {}).copy()
    return result.tobytes() if result or result.thumbnail is not None else b""


def _filtered_xmp(raw: bytes, policy: MetadataPolicy) -> bytes:
    if len(raw) > 16 * 1024 * 1024:
        return b""
    try:
        text = raw.decode("utf-8")
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            return b""
        root = ET.fromstring(text)  # noqa: S314 - DTDs and entities are rejected above.
    except (UnicodeDecodeError, ET.ParseError):
        return b""
    structural = {"adobe:ns:meta/", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}

    def keep(name: str) -> bool:
        if not policy.location and _location(name):
            return False
        namespace = name[1:].split("}", 1)[0] if name.startswith("{") else ""
        return policy.xmp_fields is None or namespace in structural or name in policy.xmp_fields

    for element in root.iter():
        for name in list(element.attrib):
            if not keep(name):
                del element.attrib[name]
        for child in list(element):
            if not keep(child.tag):
                element.remove(child)
    return ET.tostring(root, encoding="utf-8")


@dataclass(frozen=True)
class MetadataPolicy:
    """Select metadata categories and optional EXIF, XMP, and PNG text fields.

    All categories are retained by default. Allowlists use numeric EXIF tags,
    expanded XML names (``{namespace}field``), and PNG text keywords.
    ``location=False`` removes the GPS IFD, MakerNote, EXIF thumbnail, and known
    XMP/text location fields. Opaque/custom data is not a privacy guarantee.
    Invalid XMP is discarded when filtering is requested.

    Examples:
        ```python
        from blanket import ImageMetadata

        policy = ImageMetadata.MetadataPolicy(location=False, text=False)
        ```
    """

    exif: bool = True
    xmp: bool = True
    text: bool = True
    location: bool = True
    icc_profile: bool = True
    previews: bool = True
    exif_tags: frozenset[int] | None = None
    xmp_fields: frozenset[str] | None = None
    text_keys: frozenset[str] | None = None

    def __post_init__(self) -> None:
        for name in ("exif", "xmp", "text", "location", "icc_profile", "previews"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name, kind in (("exif_tags", int), ("xmp_fields", str), ("text_keys", str)):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, (str, bytes)):
                    raise TypeError(f"{name} must be a collection of fields")
                fields = frozenset(value)
                if any(isinstance(v, bool) or not isinstance(v, kind) for v in fields):
                    raise TypeError(f"invalid {name} field")
                if name == "exif_tags" and any(not 0 <= v <= 65535 for v in fields):
                    raise ValueError("EXIF tags must be between 0 and 65535")
                object.__setattr__(self, name, fields)

    def apply(self, info: Mapping[object, object]) -> dict[object, object]:
        """Return filtered metadata without changing the source mapping.

        Examples:
            ```python
            from blanket import ImageMetadata

            result = ImageMetadata.MetadataPolicy(icc_profile=False).apply({"icc_profile": b"profile"})
            ```
        """
        result = dict(info)
        for key in _EXIF_ALIASES:
            raw = result.get(key)
            if not self.exif:
                result.pop(key, None)
            elif raw is not None and (self.exif_tags is not None or not self.location or not self.previews):
                if key != "exif":
                    result.pop(key, None)
                    continue
                if not isinstance(raw, bytes):
                    raise TypeError("exif must be bytes")
                result[key] = _filtered_exif(raw, self)
        for key in _XMP_ALIASES:
            raw = result.get(key)
            if not self.xmp:
                result.pop(key, None)
            elif raw is not None and (self.xmp_fields is not None or not self.location):
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                if not isinstance(raw, bytes):
                    raise TypeError("xmp must be bytes or text")
                result[key] = _filtered_xmp(raw, self)
        if not self.icc_profile:
            result.pop("icc_profile", None)
        for key, value in list(result.items()):
            if key in (*_XMP_ALIASES, *_EXIF_ALIASES, "icc_profile"):
                continue
            if (
                isinstance(key, str)
                and (isinstance(value, str) or key == "comment")
                and (not self.text or (self.text_keys is not None and key not in self.text_keys) or (not self.location and _location(key)))
            ):
                del result[key]
        return result
