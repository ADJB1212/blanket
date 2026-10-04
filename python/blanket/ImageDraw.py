"""Draw points and rectangles in place using a Pillow-shaped API.

Create a context with :func:`Draw`. Points and rectangles support color inks,
clipping, and RGBA blending onto RGB images. Other drawing primitives and
text operations currently raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

from . import Image
from ._blanket import draw_points, draw_rectangle, draw_unimplemented
from ._color import color_pixel

if TYPE_CHECKING:
    from collections.abc import Sequence

    Coords = Sequence[float] | Sequence[tuple[float, float]]
    Color = int | float | str | tuple[int, ...] | None

__all__ = ["Draw", "ImageDraw", "floodfill"]


class ImageDraw:
    """In-place drawing context for a Blanket image.

    Drawing changes the supplied image directly. Coordinates use an origin
    at the top-left corner; fractional values are truncated toward zero.
    Pixels outside the image are ignored, including negative coordinates.

    Inks accept color names, packed integers, or channel tuples in the drawing
    mode. Indexed images require numeric indices rather than palette color
    allocation. Integer and float images accept numeric sample values.
    Multichannel 16-bit images must be converted to 8-bit before drawing.

    An RGBA context on an RGB image blends ink alpha into existing pixels.
    Drawing on an RGBA image replaces all channels, including alpha.
    """

    def __init__(self, im: Image.Image, mode: str | None = None) -> None:
        """Configure a drawing context for `im`.

        Args:
            im: Image to draw on. Pixels are modified in place.
            mode: Color mode for ink values. Defaults to the image mode. Use `RGBA` on an `RGB` image for alpha blending.

        Raises:
            TypeError: If `im` is not a Blanket image.
            ValueError: If the image is closed or the drawing mode does not
                match the image mode, except for `RGBA` on `RGB`.

        Examples:
            ```python
            from blanket import Image, ImageDraw

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            draw = ImageDraw.Draw(image)
            print(draw.mode)
            ```
        """
        if not isinstance(im, Image.Image):
            raise TypeError("image must be a Blanket Image")
        im.load()
        if mode is None:
            mode = im.mode
        blend = 0
        if mode != im.mode:
            if im.mode != "RGB" or mode != "RGBA":
                raise ValueError("mode mismatch")
            blend = 1
        self.im = im
        self.mode = mode
        self.blend = blend
        self.fontmode = "1" if mode in ("1", "P") else "L"
        self.font = None

    def _ink(self, fill: Color) -> int | float | tuple[int, ...]:
        """Normalize an ink to native pixel values in the drawing mode."""
        if fill is None:
            fill = 1 if self.mode in ("I", "F") else (0 if self.mode in ("1", "L", "P") else -1)
        if self.mode in ("I", "F", "I;16", "I;16L", "I;16B", "P"):
            if isinstance(fill, str):
                raise ValueError("drawing in this mode requires a numeric ink")
            return fill
        pixel = color_pixel(fill, self.mode)
        return pixel[0] if len(pixel) == 1 else tuple(pixel)

    @staticmethod
    def _coords(xy: Coords) -> list[tuple[int, int]]:
        """Normalize flat or paired coordinates, truncating toward zero."""
        values = list(xy)
        if not values:
            return []
        if isinstance(values[0], (tuple, list)):
            return [(int(x), int(y)) for x, y in values]
        if len(values) % 2:
            raise ValueError("wrong number of coordinates")
        return [(int(values[i]), int(values[i + 1])) for i in range(0, len(values), 2)]

    def arc(self, xy: Coords, start: float, end: float, fill: Color = None, width: int = 1) -> None:
        """Draw an arc inside a bounding box. Not implemented."""
        draw_unimplemented("arc")

    def chord(self, xy: Coords, start: float, end: float, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw a chord inside a bounding box. Not implemented."""
        draw_unimplemented("chord")

    def ellipse(self, xy: Coords, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw an ellipse inside a bounding box. Not implemented."""
        draw_unimplemented("ellipse")

    def line(self, xy: Coords, fill: Color = None, width: int = 0, joint: str | None = None) -> None:
        """Draw a polyline. Not implemented."""
        draw_unimplemented("line")

    def pieslice(self, xy: Coords, start: float, end: float, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw a pieslice inside a bounding box. Not implemented."""
        draw_unimplemented("pieslice")

    def point(self, xy: Coords, fill: Color = None) -> None:
        """Draw one or more points in place.

        Coordinates outside the image are ignored. Repeated coordinates draw
        repeatedly, including repeated alpha blending in an RGBA context on
        an RGB image. An empty coordinate sequence draws nothing.

        Args:
            xy: A single `(x, y)` pair, a flat sequence of coordinate values,
                or a sequence of `(x, y)` pairs. Fractional values are
                truncated toward zero.
            fill: Ink color or sample value. `None` uses the default ink:
                zero for `1`, `L`, and `P`; one for `I` and `F`; maximum
                channel values for the remaining modes.

        Raises:
            TypeError: If coordinates or the ink have an incompatible type.
            ValueError: If coordinates are malformed, the image is closed,
                the ink is invalid, or the sample depth is unsupported.

        Examples:
            ```python
            from blanket import Image, ImageDraw

            image = Image.new("RGB", (8, 8), "white")
            draw = ImageDraw.Draw(image)
            draw.point((2, 3), fill="red")
            draw.point([(1, 1), (6, 6)], fill=(0, 0, 255))
            ```

        Notes:
            For Pillow compatibility, points on 16-bit integer images write
            samples in native byte order, even in explicit-endian modes.
        """
        draw_points(self.im._native, self._coords(xy), self._ink(fill), bool(self.blend))

    def polygon(self, xy: Coords, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw a polygon. Not implemented."""
        draw_unimplemented("polygon")

    def regular_polygon(
        self, bounding_circle: tuple[float, float, float] | tuple[tuple[float, float], float], n_sides: int, rotation: float = 0, fill: Color = None, outline: Color = None, width: int = 1
    ) -> None:
        """Draw a regular polygon. Not implemented."""
        draw_unimplemented("regular_polygon")

    def rectangle(self, xy: Coords, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw a rectangle with optional fill and outline in place.

        Both bounding endpoints are included. The fill is drawn first and
        the outline afterward. Pixels outside the image are ignored. If both
        colors are `None`, nothing is drawn.

        Args:
            xy: Bounds as `(x0, y0, x1, y1)` or `[(x0, y0), (x1, y1)]`,
                ordered from top-left to bottom-right. Fractional values are
                truncated toward zero before checking the order.
            fill: Interior ink color or sample value. `None` skips the fill.
            outline: Border ink color or sample value. `None` skips the
                outline. An outline equal to the fill is skipped.
            width: Outline thickness in pixels. Values at or below zero
                skip the outline.

        Raises:
            TypeError: If coordinates, inks, or `width` have incompatible
                types.
            ValueError: If bounds are malformed or reversed, the image is
                closed, an ink is invalid, or the sample depth is unsupported.

        Examples:
            ```python
            from blanket import Image, ImageDraw

            image = Image.new("RGB", (16, 16), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((2, 2, 13, 13), fill="blue", outline="black", width=2)
            ImageDraw.Draw(image, "RGBA").rectangle((4, 4, 11, 11), fill=(255, 0, 0, 128))
            ```

        Notes:
            Outlines normally extend inward. Wide outlines and degenerate
            bounds follow Pillow's overlapping edge behavior and may extend
            beyond the bounds. Rectangles on 16-bit integer images respect
            the mode's byte order.
        """
        points = self._coords(xy)
        if len(points) != 2:
            raise ValueError("rectangle requires exactly two coordinates")
        (x0, y0), (x1, y1) = points
        if x1 < x0 or y1 < y0:
            raise ValueError("rectangle coordinates must be ordered")
        width = index(width)
        bounds = (x0, y0, x1, y1)
        if fill is not None:
            draw_rectangle(self.im._native, bounds, self._ink(fill), 0, bool(self.blend))
        if outline is not None and outline != fill and width > 0:
            draw_rectangle(self.im._native, bounds, self._ink(outline), width, bool(self.blend))

    def rounded_rectangle(self, xy: Coords, radius: float = 0, fill: Color = None, outline: Color = None, width: int = 1, corners: tuple[bool, bool, bool, bool] | None = None) -> None:
        """Draw a rounded rectangle. Not implemented."""
        draw_unimplemented("rounded_rectangle")

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        fill: Color = None,
        font: object | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: Sequence[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        stroke_fill: Color = None,
        embedded_color: bool = False,
        font_size: float | None = None,
    ) -> None:
        """Draw text. Not implemented; requires ImageFont."""
        draw_unimplemented("text")

    def multiline_text(
        self,
        xy: tuple[float, float],
        text: str,
        fill: Color = None,
        font: object | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: Sequence[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        stroke_fill: Color = None,
        embedded_color: bool = False,
        font_size: float | None = None,
    ) -> None:
        """Draw multiline text. Not implemented; requires ImageFont."""
        draw_unimplemented("multiline_text")

    def textlength(
        self,
        text: str,
        font: object | None = None,
        direction: str | None = None,
        features: Sequence[str] | None = None,
        language: str | None = None,
        embedded_color: bool = False,
        font_size: float | None = None,
    ) -> float:
        """Return the advance length of `text`. Not implemented."""
        draw_unimplemented("textlength")
        return 0.0

    def textbbox(
        self,
        xy: tuple[float, float],
        text: str,
        font: object | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: Sequence[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        embedded_color: bool = False,
        font_size: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Return the bounding box of `text`. Not implemented."""
        draw_unimplemented("textbbox")
        return (0.0, 0.0, 0.0, 0.0)

    def multiline_textbbox(
        self,
        xy: tuple[float, float],
        text: str,
        font: object | None = None,
        anchor: str | None = None,
        spacing: float = 4,
        align: str = "left",
        direction: str | None = None,
        features: Sequence[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        embedded_color: bool = False,
        font_size: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Return the bounding box of multiline `text`. Not implemented."""
        draw_unimplemented("multiline_textbbox")
        return (0.0, 0.0, 0.0, 0.0)


def Draw(im: Image.Image, mode: str | None = None) -> ImageDraw:
    """Create a drawing context for `im`.

    Args:
        im: Image to draw on.
        mode: Optional ink mode. Defaults to the image mode.

    Examples:
        ```python
        from blanket import Image, ImageDraw

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        draw = ImageDraw.Draw(image)
        ```
    """
    return ImageDraw(im, mode)


def floodfill(image: Image.Image, xy: tuple[float, float], value: Color, border: Color = None, thresh: float = 0) -> None:
    """Fill connected pixels starting at `xy`. Not implemented.

    Args:
        image: Image to fill.
        xy: Seed coordinate.
        value: Fill color.
        border: Optional border color that stops the fill.
        thresh: Maximum difference from the seed color while filling.

    Examples:
        ```python
        from blanket import Image, ImageDraw

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        ImageDraw.floodfill(image, (0, 0), (255, 0, 0))
        ```
    """
    draw_unimplemented("floodfill")
