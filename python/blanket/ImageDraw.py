"""Draw geometry, text, and connected fills directly on Blanket images.

Use :func:`Draw` to create a context. Geometric operations support clipping,
mode-specific inks, and optional supersampled coverage. Text uses a small
bitmap fallback or an explicitly supplied font object; Pillow is not required.
"""

from __future__ import annotations

from math import ceil, cos, isfinite, pi, sin
from operator import index
from typing import TYPE_CHECKING

from . import Image
from ._blanket import draw_ellipse, draw_floodfill, draw_mask, draw_path, draw_points, draw_rectangle
from ._color import color_pixel
from ._draw_font import BitmapFont

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
    Geometric drawing on an RGBA image replaces all channels, including alpha.
    Text masks apply coverage to existing pixels. Set `antialias=True` for
    supersampled geometric edges on 8-bit non-indexed images. Curved edges
    and wide strokes can differ from Pillow's rasterization.

    Attributes:
        im: Image modified by this context.
        mode: Mode used to interpret ink values.
        blend: Whether RGBA inks blend onto an RGB image.
        antialias: Whether geometric edges use supersampled coverage.
        font: Default font object for text, or `None` for the bitmap fallback.
        fontmode: Font mask mode, initially `1` for bilevel and indexed images
            and `L` otherwise. Set to `1` to request binary font masks.
    """

    def __init__(self, im: Image.Image, mode: str | None = None, *, antialias: bool = False) -> None:
        """Configure a drawing context for `im`.

        Args:
            im: Image to draw on. Pixels are modified in place.
            mode: Color mode for ink values. Defaults to the image mode. Use `RGBA` on an `RGB` image for alpha blending.
            antialias: Use supersampled coverage for geometric edges.

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
        self.antialias = antialias

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
        """Draw the curved edge of an ellipse between two angles.

        Angles run clockwise from the positive x-axis. A sweep of at least
        360 degrees draws a full arc; equal angles draw nothing. Negative
        sweeps wrap clockwise through zero. The arc has no radial edges.

        Args:
            xy: Ellipse bounds as `(x0, y0, x1, y1)` or two coordinate pairs,
                ordered from top-left to bottom-right.
            start: Starting angle in degrees; must be finite.
            end: Ending angle in degrees; must be finite.
            fill: Stroke ink. `None` uses the context's default ink.
            width: Stroke thickness in pixels. Nonpositive values draw nothing.

        Raises:
            TypeError: If coordinates, ink, or width have incompatible types.
            ValueError: If bounds or angles are invalid, the image is closed,
                or the ink, sample depth, or anti-aliasing mode is unsupported.
        """
        self._path(self._curve(xy, start, end), fill, width=index(width))

    def chord(self, xy: Coords, start: float, end: float, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw an elliptical segment closed by a straight chord.

        The fill is drawn before the outline. The outline includes both the
        curved edge and the straight line joining its endpoints.

        Args:
            xy: Ordered ellipse bounds as four values or two coordinate pairs.
            start: Starting angle in degrees, clockwise from the positive
                x-axis. Must be finite.
            end: Ending angle in degrees. Negative sweeps wrap clockwise;
                sweeps of at least 360 degrees cover the full ellipse.
            fill: Interior ink, or `None` to skip the fill.
            outline: Border ink, or `None` to skip the outline. An outline
                equal to the fill is skipped.
            width: Border thickness in pixels. Nonpositive values skip it.

        Raises:
            TypeError: If coordinates, inks, or width have incompatible types.
            ValueError: If bounds or angles are invalid, the image is closed,
                or the ink, sample depth, or anti-aliasing mode is unsupported.
        """
        self._shape(self._curve(xy, start, end), fill, outline, width)

    def ellipse(self, xy: Coords, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw an ellipse inside an inclusive bounding box.

        The fill is drawn first. The outline extends inward and is skipped
        when it equals the fill. If both inks are `None`, nothing is drawn.

        Args:
            xy: Ordered bounds as `(x0, y0, x1, y1)` or two coordinate pairs.
                Fractional coordinates are truncated toward zero.
            fill: Interior ink, or `None` to skip the fill.
            outline: Border ink, or `None` to skip the outline.
            width: Border thickness in pixels. Nonpositive values skip it.

        Raises:
            TypeError: If coordinates, inks, or width have incompatible types.
            ValueError: If bounds are malformed or reversed, the image is
                closed, or the ink, sample depth, or anti-aliasing mode is
                unsupported.

        Examples:
            ```python
            from blanket import Image, ImageDraw

            image = Image.new("RGB", (32, 24), "white")
            ImageDraw.Draw(image).ellipse((2, 2, 29, 21), fill="blue", outline="black", width=2)
            ```
        """
        bounds = self._bounds(xy)
        width = index(width)
        if fill is not None:
            draw_ellipse(self.im._native, bounds, self._ink(fill), 0, bool(self.blend), bool(self.antialias))
        if outline is not None and outline != fill and width > 0:
            draw_ellipse(self.im._native, bounds, self._ink(outline), width, bool(self.blend), bool(self.antialias))

    def line(self, xy: Coords, fill: Color = None, width: int = 0, joint: str | None = None) -> None:
        """Draw connected line segments through a sequence of coordinates.

        Pixels outside the image are ignored. A one-coordinate sequence draws
        a point; an empty sequence draws nothing. Wide strokes use rounded
        caps and joins for both supported `joint` values.

        Args:
            xy: Flat coordinate values or a sequence of `(x, y)` pairs.
                Fractional values are truncated toward zero.
            fill: Stroke ink. `None` uses the context's default ink.
            width: Stroke thickness in pixels. Values at or below one use
                a one-pixel stroke.
            joint: `None` or `"curve"`. Both currently use the same joins.

        Raises:
            TypeError: If coordinates, ink, or width have incompatible types.
            ValueError: If coordinates are malformed, `joint` is unsupported,
                the image is closed, or the ink, sample depth, or anti-aliasing
                mode is unsupported.
        """
        if joint not in (None, "curve"):
            raise ValueError("joint must be None or 'curve'")
        self._path(self._coords(xy), fill, width=max(1, index(width)))

    def pieslice(self, xy: Coords, start: float, end: float, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw an elliptical sector bounded by an arc and two radii.

        The fill is drawn before the outline, which includes the radial
        edges. Equal angles draw nothing.

        Args:
            xy: Ordered ellipse bounds as four values or two coordinate pairs.
            start: Starting angle in degrees, clockwise from the positive
                x-axis. Must be finite.
            end: Ending angle in degrees. Negative sweeps wrap clockwise;
                sweeps of at least 360 degrees cover the full ellipse.
            fill: Interior ink, or `None` to skip the fill.
            outline: Border ink, or `None` to skip the outline. An outline
                equal to the fill is skipped.
            width: Border thickness in pixels. Nonpositive values skip it.

        Raises:
            TypeError: If coordinates, inks, or width have incompatible types.
            ValueError: If bounds or angles are invalid, the image is closed,
                or the ink, sample depth, or anti-aliasing mode is unsupported.
        """
        bounds = self._bounds(xy)
        points = self._curve(xy, start, end)
        if not points:
            return
        if abs(end - start) < 360:
            points.append(((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2))
        self._shape(points, fill, outline, width)

    def _bounds(self, xy: Coords) -> tuple[int, int, int, int]:
        """Return two truncated, ordered bounding endpoints as four integers."""
        points = self._coords(xy)
        if len(points) != 2:
            raise ValueError("bounding box requires exactly two coordinates")
        (x0, y0), (x1, y1) = points
        if x1 < x0 or y1 < y0:
            raise ValueError("bounding coordinates must be ordered")
        return x0, y0, x1, y1

    def _curve(self, xy: Coords, start: float, end: float) -> list[tuple[float, float]]:
        """Sample a clockwise elliptical arc, returning no points for zero sweep."""
        x0, y0, x1, y1 = self._bounds(xy)
        if not isfinite(start) or not isfinite(end):
            raise ValueError("angles must be finite")
        sweep = end - start
        sweep = 360 if sweep >= 360 else sweep % 360
        if sweep == 0:
            return []
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        count = max(1, min(65536, ceil(max(rx, ry) * sweep * pi / 180 * 2)))
        return [(x0 + rx + rx * cos((start + sweep * i / count) * pi / 180), y0 + ry + ry * sin((start + sweep * i / count) * pi / 180)) for i in range(count + 1)]

    def _path(self, points: Sequence[tuple[float, float]], fill: Color, *, width: int = 1, closed: bool = False, filled: bool = False) -> None:
        """Rasterize a path using the context's ink, blend, and coverage settings."""
        style = "fill" if filled else ("outline" if closed else "line")
        draw_path(self.im._native, list(points), self._ink(fill), width, style, bool(self.blend), bool(self.antialias))

    def _shape(self, points: Sequence[tuple[float, float]], fill: Color, outline: Color, width: int) -> None:
        """Fill a closed path, then draw its distinct positive-width outline."""
        width = index(width)
        if fill is not None:
            self._path(points, fill, closed=True, filled=True)
        if outline is not None and outline != fill and width > 0:
            self._path(points, outline, width=width, closed=True)

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
        """Draw a closed polygon using the even-odd fill rule.

        The last vertex connects to the first automatically. The fill is
        drawn before the outline; path outlines are centered on the edges.

        Args:
            xy: Flat coordinate values or at least two `(x, y)` pairs.
                Fractional values are truncated toward zero.
            fill: Interior ink, or `None` to skip the fill.
            outline: Border ink, or `None` to skip the outline. An outline
                equal to the fill is skipped.
            width: Border thickness in pixels. Nonpositive values skip it.

        Raises:
            TypeError: If coordinates, inks, or width have incompatible types.
            ValueError: If fewer than two vertices are supplied, coordinates
                are malformed, the image is closed, or the ink, sample depth,
                or anti-aliasing mode is unsupported.
        """
        points = self._coords(xy)
        if len(points) < 2:
            raise ValueError("polygon requires at least two coordinates")
        self._shape(points, fill, outline, width)

    def regular_polygon(
        self, bounding_circle: tuple[float, float, float] | tuple[tuple[float, float], float], n_sides: int, rotation: float = 0, fill: Color = None, outline: Color = None, width: int = 1
    ) -> None:
        """Draw a regular polygon inscribed in a bounding circle.

        Vertices are rounded to two decimal places before their coordinates
        are truncated for polygon drawing. Rotation increases counterclockwise.

        Args:
            bounding_circle: Center and radius as `(x, y, radius)` or
                `((x, y), radius)`. Values must be finite; radius must be positive.
            n_sides: Integer number of sides, at least three.
            rotation: Counterclockwise rotation in degrees. Zero places the
                first vertex at the bottom-left of the polygon.
            fill: Interior ink, or `None` to skip the fill.
            outline: Border ink, or `None` to skip the outline. An outline
                equal to the fill is skipped.
            width: Border thickness in pixels. Nonpositive values skip it.

        Raises:
            TypeError: If the circle, side count, inks, or width have
                incompatible types.
            ValueError: If the circle, rotation, or side count is invalid,
                the image is closed, or drawing in the image mode is unsupported.
        """
        n_sides = index(n_sides)
        if n_sides < 3:
            raise ValueError("n_sides must be at least 3")
        if len(bounding_circle) == 3:
            x, y, radius = bounding_circle
        else:
            (x, y), radius = bounding_circle
        if radius <= 0 or not all(isfinite(v) for v in (x, y, radius, rotation)):
            raise ValueError("invalid bounding circle or rotation")
        angles = [(270 - 180 / n_sides + rotation + i * 360 / n_sides) * pi / 180 for i in range(n_sides)]
        self.polygon([(round(x + radius * cos(angle), 2), round(y - radius * sin(angle), 2)) for angle in angles], fill, outline, width)

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
        if self.antialias:
            self._shape([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill, outline, width)
            return
        if fill is not None:
            draw_rectangle(self.im._native, bounds, self._ink(fill), 0, bool(self.blend))
        if outline is not None and outline != fill and width > 0:
            draw_rectangle(self.im._native, bounds, self._ink(outline), width, bool(self.blend))

    def rounded_rectangle(self, xy: Coords, radius: float = 0, fill: Color = None, outline: Color = None, width: int = 1, corners: tuple[bool, bool, bool, bool] | None = None) -> None:
        """Draw a rectangle with individually selectable rounded corners.

        The radius is clamped to half the smaller bounding dimension. The
        fill is drawn before the centered path outline.

        Args:
            xy: Ordered bounds as four values or two coordinate pairs.
                Fractional coordinates are truncated toward zero.
            radius: Corner radius in pixels; must be finite and nonnegative.
            fill: Interior ink, or `None` to skip the fill.
            outline: Border ink, or `None` to skip the outline. An outline
                equal to the fill is skipped.
            width: Border thickness in pixels. Nonpositive values skip it.
            corners: Four flags in top-left, top-right, bottom-right,
                bottom-left order. `None` rounds all corners.

        Raises:
            TypeError: If coordinates, radius, inks, or width have incompatible
                types.
            ValueError: If bounds, radius, or corner flags are invalid, the
                image is closed, or drawing in the image mode is unsupported.
        """
        x0, y0, x1, y1 = self._bounds(xy)
        if not isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        radius = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
        corners = (True, True, True, True) if corners is None else corners
        if len(corners) != 4:
            raise ValueError("corners must contain four flags")
        points = []
        for enabled, (cx, cy, start, corner) in zip(
            corners,
            [(x0 + radius, y0 + radius, 180, (x0, y0)), (x1 - radius, y0 + radius, 270, (x1, y0)), (x1 - radius, y1 - radius, 0, (x1, y1)), (x0 + radius, y1 - radius, 90, (x0, y1))],
            strict=True,
        ):
            if enabled and radius:
                count = max(1, min(16384, ceil(radius * pi)))
                points.extend((cx + radius * cos((start + 90 * i / count) * pi / 180), cy + radius * sin((start + 90 * i / count) * pi / 180)) for i in range(count + 1))
            else:
                points.append(corner)
        self._shape(points, fill, outline, width)

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
        """Draw text using the bitmap fallback or a supplied font object.

        Strings containing newlines are passed to :meth:`multiline_text`.
        Font objects must provide `getlength`, `getbbox`, and `getmask2` or
        `getmask`. Coverage masks require 8-bit non-indexed images; use
        `fontmode="1"` for binary masks on other supported drawing modes.

        Args:
            xy: Text anchor position in image coordinates.
            text: String to draw.
            fill: Text ink. `None` uses the context's default ink.
            font: Explicit font object. Otherwise use `self.font`, then the
                built-in fixed-width bitmap fallback.
            anchor: Two-character font anchor, such as `"la"` or `"mm"`.
                `None` uses the font's default. Supported anchors depend on
                the supplied font.
            spacing: Additional pixels between lines when text contains newlines.
            align: Multiline alignment: `"left"`, `"center"`, or `"right"`.
            direction: Text direction passed to the font when specified.
            features: OpenType feature names passed to the font when specified.
            language: Language tag passed to the font when specified.
            stroke_width: Finite, nonnegative stroke thickness, rounded up to
                a whole pixel for rendering.
            stroke_fill: Stroke ink. `None` uses the text ink.
            embedded_color: Embedded-color font request. Currently unsupported.
            font_size: Requested bitmap fallback height. Defaults to seven
                pixels and scales in whole multiples of seven. Ignored when
                an explicit or context font is used.

        Raises:
            TypeError: If coordinates, inks, or the font interface are incompatible.
            ValueError: If stroke width, font size, or anchor is invalid,
                the image is closed, or mask coverage is unsupported for its mode.
            NotImplementedError: If `embedded_color` is true.

        Notes:
            The bitmap fallback uses uppercase glyphs for lowercase letters
            and `?` for unsupported characters. It ignores font shaping options.
            Its appearance and metrics differ from Pillow's default font.

        Examples:
            ```python
            from blanket import Image, ImageDraw

            image = Image.new("RGB", (120, 40), "white")
            draw = ImageDraw.Draw(image)
            draw.text((8, 8), "Blanket", fill="blue", font_size=14)
            ```
        """
        if "\n" in text:
            return self.multiline_text(xy, text, fill, font, anchor, spacing, align, direction, features, language, stroke_width, stroke_fill, embedded_color, font_size)
        font = self._font(font, font_size)
        self._text_options(direction, features, language, embedded_color)
        if not isfinite(stroke_width) or stroke_width < 0:
            raise ValueError("stroke_width must be finite and nonnegative")
        stroke = ceil(stroke_width)
        options = self._font_options(direction, features, language)
        mode = self.fontmode

        def render(ink: Color, outline: int) -> None:
            """Render one text or stroke mask at the requested anchor position."""
            native_stroke = outline if isinstance(font, BitmapFont) else 0
            if hasattr(font, "getmask2"):
                mask, offset = font.getmask2(text, mode, anchor=anchor, stroke_width=outline - native_stroke, start=(xy[0] - int(xy[0]), xy[1] - int(xy[1])), stroke_filled=True, **options)
            else:
                mask = font.getmask(text, mode, stroke_width=outline, **options)
                offset = font.getbbox(text, anchor=anchor, stroke_width=outline, **options)[:2]
            data = bytes(mask)
            if mode == "1":
                data = bytes(255 if sample else 0 for sample in data)
            position = (int(xy[0]) + int(offset[0]), int(xy[1]) + int(offset[1]))
            draw_mask(self.im._native, position, mask.size, data, self._ink(ink), bool(self.blend), native_stroke)

        stroke_fill = fill if stroke_fill is None else stroke_fill
        if stroke:
            render(stroke_fill, stroke)
        if not stroke or self._ink(fill) != self._ink(stroke_fill):
            render(fill, 0)

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
        """Draw newline-separated text with spacing and alignment.

        Each line uses the selected font's advance length for alignment.
        Line spacing includes the font's `A` bounding height and stroke width.

        Args:
            xy: Position of the multiline text anchor.
            text: String whose lines are separated by `\n`.
            fill: Text ink. `None` uses the context's default ink.
            font: Explicit font object; otherwise use `self.font` or the
                built-in bitmap fallback.
            anchor: Two-character anchor. Horizontal codes are `l`, `m`, and
                `r`; vertical codes are `a`, `m`, `s`, and `d`. Defaults to `"la"`.
            spacing: Finite additional spacing between lines in pixels.
            align: Line alignment: `"left"`, `"center"`, or `"right"`.
            direction: Direction passed to the font. Vertical `"ttb"` layout
                is unsupported for multiline text.
            features: OpenType feature names passed to the font when specified.
            language: Language tag passed to the font when specified.
            stroke_width: Finite, nonnegative stroke thickness, rounded up
                to a whole pixel for rendering.
            stroke_fill: Stroke ink. `None` uses the text ink.
            embedded_color: Embedded-color font request. Currently unsupported.
            font_size: Bitmap fallback height, scaled in multiples of seven
                pixels. Ignored when an explicit or context font is used.

        Raises:
            TypeError: If arguments or the font interface are incompatible.
            ValueError: If alignment, anchor, spacing, stroke width, or direction
                is invalid, or drawing in the image mode is unsupported.
            NotImplementedError: If `embedded_color` is true.
        """
        font = self._font(font, font_size)
        self._text_options(direction, features, language, embedded_color)
        lines = self._text_layout(xy, text, font, anchor, spacing, align, direction, features, language, stroke_width)
        for position, line in lines:
            self.text(position, line, fill, font, anchor, direction=direction, features=features, language=language, stroke_width=stroke_width, stroke_fill=stroke_fill, embedded_color=embedded_color)

    def _font(self, font: object | None, font_size: float | None) -> object:
        """Resolve an explicit font, the context font, or the bitmap fallback."""
        if font is not None:
            return font
        if self.font is not None:
            return self.font
        return BitmapFont(font_size)

    @staticmethod
    def _font_options(direction: str | None, features: Sequence[str] | None, language: str | None) -> dict[str, object]:
        """Collect only the explicitly supplied font shaping options."""
        return {key: value for key, value in (("direction", direction), ("features", features), ("language", language)) if value is not None}

    @staticmethod
    def _text_options(direction: str | None, features: Sequence[str] | None, language: str | None, embedded_color: bool) -> None:
        """Reject embedded-color text until a supporting font backend exists."""
        if embedded_color:
            raise NotImplementedError("embedded color fonts require the ImageFont backend")

    def _text_layout(
        self,
        xy: tuple[float, float],
        text: str,
        font: object,
        anchor: str | None,
        spacing: float,
        align: str,
        direction: str | None,
        features: Sequence[str] | None,
        language: str | None,
        stroke_width: float,
    ) -> list[tuple[tuple[float, float], str]]:
        """Return anchored line positions using font advances and line spacing."""
        if align not in ("left", "center", "right"):
            raise ValueError("align must be left, center, or right")
        if direction == "ttb":
            raise ValueError("multiline text does not support vertical direction")
        anchor = "la" if anchor is None else anchor
        if len(anchor) != 2 or anchor[0] not in "lmr" or anchor[1] not in "amsd":
            raise ValueError("invalid multiline text anchor")
        if not isfinite(spacing) or not isfinite(stroke_width) or stroke_width < 0:
            raise ValueError("spacing and stroke_width must be finite; stroke_width must be nonnegative")
        options = self._font_options(direction, features, language)
        lines = text.split("\n")
        widths = [float(font.getlength(line, **options)) for line in lines]
        max_width = max(widths)
        line_height = font.getbbox("A", stroke_width=stroke_width, **options)[3] + stroke_width + spacing
        x, y = xy
        y -= {"a": 0, "s": 0, "m": (len(lines) - 1) * line_height / 2, "d": (len(lines) - 1) * line_height}[anchor[1]]
        positions = []
        for i, (line, width) in enumerate(zip(lines, widths, strict=True)):
            shift = {"left": 0, "center": (max_width - width) / 2, "right": max_width - width}[align]
            shift -= {"l": 0, "m": (max_width - width) / 2, "r": max_width - width}[anchor[0]]
            positions.append(((x + shift, y + i * line_height), line))
        return positions

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
        """Return the single-line advance length in pixels.

        Font kerning is included when provided by the font. The advance
        describes the next text position; glyph bounds may extend beyond it.

        Args:
            text: Single-line string to measure. Newlines are unsupported.
            font: Explicit font object; otherwise use `self.font` or the
                built-in bitmap fallback.
            direction: Direction passed to the font when specified.
            features: OpenType feature names passed to the font when specified.
            language: Language tag passed to the font when specified.
            embedded_color: Embedded-color font request. Currently unsupported.
            font_size: Bitmap fallback height, scaled in multiples of seven
                pixels. Ignored when an explicit or context font is used.

        Returns:
            The font's advance length as a float, without stroke expansion.

        Raises:
            TypeError: If the text or font interface is incompatible.
            ValueError: If text contains newlines or the fallback size is invalid.
            NotImplementedError: If `embedded_color` is true.
        """
        if "\n" in text:
            raise ValueError("textlength does not support multiline text")
        self._text_options(direction, features, language, embedded_color)
        return float(self._font(font, font_size).getlength(text, **self._font_options(direction, features, language)))

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
        """Return text bounds at `xy`, including the anchor and stroke.

        Strings containing newlines are measured with
        :meth:`multiline_textbbox`. Measurement does not modify the image.

        Args:
            xy: Text anchor position in image coordinates.
            text: String to measure.
            font: Explicit font object; otherwise use `self.font` or the
                built-in bitmap fallback.
            anchor: Font anchor, such as `"la"` or `"mm"`. `None` uses the
                font's default. Supported anchors depend on the font.
            spacing: Additional spacing in pixels for newline-separated text.
            align: Multiline alignment: `"left"`, `"center"`, or `"right"`.
            direction: Direction passed to the font when specified.
            features: OpenType feature names passed to the font when specified.
            language: Language tag passed to the font when specified.
            stroke_width: Finite, nonnegative expansion passed to the font's
                bounding-box calculation.
            embedded_color: Embedded-color font request. Currently unsupported.
            font_size: Bitmap fallback height, scaled in multiples of seven
                pixels. Ignored when an explicit or context font is used.

        Returns:
            `(left, top, right, bottom)` in image coordinates. These bounds
            include font margins and are not clipped to the image.

        Raises:
            TypeError: If arguments or the font interface are incompatible.
            ValueError: If stroke width, font size, anchor, or multiline
                layout options are invalid.
            NotImplementedError: If `embedded_color` is true.
        """
        if "\n" in text:
            return self.multiline_textbbox(xy, text, font, anchor, spacing, align, direction, features, language, stroke_width, embedded_color, font_size)
        self._text_options(direction, features, language, embedded_color)
        if not isfinite(stroke_width) or stroke_width < 0:
            raise ValueError("stroke_width must be finite and nonnegative")
        box = self._font(font, font_size).getbbox(text, anchor=anchor, stroke_width=stroke_width, **self._font_options(direction, features, language))
        return box[0] + xy[0], box[1] + xy[1], box[2] + xy[0], box[3] + xy[1]

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
        """Return the union of the bounds of newline-separated text.

        Uses the same line positions as :meth:`multiline_text` and does not
        modify the image. Font margins and stroke expansion are included.

        Args:
            xy: Position of the multiline text anchor.
            text: String whose lines are separated by `\n`.
            font: Explicit font object; otherwise use `self.font` or the
                built-in bitmap fallback.
            anchor: Two-character anchor using horizontal `l`, `m`, or `r`
                and vertical `a`, `m`, `s`, or `d`. Defaults to `"la"`.
            spacing: Finite additional spacing between lines in pixels.
            align: Line alignment: `"left"`, `"center"`, or `"right"`.
            direction: Direction passed to the font. Vertical `"ttb"` layout
                is unsupported for multiline text.
            features: OpenType feature names passed to the font when specified.
            language: Language tag passed to the font when specified.
            stroke_width: Finite, nonnegative expansion passed to the font's
                bounding-box calculation.
            embedded_color: Embedded-color font request. Currently unsupported.
            font_size: Bitmap fallback height, scaled in multiples of seven
                pixels. Ignored when an explicit or context font is used.

        Returns:
            `(left, top, right, bottom)` in image coordinates, without
            clipping to the image.

        Raises:
            TypeError: If arguments or the font interface are incompatible.
            ValueError: If alignment, anchor, spacing, stroke width, direction,
                or fallback size is invalid.
            NotImplementedError: If `embedded_color` is true.
        """
        font = self._font(font, font_size)
        self._text_options(direction, features, language, embedded_color)
        boxes = [
            self.textbbox(position, line, font, anchor, direction=direction, features=features, language=language, stroke_width=stroke_width)
            for position, line in self._text_layout(xy, text, font, anchor, spacing, align, direction, features, language, stroke_width)
        ]
        return min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)


def Draw(im: Image.Image, mode: str | None = None, *, antialias: bool = False) -> ImageDraw:
    """Create a drawing context for `im`.

    Args:
        im: Image to draw on.
        mode: Optional ink mode. Defaults to the image mode.
        antialias: Use supersampled coverage for geometric edges on 8-bit images.

    Returns:
        A context that modifies `im` in place.

    Raises:
        TypeError: If `im` is not a Blanket image.
        ValueError: If `im` is closed or the drawing mode is incompatible.

    Examples:
        ```python
        from blanket import Image, ImageDraw

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        draw = ImageDraw.Draw(image)
        ```
    """
    return ImageDraw(im, mode, antialias=antialias)


def floodfill(image: Image.Image, xy: tuple[float, float], value: Color, border: Color = None, thresh: float = 0) -> None:
    """Fill four-connected pixels starting at `xy`.

    The fill is applied in place. Seeds outside the image are ignored.
    Without a border, pixels are compared to the seed using the sum of
    absolute channel differences, or the absolute scalar sample difference.
    With a border, filling stops at pixels equal to the border or fill ink.

    Args:
        image: Image to fill.
        xy: Seed coordinate. Fractional values are truncated toward zero.
        value: Fill ink in the image mode, following the drawing context's
            color and numeric sample rules.
        border: Optional ink that bounds the fill. `None` uses seed similarity.
        thresh: Finite, nonnegative maximum difference from the seed. With a
            border, neighbor selection uses exact ink equality instead.

    Raises:
        TypeError: If the image, coordinates, or inks have incompatible types.
        ValueError: If the threshold is invalid, the image is closed, or the
            ink or sample depth is unsupported.

    Examples:
        ```python
        from blanket import Image, ImageDraw

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        ImageDraw.floodfill(image, (0, 0), (255, 0, 0))
        ```
    """
    if not isfinite(thresh) or thresh < 0:
        raise ValueError("thresh must be finite and nonnegative")
    draw = Draw(image)
    draw_floodfill(image._native, (int(xy[0]), int(xy[1])), draw._ink(value), None if border is None else draw._ink(border), thresh)
