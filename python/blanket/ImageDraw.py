"""Draw geometry, text, and connected fills directly on Blanket images.

Use :func:`Draw` to create a context. Geometric operations support clipping,
mode-specific inks, and optional supersampled coverage. Text uses a small
bitmap fallback or an explicitly supplied font object; Pillow is not required.
"""

from __future__ import annotations

__lazy_modules__ = {"collections.abc", "blanket.ImagePath", "blanket._draw_font", "math", "operator"}

from math import atan2, ceil, cos, degrees, floor, isfinite, pi, radians, sin
from operator import index
from typing import TYPE_CHECKING

from . import Image
from ._blanket import draw_curve, draw_ellipse, draw_floodfill, draw_image_mask, draw_mask, draw_path, draw_points, draw_polygon, draw_rectangle
from ._color import color_pixel
from ._draw_font import BitmapFont, PublicDomainFont
from .ImageFont import load_default

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .ImagePath import Path

    Coords = Sequence[float] | Sequence[tuple[float, float]] | Path
    Color = int | float | str | tuple[int, ...] | None

__all__ = ["Draw", "ImageDraw", "Outline", "floodfill"]


class Outline:
    """Flattened cubic Bezier and line segments for ImageDraw.shape()."""

    def __init__(self) -> None:
        """Initialize an empty Outline.

        Examples:
            ```python
            from blanket.ImageDraw import Outline

            outline = Outline()
            ```
        """
        self.x = self.x0 = 0.0
        self.y = self.y0 = 0.0
        self.edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def move(self, x0: float, y0: float) -> None:
        """Move the current position to (x0, y0) without drawing an edge.

        Args:
            x0: Target x-coordinate.
            y0: Target y-coordinate.

        Examples:
            ```python
            from blanket.ImageDraw import Outline

            outline = Outline()
            outline.move(10.0, 10.0)
            ```
        """
        self.x = self.x0 = float(x0)
        self.y = self.y0 = float(y0)

    def line(self, x1: float, y1: float) -> None:
        """Add a line segment from the current position to (x1, y1).

        Args:
            x1: Ending x-coordinate.
            y1: Ending y-coordinate.

        Examples:
            ```python
            from blanket.ImageDraw import Outline

            outline = Outline()
            outline.move(0.0, 0.0)
            outline.line(10.0, 10.0)
            ```
        """
        x1, y1 = float(x1), float(y1)
        self.edges.append(((int(self.x), int(self.y)), (int(x1), int(y1))))
        self.x, self.y = x1, y1

    def curve(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> None:
        """Add a cubic Bezier curve segment to (x3, y3).

        Args:
            x1: First control point x-coordinate.
            y1: First control point y-coordinate.
            x2: Second control point x-coordinate.
            y2: Second control point y-coordinate.
            x3: Destination endpoint x-coordinate.
            y3: Destination endpoint y-coordinate.

        Examples:
            ```python
            from blanket.ImageDraw import Outline

            outline = Outline()
            outline.move(0.0, 0.0)
            outline.curve(5.0, 0.0, 10.0, 5.0, 10.0, 10.0)
            ```
        """
        x1, y1, x2, y2, x3, y3 = float(x1), float(y1), float(x2), float(y2), float(x3), float(y3)
        xo, yo = self.x, self.y
        for step in range(1, 33):
            t = step / 32.0
            t2 = t * t
            t3 = t2 * t
            u = 1.0 - t
            u2 = u * u
            u3 = u2 * u
            x = self.x * u3 + 3.0 * (x1 * t * u2 + x2 * t2 * u) + x3 * t3 + 0.5
            y = self.y * u3 + 3.0 * (y1 * t * u2 + y2 * t2 * u) + y3 * t3 + 0.5
            self.edges.append(((int(xo), int(yo)), (int(x), int(y))))
            xo, yo = x, y
        self.x, self.y = xo, yo

    def close(self) -> None:
        """Close the outline by connecting the current position back to the origin.

        Examples:
            ```python
            from blanket.ImageDraw import Outline

            outline = Outline()
            outline.move(0.0, 0.0)
            outline.line(10.0, 0.0)
            outline.line(10.0, 10.0)
            outline.close()
            ```
        """
        if self.x == self.x0 and self.y == self.y0:
            return
        self.line(self.x0, self.y0)

    def transform(self, matrix: Sequence[float]) -> None:
        """Transform all points in the outline using an affine matrix.

        Args:
            matrix: Sequence of 6 transformation coefficients `(a, b, c, d, e, f)`.

        Examples:
            ```python
            from blanket.ImageDraw import Outline

            outline = Outline()
            outline.move(1.0, 1.0)
            outline.transform([1.0, 0.0, 5.0, 0.0, 1.0, 5.0])
            ```
        """
        if len(matrix) != 6:
            raise TypeError("transform matrix must contain six values")
        a, b, c, d, e, f = (float(value) for value in matrix)
        if not all(isfinite(value) for value in (a, b, c, d, e, f)):
            raise ValueError("transform matrix must be finite")

        def point(x: float, y: float) -> tuple[int, int]:
            """Apply the affine matrix to coordinates (x, y)."""
            return int(a * x + b * y + c), int(d * x + e * y + f)

        edges = [(point(*start), point(*end)) for start, end in self.edges]
        self.x, self.y = point(self.x, self.y)
        self.x0, self.y0 = point(self.x0, self.y0)
        self.edges = edges


_BINARY_TABLE = bytes([0] + [255] * 255)


def _compute_regular_polygon_vertices(bounding_circle: Sequence[Sequence[float] | float], n_sides: int, rotation: float) -> list[tuple[float, float]]:
    """Generate a list of vertices for a 2D regular polygon."""
    if not isinstance(n_sides, int) or isinstance(n_sides, bool):
        raise TypeError("n_sides should be an int")
    if n_sides < 3:
        raise ValueError("n_sides should be an int > 2")
    if not isinstance(bounding_circle, (list, tuple)):
        raise TypeError("bounding_circle should be a sequence")
    if len(bounding_circle) == 3:
        if not all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in bounding_circle):
            raise ValueError("bounding_circle should only contain numeric data")
        centroid_x, centroid_y, polygon_radius = (float(v) for v in bounding_circle)
    elif len(bounding_circle) == 2 and isinstance(bounding_circle[0], (list, tuple)):
        if len(bounding_circle[0]) != 2:
            raise ValueError("bounding_circle centre should contain 2D coordinates (e.g. (x, y))")
        if not all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in bounding_circle[0]) or not isinstance(bounding_circle[1], (int, float)) or isinstance(bounding_circle[1], bool):
            raise ValueError("bounding_circle should only contain numeric data")
        centroid_x, centroid_y = (float(v) for v in bounding_circle[0])
        polygon_radius = float(bounding_circle[1])
    else:
        raise ValueError("bounding_circle should contain 2D coordinates and a radius (e.g. (x, y, r) or ((x, y), r) )")

    if polygon_radius <= 0:
        raise ValueError("bounding_circle radius should be > 0")
    if not isinstance(rotation, (int, float)) or isinstance(rotation, bool):
        raise ValueError("rotation should be an int or float")

    def _apply_rotation(point: tuple[float, float], deg: float) -> tuple[float, float]:
        """Rotate a 2D point around the centroid by given degrees."""
        rad = radians(360 - deg)
        return (round(point[0] * cos(rad) - point[1] * sin(rad) + centroid_x, 2), round(point[1] * cos(rad) + point[0] * sin(rad) + centroid_y, 2))

    degrees = 360 / n_sides
    current_angle = (270 - 0.5 * degrees) + rotation
    angles = []
    for _ in range(n_sides):
        angles.append(current_angle)
        current_angle += degrees
        if current_angle > 360:
            current_angle -= 360

    return [_apply_rotation((polygon_radius, 0.0), angle) for angle in angles]


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

    font: object | None = None

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
        im._ensure_mutable()
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
        self.antialias = antialias

    def getfont(self) -> object:
        """Return the active default font, initializing it on first access."""
        if not self.font:
            self.font = load_default()
        return self.font

    def shape(self, shape: Outline, fill: Color = None, outline: Color = None) -> None:
        """Close and rasterize an Outline using the scanline polygon filler."""
        shape.close()
        points = [edge[0] for edge in shape.edges]
        if shape.edges:
            points.append(shape.edges[-1][1])
        if fill is None and outline is None:
            outline = 1 if self.mode in ("I", "F") else -1
        fill_ink = None if fill is None else self._ink(fill)
        outline_ink = None if outline is None else self._ink(outline)
        for ink in (fill_ink, None if outline_ink == fill_ink else outline_ink):
            if ink is not None and points:
                draw_polygon(self.im._native, points, 1, True, True, ink, bool(self.blend))

    def _ink(self, fill: Color) -> int | float | tuple[int, ...]:
        """Normalize an ink to native pixel values in the drawing mode."""
        if fill is None:
            fill = 1 if self.mode in ("I", "F") else -1
        if self.mode == "P" and not isinstance(fill, int):
            palette = self.im.palette
            if palette is None:
                raise ValueError("image has no palette")
            if isinstance(fill, str):
                fill = tuple(color_pixel(fill, palette.mode))
            ink = palette.getcolor(fill, self.im)
            self.im._sync_palette()
            return ink
        if self.mode in ("I", "F", "I;16", "I;16L", "I;16B", "I;16N", "P"):
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
        if not self.antialias:
            draw_curve(self.im._native, self._bounds(xy), start, end, index(width), "arc", False, self._ink(fill), bool(self.blend))
            return
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
        if not self.antialias:
            self._native_curve(xy, start, end, "chord", fill, outline, width)
            return
        self._shape(self._curve(xy, start, end), fill, outline, width)

    def bitmap(self, xy: Sequence[float], bitmap: Image.Image, fill: Color = None) -> None:
        """Draw ink through a bilevel, grayscale, or RGBA alpha mask."""
        x, y = xy
        mask = bitmap.getchannel("A") if bitmap.mode == "RGBA" else bitmap
        if mask.mode not in ("1", "L"):
            raise ValueError("bad transparency mask")
        if bitmap.mode != "RGBA":
            mask.load()
            if mask._native is self.im._native:
                mask = mask.copy()
            draw_image_mask(self.im._native, (int(x), int(y)), mask._native, self._ink(fill), bool(self.blend))
            return
        position = (int(x), int(y), int(x) + mask.width, int(y) + mask.height)
        ink = Image.new(self.im.mode, mask.size, self._ink(fill))
        self.im.paste(ink, position, mask)

    def circle(self, xy: Sequence[float], radius: float, fill: Color = None, outline: Color = None, width: int = 1) -> None:
        """Draw a circle centered at xy using the ellipse rasterizer."""
        x, y = xy
        self.ellipse((x - radius, y - radius, x + radius, y + radius), fill, outline, width)

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
        fill_ink = None if fill is None else self._ink(fill)
        outline_ink = None if outline is None else self._ink(outline)
        if fill_ink is not None:
            draw_ellipse(self.im._native, bounds, fill_ink, 0, bool(self.blend), bool(self.antialias))
        if outline_ink is not None and outline_ink != fill_ink and width > 0:
            draw_ellipse(self.im._native, bounds, outline_ink, width, bool(self.blend), bool(self.antialias))

    def line(self, xy: Coords, fill: Color = None, width: int = 0, joint: str | None = None) -> None:
        """Draw connected line segments through a sequence of coordinates.

        Pixels outside the image are ignored. Fewer than two coordinates draw
        nothing. Wide strokes use rounded
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
        width = index(width)
        if width == 0:
            width = 1
        if not self.antialias:
            points = self._coords(xy)
            ink = self._ink(fill)
            if len(points) < 2:
                self.im._native.load()
                return
            draw_polygon(self.im._native, points, max(1, width), False, False, ink, bool(self.blend))
            if joint == "curve" and width > 4:
                for previous, point, following in zip(points, points[1:], points[2:], strict=False):
                    angles = [degrees(atan2(end[0] - start[0], start[1] - end[1])) % 360 for start, end in ((previous, point), (point, following))]
                    if angles[0] == angles[1]:
                        continue
                    flipped = (angles[1] > angles[0] and angles[1] - 180 > angles[0]) or (angles[1] < angles[0] and angles[1] + 180 > angles[0])
                    start, end = (angles[1] + 90, angles[0] + 90) if flipped else (angles[0] - 90, angles[1] - 90)
                    self.pieslice((point[0] - width / 2 + 1, point[1] - width / 2 + 1, point[0] + width / 2 - 1, point[1] + width / 2 - 1), start - 90, end - 90, fill=fill)
                    if width > 8:
                        gap = []
                        for angle in angles:
                            radians = (angle + (90 if flipped else -90) - 90) * pi / 180
                            dx, dy = (width / 2 - 1) * cos(radians), (width / 2 - 1) * sin(radians)
                            gap.append((point[0] + (floor(dx) if dx > 0 else ceil(dx)), point[1] + (floor(dy) if dy > 0 else ceil(dy))))
                        self.line([gap[0], point, gap[1]], fill, 3)
            return
        points = self._coords(xy)
        if len(points) < 2:
            self._ink(fill)
            self.im._native.load()
            return
        self._path(points, fill, width=max(1, width))

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
        if not self.antialias:
            self._native_curve(xy, start, end, "pie", fill, outline, width)
            return
        bounds = self._bounds(xy)
        points = self._curve(xy, start, end)
        if not points:
            return
        if abs(end - start) < 360:
            points.append(((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2))
        self._shape(points, fill, outline, width)

    def _native_curve(self, xy: Coords, start: float, end: float, style: str, fill: Color, outline: Color, width: int) -> None:
        """Render a filled or outlined arc, chord, or pie slice natively."""
        bounds = self._bounds(xy)
        fill_ink = None if fill is None else self._ink(fill)
        outline_ink = None if outline is None else self._ink(outline)
        if fill_ink is not None:
            draw_curve(self.im._native, bounds, start, end, 0, style, True, fill_ink, bool(self.blend))
        if outline_ink is not None and outline_ink != fill_ink and width > 0:
            draw_curve(self.im._native, bounds, start, end, index(width), style, False, outline_ink, bool(self.blend))

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
        fill_ink = None if fill is None else self._ink(fill)
        outline_ink = None if outline is None else self._ink(outline)
        if fill_ink is not None:
            self._path(points, fill_ink, closed=True, filled=True)
        if outline_ink is not None and outline_ink != fill_ink and width > 0:
            self._path(points, outline_ink, width=width, closed=True)

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
        if not self.antialias:
            fill_ink = None if fill is None else self._ink(fill)
            outline_ink = None if outline is None else self._ink(outline)
            if fill_ink is not None:
                draw_polygon(self.im._native, points, 1, True, True, fill_ink, bool(self.blend))
            if outline_ink is not None and outline_ink != fill_ink and width > 0:
                draw_polygon(self.im._native, points, index(width), False, True, outline_ink, bool(self.blend))
            return
        self._shape(points, fill, outline, width)

    def regular_polygon(self, bounding_circle: tuple[float, float, float] | tuple[tuple[float, float], float], n_sides: int, rotation: float = 0, fill: Color = None, outline: Color = None, width: int = 1) -> None:
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
        points = _compute_regular_polygon_vertices(bounding_circle, n_sides, rotation)
        self.polygon(points, fill, outline, width)

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
        fill_ink = None if fill is None else self._ink(fill)
        outline_ink = None if outline is None else self._ink(outline)
        if fill_ink is not None:
            draw_rectangle(self.im._native, bounds, fill_ink, 0, bool(self.blend))
        if outline_ink is not None and outline_ink != fill_ink and width > 0:
            draw_rectangle(self.im._native, bounds, outline_ink, width, bool(self.blend))

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
        corners = (True, True, True, True) if corners is None else corners
        if len(corners) != 4:
            raise ValueError("corners must contain four flags")
        if not self.antialias:
            self._rounded_rectangle(xy, radius, fill, outline, index(width), corners)
            return
        radius = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
        points = []
        for enabled, (cx, cy, start, corner) in zip(corners, [(x0 + radius, y0 + radius, 180, (x0, y0)), (x1 - radius, y0 + radius, 270, (x1, y0)), (x1 - radius, y1 - radius, 0, (x1, y1)), (x0 + radius, y1 - radius, 90, (x0, y1))], strict=True):
            if enabled and radius:
                count = max(1, min(16384, ceil(radius * pi)))
                points.extend((cx + radius * cos((start + 90 * i / count) * pi / 180), cy + radius * sin((start + 90 * i / count) * pi / 180)) for i in range(count + 1))
            else:
                points.append(corner)
        self._shape(points, fill, outline, width)

    def _rounded_rectangle(self, xy: Coords, radius: float, fill: Color, outline: Color, width: int, corners: tuple[bool, bool, bool, bool]) -> None:
        """Render a rounded rectangle with specified corner radii."""
        coordinates = list(xy)
        if len(coordinates) == 2:
            (x0, y0), (x1, y1) = coordinates
        else:
            x0, y0, x1, y1 = coordinates
        diameter = min(x1 - x0, y1 - y0, radius * 2)
        x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
        full_x = full_y = False
        if all(corners):
            full_x = diameter >= x1 - x0 - 1
            if full_x:
                diameter = x1 - x0
            full_y = diameter >= y1 - y0 - 1
            if full_y:
                diameter = y1 - y0
            if full_x and full_y:
                self.ellipse(xy, fill, outline, width)
                return
        if diameter == 0 or not any(corners):
            self.rectangle(xy, fill, outline, width)
            return
        r = int(diameter // 2)
        if full_x:
            parts = [((x0, y0, x0 + diameter, y0 + diameter), 180, 360), ((x0, y1 - diameter, x0 + diameter, y1), 0, 180)]
        elif full_y:
            parts = [((x0, y0, x0 + diameter, y0 + diameter), 90, 270), ((x1 - diameter, y0, x1, y0 + diameter), 270, 90)]
        else:
            parts = [
                part
                for enabled, part in zip(
                    corners, [((x0, y0, x0 + diameter, y0 + diameter), 180, 270), ((x1 - diameter, y0, x1, y0 + diameter), 270, 360), ((x1 - diameter, y1 - diameter, x1, y1), 0, 90), ((x0, y1 - diameter, x0 + diameter, y1), 90, 180)], strict=True
                )
                if enabled
            ]

        def rectangle(bounds: tuple[int, int, int, int], ink: Color) -> None:
            """Draw a filled rectangle within the given bounds."""
            draw_rectangle(self.im._native, bounds, self._ink(ink), 0, bool(self.blend))

        fill_ink = None if fill is None else self._ink(fill)
        outline_ink = None if outline is None else self._ink(outline)
        for ink, filled in ((fill_ink, True), (outline_ink, False)):
            if ink is None or (not filled and (ink == fill_ink or width <= 0)):
                continue
            for bounds, start, end in parts:
                draw_curve(self.im._native, tuple(int(v) for v in bounds), start, end, 0 if filled else width, "pie" if filled else "arc", filled, ink, bool(self.blend))
            if filled:
                if full_x:
                    rectangle((x0, y0 + r + 1, x1, y1 - r - 1), ink)
                elif x1 - r - 1 >= x0 + r + 1:
                    rectangle((x0 + r + 1, y0, x1 - r - 1, y1), ink)
                if not full_x and not full_y:
                    rectangle((x0, y0 + (r + 1 if corners[0] else 0), x0 + r, y1 - (r + 1 if corners[3] else 0)), ink)
                    rectangle((x1 - r, y0 + (r + 1 if corners[1] else 0), x1, y1 - (r + 1 if corners[2] else 0)), ink)
            else:
                if not full_x:
                    rectangle((x0 + (r + 1 if corners[0] else 0), y0, x1 - (r + 1 if corners[1] else 0), y0 + width - 1), ink)
                    rectangle((x0 + (r + 1 if corners[3] else 0), y1 - width + 1, x1 - (r + 1 if corners[2] else 0), y1), ink)
                if not full_y:
                    rectangle((x0, y0 + (r + 1 if corners[0] else 0), x0 + width - 1, y1 - (r + 1 if corners[3] else 0)), ink)
                    rectangle((x1 - width + 1, y0 + (r + 1 if corners[1] else 0), x1, y1 - (r + 1 if corners[2] else 0)), ink)

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
        Use `ImageFont.truetype` to load TrueType and OpenType fonts.
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
            font_size: Requested bitmap fallback height. Defaults to eight
                pixels and scales in whole multiples of eight. Ignored when
                an explicit or context font is used.

        Raises:
            TypeError: If coordinates, inks, or the font interface are incompatible.
            ValueError: If stroke width, font size, or anchor is invalid,
                the image is closed, or mask coverage is unsupported for its mode.
            NotImplementedError: If `embedded_color` is true.

        Notes:
            The public-domain bitmap font includes Basic Latin lowercase letters
            and uses `?` for unsupported characters. It ignores font shaping options.
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
            if mode == "1" and not isinstance(font, BitmapFont):
                data = data.translate(_BINARY_TABLE)
            position = (int(xy[0]) + int(offset[0]), int(xy[1]) + int(offset[1]))
            draw_mask(self.im._native, position, mask.size, data, self._ink(ink), bool(self.blend), native_stroke)

        stroke_fill = fill if stroke_fill is None else stroke_fill
        if stroke:
            render(stroke_fill, stroke)
        if not stroke or self._ink(fill) != self._ink(stroke_fill):
            render(fill, 0)
        return None

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
            font_size: Bitmap fallback height, scaled in multiples of eight
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
        return PublicDomainFont(font_size)

    @staticmethod
    def _font_options(direction: str | None, features: Sequence[str] | None, language: str | None) -> dict[str, object]:
        """Collect only the explicitly supplied font shaping options."""
        return {key: value for key, value in (("direction", direction), ("features", features), ("language", language)) if value is not None}

    @staticmethod
    def _text_options(direction: str | None, features: Sequence[str] | None, language: str | None, embedded_color: bool) -> None:
        """Reject embedded-color text until a supporting font backend exists."""
        if embedded_color:
            raise NotImplementedError("embedded color fonts are not supported")

    def _text_layout(
        self, xy: tuple[float, float], text: str, font: object, anchor: str | None, spacing: float, align: str, direction: str | None, features: Sequence[str] | None, language: str | None, stroke_width: float
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

    def textlength(self, text: str, font: object | None = None, direction: str | None = None, features: Sequence[str] | None = None, language: str | None = None, embedded_color: bool = False, font_size: float | None = None) -> float:
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
