from __future__ import annotations

import math
import struct

import pytest
from PIL import Image as PILImage, ImageMath as PILMath

from blanket import Image, ImageMath


def pair(mode: str, size: tuple[int, int] = (3, 2)) -> tuple[Image.Image, PILImage.Image]:
    values = [-7, -2, 0, 1, 3, 12] if mode in ("I", "F") else [0, 1, 12, 127, 128, 255]
    if mode == "F":
        values = [v / 2 for v in values]
    reference = PILImage.new(mode, size)
    reference.putdata([values[i % len(values)] for i in range(size[0] * size[1])])
    return Image.frombytes(mode, size, reference.tobytes()), reference


def same(actual: Image.Image, expected: PILImage.Image) -> None:
    assert (actual.mode, actual.size) == (expected.mode, expected.size)
    if actual.mode == "F":
        reference = (value[0] for value in struct.iter_unpack("<f", expected.tobytes()))
        for a, b in zip(actual.getdata(), reference, strict=True):
            assert math.isnan(a) if math.isnan(b) else a == pytest.approx(b)
    else:
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["1", "L", "I", "F"])
@pytest.mark.parametrize(
    "expression",
    [
        "+a",
        "-a",
        "abs(a)",
        "a + b",
        "a - b",
        "a * b",
        "a / b",
        "a % b",
        "a ** 2",
        "a + 2.9",
        "2.9 + a",
        "5 - a",
        "3 * a",
        "7 / a",
        "7 % a",
        "2 ** a",
        "a == b",
        "a != b",
        "a < b",
        "a <= b",
        "a > b",
        "a >= b",
        "min(a, b)",
        "max(a, b)",
        "equal(a, b)",
        "notequal(a, b)",
        "int(a)",
        "float(a)",
        "convert(a + b, 'L')",
        "a and b",
        "a or b",
    ],
)
def test_arithmetic_matches_pillow(mode: str, expression: str) -> None:
    a, pa = pair(mode)
    b, pb = pair(mode, (2, 3))
    if "2.9" in expression and mode != "F":
        with pytest.raises(TypeError):
            PILMath.unsafe_eval(expression, a=pa, b=pb)
        with pytest.raises(TypeError):
            ImageMath.unsafe_eval(expression, a=a, b=b)
        return
    same(ImageMath.unsafe_eval(expression, a=a, b=b), PILMath.unsafe_eval(expression, a=pa, b=pb))


@pytest.mark.parametrize("mode", ["1", "L", "I"])
@pytest.mark.parametrize("expression", ["~a", "a & 3", "3 & a", "a | 3", "3 | a", "a ^ 3", "3 ^ a", "a << 2", "a >> 2"])
def test_bitwise_matches_pillow(mode: str, expression: str) -> None:
    a, pa = pair(mode)
    same(ImageMath.eval(expression, a=a), PILMath.unsafe_eval(expression, a=pa))


@pytest.mark.parametrize("expression", ["a + b", "b - a", "a / b", "a < b", "equal(a, b)"])
def test_mixed_modes(expression: str) -> None:
    a, pa = pair("I")
    b, pb = pair("F")
    same(ImageMath.eval(expression, a=a, b=b), PILMath.unsafe_eval(expression, a=pa, b=pb))


def test_lambda_helpers_and_scalar_results() -> None:
    a, pa = pair("L")

    def expression(args):
        return args["convert"](args["max"](args["float"](args["a"]) / 2, 1.5), "L")

    same(ImageMath.lambda_eval(expression, a=a), PILMath.lambda_eval(expression, a=pa))
    assert ImageMath.eval("(2 + n, 3.5)", n=4) == (6, 3.5)
    assert ImageMath.lambda_eval(lambda args: args["n"] + 1, n=4) == 5
    assert ImageMath.lambda_eval(lambda args: args["a"], a=a) is a
    assert ImageMath.eval("not a", a=Image.new("I", (2, 2), 0)) is True


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "LA", "I;16"])
def test_unsupported_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        ImageMath.eval("a + 1", a=Image.new(mode, (1, 1)))


@pytest.mark.parametrize("expression", ["~a", "a & 1", "a | 1", "a ^ 1", "a << 1", "a >> 1"])
def test_float_bitwise_rejected(expression: str) -> None:
    with pytest.raises(TypeError, match="bad operand type"):
        ImageMath.eval(expression, a=Image.new("F", (1, 1)))


@pytest.mark.parametrize("expression", ["a.__class__", "__import__('os')", "(lambda: missing)()"])
def test_expression_names_restricted(expression: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        ImageMath.eval(expression, a=Image.new("L", (1, 1)))


@pytest.mark.parametrize("name", ["__builtins__", "open", "abs"])
def test_operand_names_restricted(name: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        ImageMath.eval("1", **{name: 1})


def test_closed_image() -> None:
    image = Image.new("I", (1, 1))
    image.close()
    with pytest.raises(ValueError):
        ImageMath.eval("a + 1", a=image)


@pytest.mark.parametrize("size", [(0, 2), (2, 0), (257, 259)])
def test_empty_and_parallel_buffers(size: tuple[int, int]) -> None:
    a, pa = pair("I", size)
    original = a.tobytes()
    same(ImageMath.eval("(a * 3 - 2) / 2", a=a), PILMath.unsafe_eval("(a * 3 - 2) / 2", a=pa))
    assert a.tobytes() == original


def test_integer_boundaries_and_zero_divisors() -> None:
    image = Image.new("I", (3, 1))
    image.putdata([-2147483648, 2147483647, -7])
    assert list(ImageMath.eval("a + 1", a=image).getdata()) == [-2147483647, -2147483648, -6]
    assert list(ImageMath.eval("a / -1", a=image).getdata()) == [-2147483648, -2147483647, 7]
    assert list(ImageMath.eval("a / 0", a=image).getdata()) == [0, 0, 0]
    assert list(ImageMath.eval("a % 0", a=image).getdata()) == [0, 0, 0]
