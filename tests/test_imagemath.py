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


@pytest.mark.parametrize("expression", ["value + 1", "(lambda x: x + 1)(value)"])
def test_repeated_expression_validates_current_operands(expression: str) -> None:
    assert ImageMath.unsafe_eval(expression, value=3) == 4
    assert ImageMath.unsafe_eval(expression, value=7) == 8
    with pytest.raises(ValueError, match="'value' not allowed"):
        ImageMath.unsafe_eval(expression, other=3)
    assert ImageMath.unsafe_eval(expression, value=9) == 10


@pytest.mark.parametrize("name", ["__builtins__", "open", "abs"])
def test_operand_names_restricted(name: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        ImageMath.eval("1", **{name: 1})


@pytest.mark.parametrize("mode", ["I", "F"])
def test_closed_image(mode: str) -> None:
    image = Image.new(mode, (1, 1))
    image.close()
    with pytest.raises(ValueError):
        ImageMath.eval("a + 1", a=image)


@pytest.mark.parametrize("value", [17, 1.0000000596046448, -0.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "expression",
    ["a + n", "n + a", "a - n", "n - a", "a * n", "n * a", "a / n", "n / a", "a % n", "n % a", "a ** n", "n ** a", "a < n", "n < a", "min(a, n)", "max(a, n)", "equal(a, n)", "notequal(a, n)"],
)
def test_float_scalars_without_constant_images(expression: str, value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    a, pa = pair("F", (17, 1))
    expected = PILMath.unsafe_eval(expression, a=pa, n=value)

    def constant_image(*args: object, **kwargs: object) -> None:
        raise AssertionError("float scalar math must not allocate a constant image")

    monkeypatch.setattr(Image, "new", constant_image)
    same(ImageMath.unsafe_eval(expression, a=a, n=value), expected)


@pytest.mark.parametrize("value", [1e100, -1e100, 10**400])
def test_float_scalar_overflow(value: float) -> None:
    image = Image.new("F", (5, 1))
    with pytest.raises(OverflowError):
        ImageMath.unsafe_eval("a + n", a=image, n=value)


@pytest.mark.parametrize("mode", ["1", "L", "I"])
@pytest.mark.parametrize("value", [0, -1, -2147483648, 2147483647, 2**63, 10**100])
@pytest.mark.parametrize("expression", ["a + n", "n - a", "a * n", "a / n", "n % a", "a & n", "a << n", "a > n"])
def test_integer_scalars_without_constant_images(mode: str, value: int, expression: str, monkeypatch: pytest.MonkeyPatch) -> None:
    image, reference = pair(mode, (17, 1))
    wrapped = ((value + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    expected = PILMath.unsafe_eval(expression, a=reference, n=PILImage.new("I", reference.size, wrapped))

    def constant_image(*args: object, **kwargs: object) -> None:
        raise AssertionError("integer scalar math must not allocate a constant image")

    monkeypatch.setattr(Image, "new", constant_image)
    same(ImageMath.unsafe_eval(expression, a=image, n=value), expected)


def test_integer_scalar_subclasses_use_their_index_value() -> None:
    class Scalar(int):
        def __add__(self, other: object) -> int:
            raise AssertionError("scalar coercion must not invoke arithmetic overrides")

    image = Image.new("I", (3, 1), 17)
    assert ImageMath.unsafe_eval("a + n", a=image, n=Scalar(3)).getdata() == [20, 20, 20]
    assert ImageMath.unsafe_eval("n - a", a=image, n=Scalar(3)).getdata() == [-14, -14, -14]


@pytest.mark.parametrize("size", [(0, 2), (2, 0), (257, 259), (1025, 1025)])
@pytest.mark.parametrize("mode", ["I", "F"])
def test_empty_and_parallel_buffers(size: tuple[int, int], mode: str) -> None:
    a, pa = pair(mode, size)
    original = a.tobytes()
    same(ImageMath.eval("(a * 3 - 2) / 2", a=a), PILMath.unsafe_eval("(a * 3 - 2) / 2", a=pa))
    assert a.tobytes() == original


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 16387, 262147])
def test_compound_integer_float_rounding_and_parallel_tails(count: int) -> None:
    values = [-2147483648, 2147483647, -16777217, 16777217, -1, 0, 1, 254, 255, 256]
    values = (values * ((count + len(values) - 1) // len(values)))[:count]
    raw = struct.pack(f"<{count}i", *values)
    a = Image.frombytes("I", (count, 1), raw)
    pa = PILImage.frombytes("I", (count, 1), raw)
    b = Image.frombytes("I", (count, 1), raw[::-1])
    pb = PILImage.frombytes("I", (count, 1), raw[::-1])
    expression = "convert((float(a) + float(b)) / 2, 'L')"

    def evaluate(args):
        return args["convert"]((args["float"](args["a"]) + args["float"](args["b"])) / 2, "L")

    expected = PILMath.unsafe_eval(expression, a=pa, b=pb)
    assert ImageMath.unsafe_eval(expression, a=a, b=b).tobytes() == expected.tobytes()
    assert ImageMath.lambda_eval(evaluate, a=a, b=b).tobytes() == expected.tobytes()
    assert a.tobytes() == raw


def test_integer_boundaries_and_zero_divisors() -> None:
    image = Image.new("I", (3, 1))
    image.putdata([-2147483648, 2147483647, -7])
    assert list(ImageMath.eval("a + 1", a=image).getdata()) == [-2147483647, -2147483648, -6]
    assert list(ImageMath.eval("a / -1", a=image).getdata()) == [-2147483648, -2147483647, 7]
    assert list(ImageMath.eval("a / 0", a=image).getdata()) == [0, 0, 0]
    assert list(ImageMath.eval("a % 0", a=image).getdata()) == [0, 0, 0]


@pytest.mark.parametrize("mode", ["1", "L", "I", "F"])
@pytest.mark.parametrize("sizes", [((257, 259), (257, 259)), ((263, 259), (257, 263)), ((257, 263), (263, 259)), ((17003, 5), (17001, 4))])
@pytest.mark.parametrize("expression", ["-a", "a + b", "a / b", "a < b", "min(a, b)", "equal(a, b)"])
def test_chunk_boundaries_and_operand_strides(mode: str, sizes: tuple[tuple[int, int], tuple[int, int]], expression: str) -> None:
    a, pa = pair(mode, sizes[0])
    b, pb = pair(mode, sizes[1])
    same(ImageMath.eval(expression, a=a, b=b), PILMath.unsafe_eval(expression, a=pa, b=pb))


@pytest.mark.parametrize("expression", ["-a", "abs(a)", "a + b", "a / b", "min(a, b)", "max(a, b)", "a == b", "equal(a, b)"])
def test_float_special_values(expression: str) -> None:
    values = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0, -1.0]
    pa = PILImage.frombytes("F", (7, 1), struct.pack("<7f", *values))
    pb = PILImage.frombytes("F", (7, 1), struct.pack("<7f", *reversed(values)))
    a = Image.frombytes("F", pa.size, pa.tobytes())
    b = Image.frombytes("F", pb.size, pb.tobytes())
    same(ImageMath.eval(expression, a=a, b=b), PILMath.unsafe_eval(expression, a=pa, b=pb))
