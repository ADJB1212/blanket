"""Image expressions with Rust pixel arithmetic for modes 1, L, I, and F."""

from __future__ import annotations

import builtins
from types import CodeType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from . import Image
from ._blanket import math_apply

__all__ = ["eval", "lambda_eval", "unsafe_eval"]


class _Operand:
    def __init__(self, image: Image.Image) -> None:
        self.im = image

    def _coerce(self, value: _Operand | float) -> Image.Image:
        if isinstance(value, _Operand):
            if value.im.mode in ("1", "L"):
                return value.im.convert("I")
            if value.im.mode in ("I", "F"):
                return value.im
            raise ValueError(f"unsupported mode: {value.im.mode}")
        mode = "I" if isinstance(value, (int, float)) and self.im.mode in ("1", "L", "I") else "F"
        return Image.new(mode, self.im.size, value)

    def _apply(self, operation: str, left: _Operand | float, right: _Operand | float | None = None, *, integer_output: bool = False) -> _Operand:
        first = self._coerce(left)
        second = None if right is None else self._coerce(right)
        if second is not None and first.mode != second.mode:
            first = first.convert("F")
            second = second.convert("F")
        first.load()
        if second is not None:
            second.load()
        return _Operand(Image.Image(math_apply(operation, first._native, None if second is None else second._native, integer_output)))

    def __bool__(self) -> bool:
        return self.im.getbbox() is not None

    def __pos__(self) -> _Operand:
        return self

    def __neg__(self) -> _Operand:
        return self._apply("neg", self)

    def __abs__(self) -> _Operand:
        return self._apply("abs", self)

    def __invert__(self) -> _Operand:
        return self._apply("invert", self)

    def __add__(self, other: _Operand | float) -> _Operand:
        return self._apply("add", self, other)

    def __radd__(self, other: float) -> _Operand:
        return self._apply("add", other, self)

    def __sub__(self, other: _Operand | float) -> _Operand:
        return self._apply("sub", self, other)

    def __rsub__(self, other: float) -> _Operand:
        return self._apply("sub", other, self)

    def __mul__(self, other: _Operand | float) -> _Operand:
        return self._apply("mul", self, other)

    def __rmul__(self, other: float) -> _Operand:
        return self._apply("mul", other, self)

    def __truediv__(self, other: _Operand | float) -> _Operand:
        return self._apply("div", self, other)

    def __rtruediv__(self, other: float) -> _Operand:
        return self._apply("div", other, self)

    def __mod__(self, other: _Operand | float) -> _Operand:
        return self._apply("mod", self, other)

    def __rmod__(self, other: float) -> _Operand:
        return self._apply("mod", other, self)

    def __pow__(self, other: _Operand | float) -> _Operand:
        return self._apply("pow", self, other)

    def __rpow__(self, other: float) -> _Operand:
        return self._apply("pow", other, self)

    def __and__(self, other: _Operand | float) -> _Operand:
        return self._apply("and", self, other)

    def __rand__(self, other: float) -> _Operand:
        return self._apply("and", other, self)

    def __or__(self, other: _Operand | float) -> _Operand:
        return self._apply("or", self, other)

    def __ror__(self, other: float) -> _Operand:
        return self._apply("or", other, self)

    def __xor__(self, other: _Operand | float) -> _Operand:
        return self._apply("xor", self, other)

    def __rxor__(self, other: float) -> _Operand:
        return self._apply("xor", other, self)

    def __lshift__(self, other: _Operand | float) -> _Operand:
        return self._apply("lshift", self, other)

    def __rshift__(self, other: _Operand | float) -> _Operand:
        return self._apply("rshift", self, other)

    def __eq__(self, other: object) -> _Operand:  # ty: ignore[invalid-method-override]
        return self._apply("eq", self, other)

    def __ne__(self, other: object) -> _Operand:  # ty: ignore[invalid-method-override]
        return self._apply("ne", self, other)

    def __lt__(self, other: _Operand | float) -> _Operand:
        return self._apply("lt", self, other)

    def __le__(self, other: _Operand | float) -> _Operand:
        return self._apply("le", self, other)

    def __gt__(self, other: _Operand | float) -> _Operand:
        return self._apply("gt", self, other)

    def __ge__(self, other: _Operand | float) -> _Operand:
        return self._apply("ge", self, other)


def _convert(value: _Operand, mode: str) -> _Operand:
    return _Operand(value.im.convert(mode))


def _context(operands: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "int": lambda value: _convert(value, "I"),
        "float": lambda value: _convert(value, "F"),
        "convert": _convert,
        "min": lambda a, b: a._apply("min", a, b),
        "max": lambda a, b: a._apply("max", a, b),
        "equal": lambda a, b: a._apply("eq", a, b, integer_output=True),
        "notequal": lambda a, b: a._apply("ne", a, b, integer_output=True),
    }
    context.update(operands)
    return {name: _Operand(value) if isinstance(value, Image.Image) else value for name, value in context.items()}


def _unwrap(value: Any) -> Any:
    return value.im if isinstance(value, _Operand) else value


def lambda_eval(expression: Callable[[dict[str, Any]], Any], **operands: Any) -> Any:
    """Evaluate a callable receiving a dictionary of images, values, and helpers.

    Helpers are int, float, convert, min, max, equal, and notequal. Image
    arithmetic runs in Rust and returns I or F images. Use split and merge
    to process multiband images.
    """
    return _unwrap(expression(_context(operands)))


def unsafe_eval(expression: str, **operands: Any) -> Any:
    """Evaluate a trusted Python expression using native image arithmetic.

    This calls Python eval and is not a sandbox for untrusted expressions
    or operands. Prefer lambda_eval for callable expressions.
    """
    for name in operands:
        if "__" in name or hasattr(builtins, name):
            raise ValueError(f"'{name}' not allowed")
    context = _context(operands)
    code = compile(expression, "<ImageMath>", "eval")

    def validate(compiled: CodeType) -> None:
        for name in compiled.co_names:
            if name not in context and name != "abs":
                raise ValueError(f"'{name}' not allowed")
        for constant in compiled.co_consts:
            if isinstance(constant, CodeType):
                validate(constant)

    validate(code)
    return _unwrap(builtins.eval(code, {"__builtins__": {"abs": abs}}, context))


def eval(expression: str, **operands: Any) -> Any:
    """Compatibility alias for unsafe_eval; evaluate only trusted expressions."""
    return unsafe_eval(expression, **operands)
