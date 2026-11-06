from collections.abc import Sequence
from math import isfinite


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def real(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    if lower is not None and number < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and number > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return number


def sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def choice(value, name, options):
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of {', '.join(options)}")
    return value
