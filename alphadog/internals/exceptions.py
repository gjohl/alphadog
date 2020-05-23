"""
Package-specific exceptions
"""


class ParameterError(ValueError):
    pass


class InputDataError(Exception):
    pass


class DimensionMismatchError(Exception):
    pass
