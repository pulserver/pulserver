"""Private MRD adapter for gradient-coefficient transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MrdCoefficientAccessor:
    """Read serialized coefficients from a keyed MRD XML string parameter.

    Both the camelCase attributes produced by ``ismrmrd.xsd`` and the
    snake_case attributes used by newer ``mrd`` bindings are supported.
    """

    header: Any = field(repr=False)
    key: str

    @property
    def name(self) -> str:
        """Return a non-sensitive source label for diagnostics."""
        return f"MRD userParameterString {self.key!r}"

    def read_coefficients(self) -> str:
        """Return the coefficient payload or raise if ``key`` is absent."""
        header = getattr(self.header, "header", self.header)
        parameters = getattr(
            header,
            "userParameters",
            getattr(header, "user_parameters", None),
        )
        if parameters is not None:
            values = getattr(
                parameters,
                "userParameterString",
                getattr(parameters, "user_parameter_string", ()),
            )
            for parameter in values or ():
                name = getattr(parameter, "name", None)
                if name == self.key:
                    value = getattr(parameter, "value", None)
                    if not isinstance(value, str):
                        raise TypeError(
                            f"MRD string parameter {self.key!r} has non-string value "
                            f"{type(value).__name__}."
                        )
                    return value
        raise KeyError(f"MRD userParameterString {self.key!r} was not found.")
