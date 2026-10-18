"""[EL-ARCH-1 A5] A required component that is missing is an error, never a silent degradation."""
from __future__ import annotations


class MissingComponentError(RuntimeError):
    def __init__(self, component: str, why: str = ""):
        super().__init__("required component missing: %s%s" % (component, (" (" + why + ")") if why else ""))
        self.component, self.why = component, why


def require(component: str, obj, why: str = ""):
    if obj is None:
        raise MissingComponentError(component, why)
    return obj
