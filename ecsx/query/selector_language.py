from dataclasses import dataclass
from typing import FrozenSet

import jax.numpy as jnp

from ..core.world_state import WorldState
from ..core.static_world import StaticWorld
from .query_engine import select as _select


@dataclass(frozen=True)
class Selector:
    required: FrozenSet[str]
    forbidden: FrozenSet[str]

    def __and__(self, other: "Selector") -> "Selector":
        return Selector(
            self.required | other.required, self.forbidden | other.forbidden
        )

    def __sub__(self, other: "Selector") -> "Selector":
        # treat subtraction as "forbid" other's required
        return Selector(self.required, self.forbidden | other.required)

    def compile(self):
        """Return a pure function (world -> mask) usable inside jit tracing (keys are static)."""
        req = tuple(sorted(self.required))
        forb = tuple(sorted(self.forbidden))

        def _fn(world: WorldState) -> jnp.ndarray:
            return _select(world, required=req, forbidden=forb)

        return _fn


def Has(name: str) -> Selector:
    return Selector(required=frozenset([name]), forbidden=frozenset())


def AllOf(*names: str) -> Selector:
    return Selector(required=frozenset(names), forbidden=frozenset())


def Without(*names: str) -> Selector:
    return Selector(required=frozenset(), forbidden=frozenset(names))


class CompilableMixin:
    def compile_for_static(self, static_names: tuple[str, ...]):
        from .query_engine import compile_selector_for_static
        return compile_selector_for_static(self.required, self.forbidden, static_names)
