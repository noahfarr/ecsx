from __future__ import annotations
from typing import Callable, Mapping, Tuple
import jax
from ..core.typing import Array
from ..core.static_world import StaticWorld
from .pipeline_builder import build_step_static

def build_batched_step_static(systems: Tuple[Callable, ...], unroll: int = 1):
    """Return a jitted function stepping a batch of StaticWorlds in one call."""
    step_one = build_step_static(systems, jit=False)  # fn(StaticWorld, inputs)->StaticWorld
    def step_unrolled(sw: StaticWorld, inputs: Mapping[str, Array]) -> StaticWorld:
        def body(w, _):
            return step_one(w, inputs), None
        sw, _ = jax.lax.scan(body, sw, None, length=unroll)
        return sw
    step_many = jax.vmap(step_unrolled, in_axes=(0, None), out_axes=0)
    return jax.jit(step_many)

