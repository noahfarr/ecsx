from typing import Callable, Mapping, Tuple
from dataclasses import replace
import jax
from jax import random, jit

from ..core.typing import Array, Key
from ..core.world_state import WorldState
from ..core.static_world import StaticWorld
from .system_base import SystemFunction, set_random_key


def build_step(systems: Tuple[SystemFunction, ...]) -> Callable[[WorldState, Mapping[str, Array]], WorldState]:
    """
    Compose systems into a single step. For now, runs eagerly (not jitted) to play
    nicely with Python dicts in WorldState. Deterministic PRNG splitting.
    """
    def step(world: WorldState, inputs: Mapping[str, Array]) -> WorldState:
        # reset per-step events
        world = world.reset_event_buffers()
        key = world.random_key
        for sys in systems:
            key, sub = random.split(key)
            world = sys(set_random_key(world, sub), sub, inputs)
        return replace(world, random_key=key, time_step=world.time_step + jax.numpy.array(1, world.time_step.dtype))
    return step

def build_step_jitted(systems):
    step = build_step(systems)
    return jax.jit(step)  # donate WorldState

def build_step_unrolled(systems, unroll: int = 8):
    step = build_step(systems)
    def step_unrolled(world, inputs):
        def body(w, _):
            return step(w, inputs), None
        world, _ = jax.lax.scan(body, world, None, length=unroll)
        return world
    return jax.jit(step_unrolled)  # donate WorldState

def build_step_static(systems, jit=True) -> Callable[[StaticWorld, Mapping[str, Array]], StaticWorld]:
    def step(sw: StaticWorld, inputs: Mapping[str, Array]) -> StaticWorld:
        sw = sw.reset_event_buffers()
        key = sw.random_key
        for sys in systems:
            key, sub = random.split(key)
            sw = sys(set_random_key(sw, sub), sub, inputs)
        return replace(sw, random_key=key, time_step=sw.time_step + 1)
    return jax.jit(step) if jit else step
