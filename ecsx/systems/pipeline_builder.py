from typing import Callable, Mapping, Tuple
from dataclasses import replace
import jax
from jax import random

from ..core.typing import Array, Key
from ..core.world_state import WorldState
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

