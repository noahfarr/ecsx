from typing import Mapping, Protocol, Tuple, Dict, Any
from dataclasses import replace

from jax import random

from ..core.typing import Array, Key
from ..core.world_state import WorldState


class SystemFunction(Protocol):
    def __call__(self, world: WorldState, key: Key, inputs: Mapping[str, Array]) -> WorldState:
        """Pure transform: must not mutate `world` in-place."""


def split_key(world: WorldState) -> Tuple[WorldState, Key]:
    k, sub = random.split(world.random_key)
    return replace(world, random_key=k), sub


def set_random_key(world: WorldState, key: Key) -> WorldState:
    return replace(world, random_key=key)

