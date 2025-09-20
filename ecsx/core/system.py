from dataclasses import replace
from typing import Callable, Mapping, Protocol, Tuple, TYPE_CHECKING

import jax

from .typing import Array, Key

if TYPE_CHECKING:  # pragma: no cover
    from .world import WorldState


class System(Protocol):
    """Callable transforming the world state.

    Systems operate purely: they must not mutate the input ``WorldState`` in
    place and instead return a new ``WorldState`` instance.
    """

    def __call__(
        self, world: "WorldState", key: Key, inputs: Mapping[str, Array]
    ) -> "WorldState": ...


def set_random_key(world: "WorldState", key: Key) -> "WorldState":
    """Return ``world`` with its random key replaced by ``key``."""

    return replace(world, random_key=key)


def build_step(
    systems: Tuple[System, ...]
) -> Callable[["WorldState", Mapping[str, Array]], "WorldState"]:
    """Compose a sequence of systems into a single step function.

    Each step resets per-frame event buffers, deterministically splits the
    world's PRNG key for each system, and increments the global ``time_step``.
    """

    def step(world: "WorldState", inputs: Mapping[str, Array]) -> "WorldState":
        world = world.reset_event_buffers()
        key = world.random_key
        for system in systems:
            key, sub = jax.random.split(key)
            world = system(set_random_key(world, sub), sub, inputs)
        return replace(
            world,
            random_key=key,
            time_step=world.time_step + jax.numpy.array(1, world.time_step.dtype),
        )

    return step


if TYPE_CHECKING:  # pragma: no cover
    from .world import WorldState
