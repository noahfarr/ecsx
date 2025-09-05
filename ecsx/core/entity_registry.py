from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp


@dataclass
class EntityRegistry:
    """
    Host-side free-list allocator for entity ids.
    Not used under jit; spawning/despawning happens on the host.
    """

    capacity: int
    alive_mask: jnp.ndarray
    _free_stack: np.ndarray
    _stack_top: int

    @staticmethod
    def create(capacity: int) -> "EntityRegistry":
        alive_mask = jnp.zeros((capacity,), dtype=bool)
        free_stack = np.arange(capacity - 1, -1, -1, dtype=np.int32)
        return EntityRegistry(capacity, alive_mask, free_stack, capacity)

    def spawn(self) -> int:
        if self._stack_top == 0:
            raise RuntimeError("No free entity ids available")
        self._stack_top -= 1
        eid = int(self._free_stack[self._stack_top])
        self.alive_mask = self.alive_mask.at[eid].set(True)
        return eid

    def despawn(self, entity_id: int) -> None:
        if not bool(self.alive_mask[entity_id]):
            return
        self.alive_mask = self.alive_mask.at[entity_id].set(False)
        self._free_stack[self._stack_top] = entity_id
        self._stack_top += 1
