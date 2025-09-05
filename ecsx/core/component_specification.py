from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class ComponentSpecification:
    name: str
    shape: tuple[int, ...]
    dtype: jnp.dtype
    default: Any

    def validate_value(self, value) -> jnp.ndarray:
        arr = jnp.asarray(value, self.dtype)
        if arr.shape != self.shape:
            raise ValueError(f"{self.name}: got {arr.shape}, expected {self.shape}")
        return arr
