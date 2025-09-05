from dataclasses import dataclass
from typing import Tuple
import jax.numpy as jnp

@dataclass(frozen=True)
class EventSpecification:
    name: str
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    capacity: int

