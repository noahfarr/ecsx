import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_grid_position_specification() -> ComponentSpecification:
    # integer grid coordinates (x, y)
    return ComponentSpecification("GridPosition", (2,), jnp.int32, jnp.array([0, 0], jnp.int32))

