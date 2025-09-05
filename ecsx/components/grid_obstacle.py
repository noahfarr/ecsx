import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_grid_obstacle_specification() -> ComponentSpecification:
    # tag; presence=obstacle
    return ComponentSpecification("GridObstacle", (), jnp.bool_, jnp.array(False))

