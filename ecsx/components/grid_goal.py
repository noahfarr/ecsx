import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_grid_goal_specification() -> ComponentSpecification:
    # tag; presence=goal
    return ComponentSpecification("GridGoal", (), jnp.bool_, jnp.array(False))

