import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification


def get_obstacle_specification() -> ComponentSpecification:
    return ComponentSpecification(
        "Obstacle", (), jnp.bool_, jnp.array(False, jnp.bool_)
    )
