import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification


def get_discrete_action_specification() -> ComponentSpecification:
    return ComponentSpecification(
        "DiscreteAction", (), jnp.int32, jnp.array(0, jnp.int32)
    )
