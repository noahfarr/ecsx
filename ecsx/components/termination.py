import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification


def get_termination_specification() -> ComponentSpecification:
    return ComponentSpecification(
        "Termination", (), jnp.bool_, jnp.array(False, jnp.bool_)
    )
