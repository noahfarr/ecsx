import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification


def get_team_specification() -> ComponentSpecification:
    """Component storing an entity's team affiliation."""

    return ComponentSpecification(
        "Team", (), jnp.int32, jnp.array(0, jnp.int32)
    )

