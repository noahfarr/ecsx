import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification


def get_flag_specification() -> ComponentSpecification:
    """Component representing a team's flag.

    Fields are packed into an ``int32`` vector of length 3::

        [owner_team, is_carried, carrier_id]

    ``owner_team``: team id the flag belongs to.
    ``is_carried``: 1 if the flag is carried by an agent, else 0.
    ``carrier_id``: entity id of the carrier or -1 if not carried.
    """

    return ComponentSpecification(
        "Flag",
        (3,),
        jnp.int32,
        jnp.array([0, 0, -1], jnp.int32),
    )

