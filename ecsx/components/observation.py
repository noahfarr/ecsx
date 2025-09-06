import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification


def get_observation_specification(
    shape: tuple[int, ...] = (2,),
    dtype: jnp.dtype = jnp.float32,
) -> ComponentSpecification:
    """Create an observation component specification.

    Parameters
    ----------
    shape:
        Shape of the observation array. Defaults to ``(2,)`` for agent position.
    dtype:
        Array data type. Defaults to ``jnp.float32``.
    """

    return ComponentSpecification(
        "Observation", shape, dtype, jnp.zeros(shape, dtype)
    )
