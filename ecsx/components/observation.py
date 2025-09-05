import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_observation_specification(observation_dimension: int) -> ComponentSpecification:
    return ComponentSpecification("Observation", (observation_dimension,), jnp.float32, jnp.zeros((observation_dimension,), jnp.float32))

