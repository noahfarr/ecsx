import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_action_specification(max_action_dimension: int) -> ComponentSpecification:
    return ComponentSpecification("Action", (max_action_dimension,), jnp.float32, jnp.zeros((max_action_dimension,), jnp.float32))

