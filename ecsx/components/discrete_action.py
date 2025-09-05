import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_discrete_action_specification(num_actions: int = 5) -> ComponentSpecification:
    # stores integer action id in [0..num_actions-1]; default 0 = no-op
    return ComponentSpecification("DiscreteAction", (1,), jnp.int32, jnp.array([0], jnp.int32))

