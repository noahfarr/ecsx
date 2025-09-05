import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_reward_specification() -> ComponentSpecification:
    return ComponentSpecification("Reward", (1,), jnp.float32, jnp.array([0.0], jnp.float32))

