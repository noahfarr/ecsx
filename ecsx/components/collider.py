import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_collider_specification() -> ComponentSpecification:
    return ComponentSpecification("Collider", (1,), jnp.float32, jnp.array([0.0], jnp.float32))  # radius

