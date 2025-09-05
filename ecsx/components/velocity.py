import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_velocity_specification() -> ComponentSpecification:
    return ComponentSpecification("Velocity", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32))

