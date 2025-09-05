import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification

def get_agent_tag_specification() -> ComponentSpecification:
    return ComponentSpecification("AgentTag", (), jnp.bool_, jnp.array(False))

