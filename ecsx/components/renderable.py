import jax.numpy as jnp
from ecsx.core.component_specification import ComponentSpecification


def get_renderable_specification() -> ComponentSpecification:
    """Specification for sprite index used for rendering.

    Each entity references a texture loaded by the rendering system.  The
    component stores an integer ``texture_id`` selecting which texture to draw
    for the entity.  ``0`` is used as the default sprite.
    """
    return ComponentSpecification(
        "Renderable",
        (),
        jnp.int32,
        jnp.array(0, jnp.int32),
    )
