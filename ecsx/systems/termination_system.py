import jax.numpy as jnp
from ecsx.query.query_engine import select

def termination_system(bound_xy: float = 1.0):
    limit = jnp.array(bound_xy, jnp.float32)
    def _system(world, key, inputs):
        mask = select(world, required=("Position", "Termination"))          # (cap,)
        pos = world.component_stores["Position"].data                        # (cap,2)
        term = world.component_stores["Termination"]
        cur  = term.data                                                     # (cap,1)
        outside = (jnp.abs(pos[:, 0]) > limit) | (jnp.abs(pos[:, 1]) > limit)
        new = jnp.where(mask[:, None], outside[:, None], cur)
        term2 = type(term)(term.specification, new, term.present_mask)
        return world._with_store("Termination", term2)
    return _system
