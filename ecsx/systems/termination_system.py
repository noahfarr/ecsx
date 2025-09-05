import jax.numpy as jnp
from ecsx.query.query_engine import select

def termination_system(bound_xy: float = 1.0):
    limit = jnp.array(bound_xy, jnp.float32)

    def _system(world, key, inputs):
        # Requires Position + Termination
        mask = select(world, required=("Position", "Termination"))
        idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
        count = jnp.sum(mask)
        if int(count) == 0:
            return world

        pos = world.component_stores["Position"].read(idx[:count])
        outside = jnp.logical_or(jnp.abs(pos[:, 0]) > limit, jnp.abs(pos[:, 1]) > limit)
        term_store = world.component_stores["Termination"]
        flags = term_store.read(idx[:count])
        flags = flags.at[:, 0].set(outside)
        term_store = term_store.write(idx[:count], flags)
        world = world._with_store("Termination", term_store)
        return world
    return _system

