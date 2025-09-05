import jax.numpy as jnp
from ecsx.query.query_engine import select

def physics_2d(delta_time: float = 0.02):
    dt = jnp.array(delta_time, jnp.float32)

    def _system(world, key, inputs):
        mask = select(world, required=("Position", "Velocity"))
        idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
        count = jnp.sum(mask)

        pos = world.component_stores["Position"].read(idx[:count])
        vel = world.component_stores["Velocity"].read(idx[:count])
        new_pos = pos + vel * dt

        store = world.component_stores["Position"].write(idx[:count], new_pos)
        world = world._with_store("Position", store)
        return world
    return _system

