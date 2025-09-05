import jax.numpy as jnp
from ecsx.query.query_engine import select

def action_application_system(velocity_scale: float = 1.0):
    scale = jnp.array(velocity_scale, jnp.float32)

    def _system(world, key, inputs):
        # Entities with Velocity and Action; copy first two action dims into velocity (scaled)
        mask = select(world, required=("Velocity", "Action"))
        idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
        count = jnp.sum(mask)
        if int(count) == 0:
            return world

        actions = world.component_stores["Action"].read(idx[:count])[:, :2]
        new_vel = actions * scale
        vel_store = world.component_stores["Velocity"].write(idx[:count], new_vel)
        world = world._with_store("Velocity", vel_store)
        return world
    return _system

