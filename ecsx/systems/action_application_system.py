import jax.numpy as jnp
from ecsx.query.query_engine import select

def action_application_system(velocity_scale: float = 1.0):
    scale = jnp.array(velocity_scale, jnp.float32)

    def _system(world, key, inputs):
        # Entities with Velocity and Action; copy first two action dims into velocity (scaled)
        mask = select(world, required=("Velocity", "Action"))
        vel_store = world.component_stores["Velocity"]
        act_store = world.component_stores["Action"]

        vel = vel_store.data
        act = act_store.data[:, :2]
        new_vel = jnp.where(mask[:, None], act * scale, vel)
        vel_store2 = type(vel_store)(vel_store.spec, new_vel, vel_store.alive_mask)
        return world._with_store("Velocity", vel_store2)
    return _system

