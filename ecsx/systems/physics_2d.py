import jax.numpy as jnp
from ecsx.query.query_engine import select

def physics_2d(delta_time: float = 0.02):
    dt = jnp.array(delta_time, jnp.float32)

    def _system(world, key, inputs):
        # Masked: pos := pos + vel*dt where both Position & Velocity alive.
        mask = select(world, required=("Position", "Velocity"))        # (cap,)
        pos_store = world.component_stores["Position"]
        vel_store = world.component_stores["Velocity"]
        pos = pos_store.data                                           # (cap,2)
        vel = vel_store.data                                           # (cap,2)
        new_pos = jnp.where(mask[:, None], pos + vel * dt, pos)        # (cap,2)
        pos_store2 = type(pos_store)(pos_store.spec, new_pos, pos_store.alive_mask)
        return world._with_store("Position", pos_store2)
    return _system

