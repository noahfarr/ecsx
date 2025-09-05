import jax.numpy as jnp
from ecsx.query.query_engine import select

def observation_system(concatenate_position_velocity: bool = True):
    """
    Writes Observation.values for entities that have Observation (and optionally Position+Velocity).
    If concatenate_position_velocity=True, Observation := concat([Position, Velocity]) assuming shape 4.
    """
    def _system(world, key, inputs):
        if concatenate_position_velocity:
            mask = select(world, required=("Observation", "Position", "Velocity"))
            idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
            count = jnp.sum(mask)
            if int(count) == 0:
                return world
            pos = world.component_stores["Position"].read(idx[:count])
            vel = world.component_stores["Velocity"].read(idx[:count])
            obs = jnp.concatenate([pos, vel], axis=1)
            store = world.component_stores["Observation"].write(idx[:count], obs)
            world = world._with_store("Observation", store)
        return world
    return _system

