import jax.numpy as jnp
from ecsx.query.query_engine import select


def observation_system(concatenate_position_velocity: bool = True):
    """
    Writes Observation for entities with Observation (+ Position, Velocity when requested).
    Masked updates only — JIT friendly (no indices / Python ints).
    """
    def _system(world, key, inputs):
        if not concatenate_position_velocity:
            return world

        mask = select(world, required=("Observation", "Position", "Velocity"))  # (capacity,)
        obs_store = world.component_stores["Observation"]
        pos = world.component_stores["Position"].data                              # (capacity, 2)
        vel = world.component_stores["Velocity"].data                              # (capacity, 2)

        new_obs_full = jnp.concatenate([pos, vel], axis=1)                         # (capacity, 4)
        # Match the observation spec length (robust if it's not exactly 4)
        target_dim = obs_store.spec.shape[0]
        new_obs = new_obs_full[:, :target_dim]

        updated = jnp.where(mask[:, None], new_obs, obs_store.data)                # (capacity, target_dim)
        obs_store2 = type(obs_store)(obs_store.spec, updated, obs_store.alive_mask)
        return world._with_store("Observation", obs_store2)
    return _system
