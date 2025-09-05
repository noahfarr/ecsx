import jax.numpy as jnp
from ecsx.query.query_engine import select

def grid_reward_system(reward_value: float = 1.0):
    r = jnp.array(reward_value, jnp.float32)

    def _system(world, key, inputs):
        gmask = select(world, required=("GridGoal", "GridPosition"))     # (cap,)
        amask = select(world, required=("Reward", "GridPosition"))       # (cap,)
        if (~(jnp.any(gmask) & jnp.any(amask))):
            return world
        pos = world.component_stores["GridPosition"].data                # (cap,2)
        goals = jnp.any(jnp.all(pos[:, None, :] == pos[None, :, :], axis=-1) & gmask[None, :], axis=1)  # (cap,)
        hit_agents = amask & goals                                       # (cap,)
        reward_store = world.component_stores["Reward"]
        reward = reward_store.data[:, 0]
        reward = jnp.where(hit_agents, reward + r, reward)
        new_store = type(reward_store)(reward_store.specification,
                                       reward_store.data.at[:, 0].set(reward),
                                       reward_store.present_mask)
        world = world._with_store("Reward", new_store)
        return world
    return _system
