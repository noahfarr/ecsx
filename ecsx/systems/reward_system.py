import jax.numpy as jnp

def reward_system(collision_event_name: str = "Collision"):
    """
    For each collision pair (a,b):
      - if a has RewardOnTouch and b has Reward, add a.value to b.Reward
      - if b has RewardOnTouch and a has Reward, add b.value to a.Reward
    """
    def _system(world, key, inputs):
        buf = world.event_buffers[collision_event_name]
        count = int(buf.count)
        if count == 0:
            return world
        pairs = buf.data[:count]  # (count, 2), entity ids (int32)

        # masks (alive per entity)
        has_reward = world.component_stores["Reward"].alive_mask
        has_rot = world.component_stores["RewardOnTouch"].alive_mask
        rot_values = world.component_stores["RewardOnTouch"].data[:, 0]  # (capacity,)

        a = pairs[:, 0]; b = pairs[:, 1]

        # a -> b
        cond_ab = jnp.logical_and(has_rot[a], has_reward[b])
        inc_ab = rot_values[a] * cond_ab.astype(jnp.float32)

        # b -> a
        cond_ba = jnp.logical_and(has_rot[b], has_reward[a])
        inc_ba = rot_values[b] * cond_ba.astype(jnp.float32)

        # scatter-add
        reward_store = world.component_stores["Reward"]
        reward = reward_store.data[:, 0]
        reward = reward.at[b].add(inc_ab)
        reward = reward.at[a].add(inc_ba)

        new_store = type(reward_store)(reward_store.spec,
                                       reward_store.data.at[:, 0].set(reward),
                                       reward_store.alive_mask)
        world = world._with_store("Reward", new_store)
        return world
    return _system

