import jax.numpy as jnp
from ecsx.query.query_engine import select

def grid_reward_system(reward_value: float = 1.0):
    r = jnp.array(reward_value, jnp.float32)

    def _system(world, key, inputs):
        # positions of goals
        gmask = select(world, required=("GridGoal", "GridPosition"))
        gidx  = jnp.nonzero(gmask, size=world.capacity, fill_value=-1)[0]
        gcnt  = jnp.sum(gmask)
        if int(gcnt) == 0:
            return world
        gpos  = world.component_stores["GridPosition"].read(gidx[:gcnt])  # (gcnt,2)

        # agents eligible: have Reward + GridPosition
        amask = select(world, required=("Reward", "GridPosition"))
        aidx  = jnp.nonzero(amask, size=world.capacity, fill_value=-1)[0]
        acnt  = jnp.sum(amask)
        if int(acnt) == 0:
            return world
        apos  = world.component_stores["GridPosition"].read(aidx[:acnt])  # (acnt,2)

        # hit[i] = any(apos[i] == any goal)
        hit = jnp.any(jnp.all(apos[:, None, :] == gpos[None, :, :], axis=-1), axis=1)  # (acnt,)

        if not bool(jnp.any(hit)):
            return world

        # scatter-add reward to agents that hit a goal
        reward_store = world.component_stores["Reward"]
        reward = reward_store.data[:, 0]
        inc = jnp.where(hit, r, 0.0)  # (acnt,)
        reward = reward.at[aidx[:acnt]].add(inc)
        new_store = type(reward_store)(
            reward_store.spec,
            reward_store.data.at[:, 0].set(reward),
            reward_store.alive_mask,
        )
        world = world._with_store("Reward", new_store)
        return world

    return _system

