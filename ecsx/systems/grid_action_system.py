import jax.numpy as jnp
from ecsx.query.query_engine import select

# Actions: 0=stay, 1=up, 2=right, 3=down, 4=left
_ACTION_DELTAS = jnp.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]], jnp.int32)

def grid_action_system(grid_width: int, grid_height: int):
    W = jnp.array(grid_width, jnp.int32)
    H = jnp.array(grid_height, jnp.int32)

    def _system(world, key, inputs):
        mask = select(world, required=("GridPosition", "DiscreteAction"))
        idx  = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
        cnt  = jnp.sum(mask)
        if int(cnt) == 0:
            return world

        pos = world.component_stores["GridPosition"].read(idx[:cnt])      # (cnt,2) int32
        act = world.component_stores["DiscreteAction"].read(idx[:cnt])    # (cnt,1) int32
        delta = _ACTION_DELTAS[act[:, 0]]                                  # (cnt,2)

        # propose move
        new_pos = pos + delta
        # clamp to bounds
        new_pos = jnp.stack([jnp.clip(new_pos[:, 0], 0, W - 1),
                             jnp.clip(new_pos[:, 1], 0, H - 1)], axis=1)

        # block obstacles
        obs_mask = select(world, required=("GridObstacle", "GridPosition"))
        obs_idx  = jnp.nonzero(obs_mask, size=world.capacity, fill_value=-1)[0]
        m        = jnp.sum(obs_mask)
        if int(m) > 0:
            obs_pos = world.component_stores["GridPosition"].read(obs_idx[:m])  # (m,2)
            # blocked[i] = any(new_pos[i] == any obstacle)
            eq = jnp.all(new_pos[:, None, :] == obs_pos[None, :, :], axis=-1)   # (cnt,m)
            blocked = jnp.any(eq, axis=1)                                       # (cnt,)
            new_pos = jnp.where(blocked[:, None], pos, new_pos)

        store = world.component_stores["GridPosition"].write(idx[:cnt], new_pos)
        world = world._with_store("GridPosition", store)
        return world

    return _system

