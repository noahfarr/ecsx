import jax.numpy as jnp
from ecsx.query.query_engine import select

def grid_observation_system(grid_width: int, grid_height: int):
    W = jnp.array(grid_width, jnp.float32)
    H = jnp.array(grid_height, jnp.float32)

    def _system(world, key, inputs):
        # single-goal assumption; compute goal_xy without host branching
        gmask = select(world, required=("GridGoal", "GridPosition"))  # (cap,)
        gpos_all = world.component_stores["GridPosition"].data        # (cap,2) int32
        has_any = jnp.any(gmask)
        # take first goal by masking-and-argmax over a sentinel score
        scores = jnp.where(gmask, jnp.arange(gmask.shape[0], dtype=jnp.int32), -1)
        idx = jnp.argmax(scores)                                      # 0 if none; guarded below
        goal_xy = jnp.where(has_any, gpos_all[idx], jnp.array([0, 0], jnp.int32))
 
        # For all with Observation + GridPosition, write [ax/W, ay/H, gx/W, gy/H]
        mask = select(world, required=("Observation", "GridPosition"))  # (cap,)
        apos = gpos_all.astype(jnp.float32)                             # (cap,2)
        obs_store = world.component_stores["Observation"]
        target_dim = obs_store.specification.shape[0]
 
        base = jnp.stack([apos[:, 0] / W, apos[:, 1] / H,
                          jnp.full((apos.shape[0],), goal_xy[0] / W),
                          jnp.full((apos.shape[0],), goal_xy[1] / H)], axis=1)  # (cap,4)
        new_obs = base[:, :target_dim]
        updated = jnp.where(mask[:, None], new_obs, obs_store.data)
        obs2 = type(obs_store)(obs_store.specification, updated, obs_store.present_mask)
        world = world._with_store("Observation", obs2)
        return world
 
     return _system
