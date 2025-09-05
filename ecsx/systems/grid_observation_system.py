import jax.numpy as jnp
from ecsx.query.query_engine import select

def grid_observation_system(grid_width: int, grid_height: int):
    W = jnp.array(grid_width, jnp.float32)
    H = jnp.array(grid_height, jnp.float32)

    def _system(world, key, inputs):
        # single-goal assumption for v0.1; if multiple, take the first
        gmask = select(world, required=("GridGoal", "GridPosition"))
        gidx  = jnp.nonzero(gmask, size=1, fill_value=-1)[0]
        has_goal = jnp.sum(gmask) > 0
        goal_xy = jnp.where(has_goal,
                            world.component_stores["GridPosition"].read(gidx[:1])[0],
                            jnp.array([0, 0], jnp.int32))

        # For all with Observation + GridPosition, write [ax/W, ay/H, gx/W, gy/H]
        mask = select(world, required=("Observation", "GridPosition"))
        idx  = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
        cnt  = jnp.sum(mask)
        if int(cnt) == 0:
            return world

        apos = world.component_stores["GridPosition"].read(idx[:cnt]).astype(jnp.float32)
        obs  = jnp.stack([apos[:, 0] / W, apos[:, 1] / H,
                          jnp.full((int(cnt),), goal_xy[0] / W),
                          jnp.full((int(cnt),), goal_xy[1] / H)], axis=1)

        store = world.component_stores["Observation"].write(idx[:cnt], obs)
        world = world._with_store("Observation", store)
        return world

    return _system

