import jax
import jax.numpy as jnp
from ecsx.environments import build_grid_world


def test_procedural_obstacles_generation():
    key = jax.random.PRNGKey(0)
    env = build_grid_world(num_agents=1, grid_size=(4, 4), num_obstacles=3, key=key)
    obst_store = env.world.get_store("Obstacle")
    pos_store = env.world.get_store("Position")
    obstacle_idx = jnp.where(obst_store.alive_mask)[0]
    assert obstacle_idx.size == 3
    obstacle_positions = pos_store.read(obstacle_idx)
    agent_positions = pos_store.read(env.agent_indices)
    for pos in obstacle_positions:
        assert not jnp.any(jnp.all(agent_positions == pos, axis=1))
    # ensure unique positions
    assert len({tuple(map(int, p)) for p in obstacle_positions.tolist()}) == 3
