import jax.numpy as jnp
from ecsx.environments import build_grid_world


def test_goal_and_background_texture_usage():
    goal = (1, 2)
    env = build_grid_world(num_agents=1, grid_size=(3, 3), goal_position=goal)
    pos_store = env.world.get_store("Position")
    rend_store = env.world.get_store("Renderable")
    idx = jnp.where(rend_store.alive_mask)[0]
    positions = pos_store.read(idx)
    texture_ids = rend_store.read(idx).reshape(-1)
    goal_idx = jnp.where(jnp.all(positions == jnp.array(goal), axis=1))[0]
    assert goal_idx.size == 1
    # Goal should use the second texture slot
    assert int(texture_ids[int(goal_idx[0])]) == 1
    # Background remains the floor texture
    assert int(env.default_inputs["background_id"]) == 3
