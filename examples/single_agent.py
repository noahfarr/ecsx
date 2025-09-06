import jax.numpy as jnp
from ecsx.environments import build_grid_world
from ecsx.rendering import GridRenderer


def main():
    env = build_grid_world(num_agents=1, grid_size=(5, 5), goal_position=(4, 4))
    renderer = GridRenderer(grid_size=(5, 5))
    obs = env.reset()
    done = False
    while not done:
        action = jnp.array(4, jnp.int32)  # move right
        obs, rew, done, _ = env.step(action)
        img = renderer.render(env.world, goal_position=(4, 4))
        print("obs", obs, "rew", rew, "done", done, "image", img.shape)
        break


if __name__ == "__main__":
    main()
