import jax.numpy as jnp
from ecsx.environments import build_grid_world
from ecsx.rendering import GridRenderer


def main():
    env = build_grid_world(num_agents=2, grid_size=(5, 5), num_obstacles=3)
    renderer = GridRenderer(grid_size=(5, 5), textures=env.default_inputs["textures"])
    obs = env.reset()
    actions = jnp.array([4, 1], jnp.int32)
    obs, rew, done, _ = env.step(actions)
    img = renderer.render(env.world)
    print("obs", obs, "rew", rew, "done", done, "image", img.shape)


if __name__ == "__main__":
    main()
