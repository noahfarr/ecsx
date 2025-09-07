import jax.numpy as jnp
import matplotlib.pyplot as plt
from ecsx.environments import build_grid_world
from ecsx.rendering import GridRenderer


def main():
    grid_size = (5, 5)
    obstacles = [(2, 1), (2, 2), (2, 3)]
    goal = (4, 4)

    env = build_grid_world(
        num_agents=1,
        grid_size=grid_size,
        obstacle_positions=obstacles,
        goal_position=goal,
    )
    renderer = GridRenderer(
        grid_size=grid_size, textures=env.default_inputs["textures"]
    )

    obs = env.reset()
    print("start obs", obs)

    action_sequence = [4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1]
    for step, action in enumerate(action_sequence, start=1):
        obs, rew, done, _ = env.step(jnp.array(action, jnp.int32))
        frame = renderer.render(env.world)

        plt.imshow(frame)
        plt.show()
        print(
            f"step {step}: action={action} obs={obs} reward={rew} frame={frame.shape}"
        )
        if rew > 0:
            print("Reached goal!")
            break


if __name__ == "__main__":
    main()
