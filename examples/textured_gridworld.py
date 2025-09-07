import jax.numpy as jnp
import matplotlib.pyplot as plt
from ecsx.environments import build_grid_world
from ecsx.rendering import GridRenderer


def main():
    """Render a small grid world with textured entities."""
    grid_size = (5, 5)
    env = build_grid_world(
        num_agents=1,
        grid_size=grid_size,
        obstacle_positions=[(2, 1), (2, 2), (2, 3)],
        goal_position=(4, 4),
    )
    renderer = GridRenderer(
        grid_size=grid_size,
        cell_size=64,  # scale tiles so textures remain visible
        textures=env.default_inputs["textures"],
        background_id=int(env.default_inputs["background_id"]),
    )

    env.reset()
    frame = renderer.render(env.world)
    plt.imshow(frame)
    plt.axis("off")
    plt.title("Textured Grid World")
    plt.show()

    actions = [4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1]
    for step, action in enumerate(actions, start=1):
        _, reward, *_ = env.step(jnp.array(action, jnp.int32))
        frame = renderer.render(env.world)
        plt.imshow(frame)
        plt.axis("off")
        plt.title(f"Step {step}")
        plt.show()
        if reward > 0:
            break


if __name__ == "__main__":
    main()
