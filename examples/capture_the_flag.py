import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ecsx.environments import build_capture_the_flag
from ecsx.rendering import GridRenderer


def main():
    team_size = 2
    grid_size = (16, 16)
    env = build_capture_the_flag(team_size=team_size, map_size=grid_size)
    renderer = GridRenderer(
        grid_size=grid_size,
        textures=env.default_inputs["textures"],
        background_id=int(env.default_inputs["background_id"]),
    )

    key = jax.random.PRNGKey(0)
    state, ts = env.reset(key)
    print("start obs", ts.observation)

    num_steps = 100
    for step in range(num_steps):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(
            action_key, shape=(team_size * 2,), minval=0, maxval=4
        )
        key, subkey = jax.random.split(key)
        state, ts = env.step(subkey, state, action)
        print("Observation", ts.observation)
        frame = renderer.render(state.world)
        plt.imshow(frame)
        plt.title(f"step {step}")
        plt.show()
        print(
            f"step {step}: action={action.tolist()} reward={ts.reward} done={ts.done}"
        )


if __name__ == "__main__":
    main()
