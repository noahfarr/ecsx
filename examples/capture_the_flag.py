import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ecsx.environments import build_capture_the_flag
from ecsx.rendering import GridRenderer


def main():
    env = build_capture_the_flag(team_size=1, map_size=(3, 1))
    renderer = GridRenderer(
        grid_size=(3, 1),
        textures=env.default_inputs["textures"],
        background_id=int(env.default_inputs["background_id"]),
    )

    key = jax.random.PRNGKey(0)
    state, ts = env.reset(key)
    print("start obs", ts.observation)

    action_sequence = [
        jnp.array([4, 0], jnp.int32),
        jnp.array([4, 0], jnp.int32),
        jnp.array([3, 0], jnp.int32),
        jnp.array([3, 0], jnp.int32),
    ]

    for step, action in enumerate(action_sequence, start=1):
        key, subkey = jax.random.split(key)
        state, ts = env.step(subkey, state, action)
        frame = renderer.render(state.world)
        plt.imshow(frame)
        plt.title(f"step {step}")
        plt.show()
        print(
            f"step {step}: action={action.tolist()} reward={ts.reward} done={ts.done}"
        )


if __name__ == "__main__":
    main()
