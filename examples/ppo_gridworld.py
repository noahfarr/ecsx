"""Train a PPO agent on the grid world environment."""

import jax
from ecsx.environments import build_grid_world
from ecsx.algorithms import PPOAgent, PPOConfig


def main():
    # Demonstrate training on a simple two-agent grid world
    env = build_grid_world(num_agents=2, grid_size=(5, 5), goal_position=(4, 4))
    config = PPOConfig(batch_size=32, epochs=2)
    agent = PPOAgent(obs_dim=2, act_dim=5, config=config, key=jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(42)
    for _ in range(5):
        key = agent.train_step(env, key)


if __name__ == "__main__":
    main()
