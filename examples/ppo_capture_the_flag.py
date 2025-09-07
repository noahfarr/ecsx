import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
from typing import NamedTuple, Any

from ecsx.environments.capture_the_flag import build_capture_the_flag


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        pi = distrax.Categorical(logits=logits)
        v = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        v = activation(v)
        v = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(v)
        v = activation(v)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)
        return pi, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config: dict[str, Any]):
    env = build_capture_the_flag(
        team_size=config["TEAM_SIZE"], map_size=config["MAP_SIZE"]
    )
    num_agents = int(env.agent_indices.shape[0])
    dummy_state, dummy_ts = env.reset(jax.random.PRNGKey(0))
    obs_dim = dummy_ts.observation.shape[-1]
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"])
        // config["NUM_STEPS"]
        // (config["NUM_ENVS"] * num_agents)
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * num_agents * config["NUM_STEPS"]
        // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng: jax.Array):
        network = ActorCritic(action_dim=5, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((obs_dim,))
        params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_keys = jax.random.split(_rng, config["NUM_ENVS"])
        env_state, ts = jax.vmap(env.reset)(reset_keys)
        obsv = ts.observation

        def _update_step(runner_state, _):
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                rng, _rng = jax.random.split(rng)
                flat_obs = last_obs.reshape((-1, obs_dim))
                pi, value = network.apply(train_state.params, flat_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                action = action.reshape(config["NUM_ENVS"], num_agents)
                log_prob = log_prob.reshape(config["NUM_ENVS"], num_agents)
                value = value.reshape(config["NUM_ENVS"], num_agents)
                rng, _rng = jax.random.split(rng)
                step_keys = jax.random.split(_rng, config["NUM_ENVS"])
                env_state, ts = jax.vmap(env.step)(step_keys, env_state, action)
                transition = Transition(
                    ts.done, action, value, ts.reward, log_prob, last_obs, ts.reward
                )
                runner_state = (train_state, env_state, ts.observation, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state
            flat_last_obs = last_obs.reshape((-1, obs_dim))
            _, last_val = network.apply(train_state.params, flat_last_obs)
            last_val = last_val.reshape(-1)

            traj_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((config["NUM_STEPS"], -1) + x.shape[3:]),
                traj_batch,
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1 - transition.done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
                    )
                    return (gae, transition.value), gae

                (_, _), advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_flat, last_val)

            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch):
                    traj_batch, advantages, targets = batch

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        ) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_flat, advantages, targets, rng)
            update_state, _ = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]
            metric = traj_flat.reward.sum()
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        runner_state = (train_state, env_state, obsv, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "TEAM_SIZE": 1,
        "MAP_SIZE": (10, 10),
    }
    rng = jax.random.PRNGKey(0)
    train_fn = jax.jit(make_train(config))
    result = train_fn(rng)
    print("Training metrics", result["metrics"])
