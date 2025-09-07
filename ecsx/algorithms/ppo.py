from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

from ecsx.core.environment import Environment


def _init_mlp(layer_sizes: Sequence[int], key: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for k, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        w = jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2 / n_in)
        b = jnp.zeros((n_out,))
        params.append((w, b))
    return params


def _mlp(params: Sequence[tuple[jax.Array, jax.Array]], x: jax.Array) -> jax.Array:
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    actor_sizes: Tuple[int, ...] = (64, 64)
    critic_sizes: Tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    batch_size: int = 128


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig, key: jax.Array):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        key1, key2 = jax.random.split(key)
        actor_sizes = (obs_dim, *config.actor_sizes, act_dim)
        critic_sizes = (obs_dim, *config.critic_sizes, 1)
        self.params = {
            "actor": _init_mlp(actor_sizes, key1),
            "critic": _init_mlp(critic_sizes, key2),
        }
        self.opt = optax.adam(config.learning_rate)
        self.opt_state = self.opt.init(self.params)

    def _actor(self, params, obs):
        return _mlp(params, obs)

    def _critic(self, params, obs):
        return _mlp(params, obs)[0]

    def policy(self, obs, key):
        """Sample actions and log probabilities for ``obs``.

        Works with either a single observation ``(obs_dim,)`` or a batch of
        observations ``(num_agents, obs_dim)``. In the batched case a separate
        action is sampled for each agent using shared network parameters.
        """

        obs = jnp.asarray(obs)
        if obs.ndim == 1:
            logits = self._actor(self.params["actor"], obs)
            log_probs = jax.nn.log_softmax(logits)
            key, sub = jax.random.split(key)
            action = jax.random.categorical(sub, logits)
            return int(action), log_probs[action], key

        logits = jax.vmap(self._actor, in_axes=(None, 0))(self.params["actor"], obs)
        log_probs = jax.nn.log_softmax(logits)
        key, sub = jax.random.split(key)
        subkeys = jax.random.split(sub, obs.shape[0])
        actions = jax.vmap(jax.random.categorical)(subkeys, logits)
        chosen = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
        return actions.astype(jnp.int32), chosen, key

    def value(self, obs):
        """Return value estimates for ``obs``.

        If ``obs`` is batched the returned value has shape ``(num_agents,)``.
        """

        obs = jnp.asarray(obs)
        if obs.ndim == 1:
            return float(self._critic(self.params["critic"], obs))
        return jax.vmap(self._critic, in_axes=(None, 0))(self.params["critic"], obs)

    def _compute_gae(self, rewards, values, dones, last_value):
        rewards = jnp.asarray(rewards)
        values = jnp.asarray(values)
        dones = jnp.asarray(dones)
        last_value = jnp.asarray(last_value)
        adv = jnp.zeros_like(last_value)
        advs = []
        values_ext = jnp.concatenate([values, last_value[None]], axis=0)
        for t in range(rewards.shape[0] - 1, -1, -1):
            delta = rewards[t] + self.config.gamma * (1 - dones[t]) * values_ext[t + 1] - values_ext[t]
            adv = delta + self.config.gamma * self.config.lam * (1 - dones[t]) * adv
            advs.insert(0, adv)
        advs = jnp.stack(advs)
        returns = advs + values_ext[:-1]
        return advs, returns

    def _update(self, batch):
        obs, act, logp, ret, adv = batch

        def loss_fn(params):
            logits = jax.vmap(self._actor, in_axes=(None, 0))(params["actor"], obs)
            logp_all = jax.nn.log_softmax(logits)
            new_logp = jnp.take_along_axis(logp_all, act[:, None], axis=1).squeeze()
            value = jax.vmap(self._critic, in_axes=(None, 0))(params["critic"], obs)
            ratio = jnp.exp(new_logp - logp)
            clipped = jnp.clip(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
            pg_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
            v_loss = jnp.mean((ret - value) ** 2)
            entropy = -jnp.mean(jnp.sum(jnp.exp(logp_all) * logp_all, axis=-1))
            return pg_loss + 0.5 * v_loss - 0.01 * entropy

        grads = jax.grad(loss_fn)(self.params)
        updates, self.opt_state = self.opt.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)

    def train_step(self, env: Environment, key: jax.Array):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        key1, key = jax.random.split(key)
        state, ts = env.reset(key1)
        for _ in range(self.config.batch_size):
            obs = jnp.asarray(ts.observation, jnp.float32)
            val = self.value(obs)
            key1, key = jax.random.split(key)
            action, logp, key1 = self.policy(obs, key1)
            state, ts = env.step(key, state, jnp.asarray(action, jnp.int32))
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(ts.reward)
            val_buf.append(val)
            done_buf.append(ts.done)
            if bool(jnp.all(ts.done)):
                key2, key = jax.random.split(key)
                state, ts = env.reset(key2)
        last_val = self.value(jnp.asarray(ts.observation, jnp.float32))
        rewards = jnp.stack(rew_buf)
        values = jnp.stack(val_buf)
        dones = jnp.stack(done_buf)
        adv, ret = self._compute_gae(rewards, values, dones, last_val)
        obs_batch = jnp.stack(obs_buf)
        act_batch = jnp.stack(act_buf)
        logp_batch = jnp.stack(logp_buf)
        if obs_batch.ndim == 3:  # (T, N, obs_dim)
            T, N, _ = obs_batch.shape
            obs_batch = obs_batch.reshape(T * N, self.obs_dim)
            act_batch = act_batch.reshape(T * N)
            logp_batch = logp_batch.reshape(T * N)
            ret = ret.reshape(T * N)
            adv = adv.reshape(T * N)
        batch = (
            obs_batch,
            act_batch.astype(jnp.int32),
            logp_batch.astype(jnp.float32),
            ret,
            adv,
        )
        for _ in range(self.config.epochs):
            self._update(batch)
        return key
