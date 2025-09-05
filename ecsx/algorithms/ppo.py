# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import optax

from .networks import init_mlp, apply_mlp

Array = jax.Array

@dataclass
class PPOConfig:
    seed: int = 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    steps_per_rollout: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    hidden_layers: Tuple[int, ...] = (64, 64)

@dataclass
class PPOState:
    key: Array
    policy_params: list
    value_params: list
    log_std: Array         # (action_dim,)
    opt_state: optax.OptState

def init(config: PPOConfig, obs_dim: int, action_dim: int) -> PPOState:
    key = jax.random.PRNGKey(config.seed)
    k1, k2 = jax.random.split(key)
    policy = init_mlp(k1, obs_dim, config.hidden_layers, action_dim)
    value = init_mlp(k2, obs_dim, config.hidden_layers, 1)
    log_std = jnp.full((action_dim,), -0.5, jnp.float32)
    params_flat = (policy, value, log_std)
    tx = optax.adam(config.learning_rate)
    opt_state = tx.init(params_flat)
    return PPOState(key, policy, value, log_std, opt_state)

# ---------- distributions (diagonal Normal, clipped at env boundary when acting) ----------
def _policy_forward(policy_params, log_std, obs):
    mu = apply_mlp(policy_params, obs)
    std = jnp.exp(log_std)
    return mu, std

def _log_prob_normal_diag(a, mu, std):
    var = std**2
    return -0.5 * (jnp.sum(((a - mu)**2) / var + 2*jnp.log(std) + jnp.log(2*jnp.pi), axis=-1))

def _sample_action(key, mu, std):
    return mu + std * jax.random.normal(key, mu.shape)

# ---------- rollout buffer helpers ----------
def _gae(returns_v, rewards, dones, gamma, lam):
    """returns_v: values (T+1,), rewards: (T,), dones: (T,)"""
    T = rewards.shape[0]
    adv = jnp.zeros_like(rewards)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * returns_v[t + 1] * (1.0 - dones[t]) - returns_v[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv = adv.at[t].set(gae)
    ret = adv + returns_v[:-1]
    return adv, ret

# ---------- training step ----------
def train_epoch(config: PPOConfig, state: PPOState, env) -> Tuple[PPOState, Dict[str, float]]:
    """Collect one rollout in Python (env.step), then do PPO updates."""
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    # Storage
    T = config.steps_per_rollout
    obs = jnp.zeros((T, obs_dim), jnp.float32)
    acts = jnp.zeros((T, act_dim), jnp.float32)
    rews = jnp.zeros((T,), jnp.float32)
    dones = jnp.zeros((T,), jnp.float32)
    logps = jnp.zeros((T,), jnp.float32)
    vals = jnp.zeros((T + 1,), jnp.float32)

    # Start from env state
    o, _ = env.reset()
    o = jnp.asarray(o, jnp.float32)

    key = state.key
    for t in range(T):
        key, sk = jax.random.split(key)
        mu, std = _policy_forward(state.policy_params, state.log_std, o[None, :])
        mu, std = mu[0], std  # shapes: (A,), (A,)
        a = _sample_action(sk, mu, std)
        logp = _log_prob_normal_diag(a, mu, std)

        # Clip to Box action space on the way out
        a_clip = jnp.clip(a, env.action_space.low, env.action_space.high)
        no, r, d, trunc, _ = env.step(jnp.asarray(a_clip, jnp.float32))
        v = apply_mlp(state.value_params, o[None, :])[0, 0]
        obs = obs.at[t].set(o)
        acts = acts.at[t].set(a)
        rews = rews.at[t].set(jnp.asarray(r, jnp.float32))
        dones = dones.at[t].set(1.0 if (d or trunc) else 0.0)
        logps = logps.at[t].set(logp)
        vals = vals.at[t].set(v)
        o = jnp.asarray(no, jnp.float32)
        if d or trunc:
            o, _ = env.reset()
            o = jnp.asarray(o, jnp.float32)

    # bootstrap last value
    vals = vals.at[T].set(apply_mlp(state.value_params, o[None, :])[0, 0])

    adv, ret = _gae(vals, rews, dones, config.gamma, config.gae_lambda)
    adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

    # Flatten batch
    batch = {
        "obs": obs,
        "acts": acts,
        "logps": logps,
        "adv": adv,
        "ret": ret,
        "val": vals[:-1],
    }

    # Optim step
    tx = optax.adam(config.learning_rate)

    def loss_fn(params_flat, batch):
        policy_params, value_params, log_std = params_flat
        mu, std = _policy_forward(policy_params, log_std, batch["obs"])
        new_logp = _log_prob_normal_diag(batch["acts"], mu, std)
        ratio = jnp.exp(new_logp - batch["logps"])
        clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        policy_loss = -jnp.mean(jnp.minimum(ratio * batch["adv"], clipped * batch["adv"]))

        vpred = apply_mlp(value_params, batch["obs"])[:, 0]
        value_loss = jnp.mean((batch["ret"] - vpred) ** 2)

        entropy = jnp.mean(0.5 * jnp.log(2 * jnp.pi * jnp.e) + jnp.mean(jnp.log(std)))

        total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        return total, (policy_loss, value_loss, entropy)

    params_flat = (state.policy_params, state.value_params, state.log_std)
    opt_state = state.opt_state

    # minibatching (simple full-batch if minibatch_size >= T)
    idx = jnp.arange(T)
    for _ in range(config.update_epochs):
        # shuffle
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, idx)
        for start in range(0, T, max(1, config.minibatch_size)):
            mb_idx = perm[start:start + config.minibatch_size]
            sub = {k: v[mb_idx] if v.shape[0] == T else v for k, v in batch.items()}
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_flat, sub)
            updates, opt_state = tx.update(grads, opt_state, params_flat)
            params_flat = optax.apply_updates(params_flat, updates)

    policy_loss, value_loss, entropy = [float(x) for x in aux]
    new_policy, new_value, new_log_std = params_flat
    return PPOState(key, new_policy, new_value, new_log_std, opt_state), {
        "loss": float(loss),
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
    }

