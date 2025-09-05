# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from ecsx.core.static_world import StaticWorld
from ecsx.systems.pipeline_batched_static import build_batched_step_static
from ecsx.integration.batched_io import (
    component_indices,
    find_agent_indices,
    read_observations_batched_by_index,
    write_actions_batched_by_index,
    read_rewards_terms_and_zero_batched_by_index,
)


# -------------------- tiny MLPs (no flax deps) --------------------

def _init_mlp(key, layers: Tuple[int, ...]) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """layers: e.g., (obs_dim, 64, 64, out_dim)"""
    params = {}
    keys = jax.random.split(key, len(layers) - 1)
    for i, (din, dout) in enumerate(zip(layers[:-1], layers[1:])):
        k1, k2 = jax.random.split(keys[i])
        w = jax.random.normal(k1, (din, dout)) * jnp.sqrt(2.0 / din)
        b = jnp.zeros((dout,))
        params[f"layer_{i}"] = (w, b)
    return params


def _apply_mlp(params, x):
    for i in range(len(params)):
        w, b = params[f"layer_{i}"]
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.tanh(x)
    return x


def _gaussian_logprob(mean, log_std, a):
    # mean/log_std: [B, A]; a: [B, A]
    var = jnp.exp(2.0 * log_std)
    logp = -0.5 * (jnp.sum(((a - mean) ** 2) / var, axis=-1) + 2.0 * jnp.sum(log_std, axis=-1) + a.shape[-1] * jnp.log(2 * jnp.pi))
    return logp


# -------------------- PPO config --------------------

@dataclass
class PPOConfig:
    batch_envs: int = 8
    horizon_T: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    train_iters: int = 4
    minibatch_frac: float = 0.25     # each iter uses this fraction of (B*T)
    entropy_coef: float = 0.0
    unroll_env: int = 8              # env unroll inside the compiled step


# -------------------- Rollout over (B, T) --------------------

def _rollout_once(
    sw_batched: StaticWorld,
    step_fn: Callable[[StaticWorld, Dict[str, jnp.ndarray]], StaticWorld],
    actor_params,
    value_params,
    idx_obs: int,
    idx_act: int,
    name_act: str,
    idx_rew: int,
    name_rew: str,
    idx_term: int,
    agent_idx: jnp.ndarray,  # [B]
    key: jnp.ndarray,
    T: int,
):
    """
    Returns:
      traj: dict of obs[T,B,D], act[T,B,A], rew[T,B], term[T,B], val[T+1,B], logp[T,B]
      sw_new, key_new
    """

    def body(carry, _):
        sw, k = carry
        k, ka, kn = jax.random.split(k, 3)

        obs = read_observations_batched_by_index(sw, idx_obs, agent_idx)  # (B, D)
        val = _apply_mlp(value_params, obs).squeeze(-1)                   # (B,)

        mu = _apply_mlp(actor_params["policy"], obs)                      # (B, A)
        log_std = actor_params["log_std"]                                 # (A,) or (1,A)
        log_std = jnp.broadcast_to(log_std, mu.shape)

        eps = jax.random.normal(ka, mu.shape)
        act = mu + jnp.exp(log_std) * eps                                 # reparameterized
        sw = write_actions_batched_by_index(sw, idx_act, name_act, agent_idx, act)

        sw = step_fn(sw, {})                                              # env step (unroll inside)

        rew, term, sw = read_rewards_terms_and_zero_batched_by_index(sw, idx_rew, name_rew, idx_term, agent_idx)
        logp = _gaussian_logprob(mu, log_std, act)

        out = {"obs": obs, "act": act, "rew": rew, "term": term, "val": val, "logp": logp}
        return (sw, kn), out

    (sw_T, key_T), outs = jax.lax.scan(body, (sw_batched, key), None, length=T)

    # Bootstrap value V_{T} from last observation
    last_obs = read_observations_batched_by_index(sw_T, idx_obs, agent_idx)
    last_val = _apply_mlp(value_params, last_obs).squeeze(-1)  # (B,)

    traj = {k: jnp.stack(v) for k, v in outs.items()}  # each (T, B, ...)
    traj["val_next"] = last_val                        # (B,)
    return traj, sw_T, key_T


def _gae_advantages(rew, term, val, val_next, gamma, lam):
    """
    Inputs: rew[T,B], term[T,B] boolean, val[T,B], val_next[B]
    Returns: adv[T,B], ret[T,B]
    """
    T, B = rew.shape
    def body(carry, t):
        gae, next_v = carry
        not_done = 1.0 - term[t].astype(jnp.float32)
        delta = rew[t] + gamma * not_done * next_v - val[t]
        gae = delta + gamma * lam * not_done * gae
        return (gae, val[t]), gae
    (_, _), adv_rev = jax.lax.scan(body, (jnp.zeros((B,), jnp.float32), val_next), jnp.arange(T - 1, -1, -1), reverse=True)
    adv = jnp.flip(adv_rev, axis=0)
    ret = adv + val
    return adv, ret


# -------------------- Optimizer (Adam minimal) --------------------

def _adam_init(params, lr):
    def zeros_like_tree(p):
        return jax.tree_util.tree_map(jnp.zeros_like, p)
    state = {"m": zeros_like_tree(params), "v": zeros_like_tree(params), "t": jnp.array(0, jnp.int32), "lr": jnp.array(lr, jnp.float32)}
    return state

def _adam_step(params, grads, state, beta1=0.9, beta2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g * g), state["v"], grads)

    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** t), v)

    lr = state["lr"]
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    new_state = {"m": m, "v": v, "t": t, "lr": lr}
    return params, new_state


# -------------------- Public training entrypoint --------------------

def ppo_train_static_batched(
    sw_batched: StaticWorld,
    systems: Tuple[Callable, ...],
    *,
    key: jnp.ndarray,
    config: PPOConfig = PPOConfig(),
    observation_name: str = "Observation",
    action_name: str = "Action",
    reward_name: str = "Reward",
    termination_name: str = "Termination",
):
    """
    Minimal end-to-end PPO on a batch of StaticWorlds.
    Returns: (sw_batched, actor_params, value_params, metrics_dict)
    """
    # Resolve component indices once (static)
    idx = component_indices(sw_batched, (observation_name, action_name, reward_name, termination_name))
    io = {
        "obs_i": idx[observation_name],
        "act_i": idx[action_name],
        "rew_i": idx[reward_name],
        "term_i": idx[termination_name],
        "act_name": action_name,
        "rew_name": reward_name,
    }

    # Build compiled batched env step (includes unroll)
    step_batched = build_batched_step_static(systems, unroll=config.unroll_env)

    # Agent indices [B]
    agent_idx = find_agent_indices(sw_batched, observation_name=observation_name, action_name=action_name)

    # Infer dims
    B = sw_batched.alive_mask.shape[0]
    obs_dim = int(sw_batched.stores[io["obs_i"]].spec.shape[0])
    act_dim = int(sw_batched.stores[io["act_i"]].spec.shape[0])

    # Init nets
    k1, k2, key = jax.random.split(key, 3)
    actor = {
        "policy": _init_mlp(k1, (obs_dim, 64, 64, act_dim)),
        "log_std": jnp.full((act_dim,), -0.5, jnp.float32),
    }
    critic = _init_mlp(k2, (obs_dim, 64, 64, 1))
    opt_actor = _adam_init(actor, config.actor_lr)
    opt_critic = _adam_init(critic, config.critic_lr)

    # ---- Rollout ----
    traj, sw_batched, key = _rollout_once(
        sw_batched, step_batched, actor, critic,
        io["obs_i"], io["act_i"], io["act_name"], io["rew_i"], io["rew_name"], io["term_i"],
        agent_idx, key, config.horizon_T,
    )

    # Compute advantages
    adv, ret = _gae_advantages(traj["rew"], traj["term"], traj["val"], traj["val_next"], config.gamma, config.lam)
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Flatten (T,B,...) → (N,...) where N=T*B
    def flat(x):
        return x.reshape((-1,) + x.shape[2:]) if x.ndim >= 2 else x.reshape((-1,))
    obs_flat  = flat(traj["obs"])
    act_flat  = flat(traj["act"])
    old_lp    = flat(traj["logp"])
    adv_flat  = flat(adv_norm)
    ret_flat  = flat(ret)

    # ---- Losses ----
    def actor_loss(actor_params, obs, act, old_logp, adv):
        mu = _apply_mlp(actor_params["policy"], obs)
        log_std = jnp.broadcast_to(actor_params["log_std"], mu.shape)
        new_logp = _gaussian_logprob(mu, log_std, act)
        ratio = jnp.exp(new_logp - old_logp)
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
        ent = -jnp.sum(jnp.exp(log_std) + 0.5 * jnp.log(2 * jnp.pi), axis=-1)  # constant wrt mu
        return -jnp.mean(jnp.minimum(unclipped, clipped) + config.entropy_coef * ent)

    def critic_loss(critic_params, obs, ret):
        v = _apply_mlp(critic_params, obs).squeeze(-1)
        return jnp.mean((v - ret) ** 2)

    # JIT per-minibatch updates
    actor_update = jax.jit(lambda p, s, o, a, olp, ad: _adam_step(p, jax.grad(actor_loss)(p, o, a, olp, ad), s))
    critic_update = jax.jit(lambda p, s, o, rt: _adam_step(p, jax.grad(critic_loss)(p, o, rt), s))

    # ---- SGD over a few iters ----
    N = obs_flat.shape[0]
    mb_size = max(1, int(config.minibatch_frac * N))
    perm_key = key
    for _ in range(config.train_iters):
        perm_key, sub = jax.random.split(perm_key)
        perm = jax.random.permutation(sub, N)
        for start in range(0, N, mb_size):
            sl = perm[start:start + mb_size]
            actor, opt_actor = actor_update(actor, opt_actor, obs_flat[sl], act_flat[sl], old_lp[sl], adv_flat[sl])
            critic, opt_critic = critic_update(critic, opt_critic, obs_flat[sl], ret_flat[sl])

    metrics = {
        "mean_return": float(traj["rew"].sum(axis=0).mean()),
        "adv_std": float(adv.std()),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "B": B,
        "T": config.horizon_T,
    }
    return sw_batched, actor, critic, metrics

