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
)

# -------- tiny MLPs (no flax deps) --------

def _init_mlp(key, layers: Tuple[int, ...]) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    params = {}
    keys = jax.random.split(key, len(layers) - 1)
    for i, (din, dout) in enumerate(zip(layers[:-1], layers[1:])):
        k1 = keys[i]
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

# -------- categorical helpers --------

def _categorical_sample_rng(key, logits):
    # Gumbel-max trick
    g = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-8) + 1e-8)
    return jnp.argmax(logits + g, axis=-1).astype(jnp.int32)

def _categorical_logprob(logits, a):
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp_all, a[..., None], axis=-1)[..., 0]

# -------- config --------

@dataclass
class PPOConfigDiscrete:
    batch_envs: int = 8
    horizon_T: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    train_iters: int = 4
    minibatch_frac: float = 0.25
    entropy_coef: float = 0.01
    unroll_env: int = 8  # env unroll inside compiled step

# -------- Adam minimal --------

def _adam_init(params, lr):
    zero = jax.tree.map(jnp.zeros_like, params)
    return {"m": zero, "v": zero, "t": jnp.array(0, jnp.int32), "lr": jnp.array(lr, jnp.float32)}

def _adam_step(params, grads, state, beta1=0.9, beta2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * (g * g), state["v"], grads)
    mhat = jax.tree.map(lambda m: m / (1 - beta1 ** t), m)
    vhat = jax.tree.map(lambda v: v / (1 - beta2 ** t), v)
    lr = state["lr"]
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t, "lr": lr}

# -------- rollout (B, T) with single controlled agent per env --------

def _rollout_once(
    sw_batched: StaticWorld,
    step_fn: Callable[[StaticWorld, Dict[str, jnp.ndarray]], StaticWorld],
    actor_params,
    critic_params,
    idx_obs: int,
    idx_act: int,
    act_name: str,
    idx_rew: int,
    idx_term: int,
    agent_idx: jnp.ndarray,  # [B] index of controlled agent (team-0)
    key: jnp.ndarray,
    T: int,
):
    def body(carry, _):
        sw, k = carry
        k, kpol = jax.random.split(k)

        obs = read_observations_batched_by_index(sw, idx_obs, agent_idx)     # (B, D)
        val = _apply_mlp(critic_params, obs).squeeze(-1)                     # (B,)

        logits = _apply_mlp(actor_params, obs)                                # (B, A)
        act = _categorical_sample_rng(kpol, logits)                           # (B,)
        sw = write_actions_batched_by_index(sw, idx_act, act_name, agent_idx, act)

        sw = step_fn(sw, {})                                                  # env step (unrolled)

        # read controlled agent reward and done flags
        r_store = sw.stores[idx_rew].data                                     # (B, C, 1)
        t_store = sw.stores[idx_term].data                                    # (B, C, 1)
        rew = jax.vmap(lambda arr, i: arr[i, 0])(r_store, agent_idx)          # (B,)
        term = jax.vmap(lambda arr, i: arr[i, 0])(t_store, agent_idx).astype(jnp.bool_)  # (B,)

        # zero ONLY controlled agent rewards (keeps others untouched)
        def _zero_one(arr, i): return arr.at[i, 0].set(0.0)
        r_new = jax.vmap(_zero_one)(r_store, agent_idx)
        sw = sw._with_store("Reward", type(sw.stores[idx_rew])(
            sw.stores[idx_rew].spec, r_new, sw.stores[idx_rew].alive_mask
        ))

        logp = _categorical_logprob(logits, act)                               # (B,)
        out = {"obs": obs, "act": act, "rew": rew, "term": term, "val": val, "logp": logp}
        return (sw, k), out

    (sw_T, key_T), outs = jax.lax.scan(body, (sw_batched, key), None, length=T)

    last_obs = read_observations_batched_by_index(sw_T, idx_obs, agent_idx)
    last_val = _apply_mlp(critic_params, last_obs).squeeze(-1)
    traj = {k: jnp.stack(v) for k, v in outs.items()}  # each (T,B,...)
    traj["val_next"] = last_val                        # (B,)
    return traj, sw_T, key_T

def _gae_adv(rew, term, val, val_next, gamma, lam):
    T, B = rew.shape
    def body(carry, t):
        gae, next_v = carry
        not_done = 1.0 - term[t].astype(jnp.float32)
        delta = rew[t] + gamma * not_done * next_v - val[t]
        gae = delta + gamma * lam * not_done * gae
        return (gae, val[t]), gae
    (_, _), adv_rev = jax.lax.scan(body, (jnp.zeros((B,), jnp.float32), val_next), jnp.arange(T-1, -1, -1), reverse=True)
    adv = jnp.flip(adv_rev, axis=0); ret = adv + val
    return adv, ret

# -------- public entry --------

def ppo_train_static_batched_discrete(
    sw_batched: StaticWorld,
    systems: Tuple[Callable, ...],
    *,
    key: jnp.ndarray,
    config: PPOConfigDiscrete = PPOConfigDiscrete(),
    observation_name: str = "Observation",
    action_name: str = "DiscreteAction",
    reward_name: str = "Reward",
    termination_name: str = "Termination",
):
    # resolve indices, build env step
    idx = component_indices(sw_batched, (observation_name, action_name, reward_name, termination_name))
    step_batched = build_batched_step_static(systems, unroll=config.unroll_env)
    agent_idx = find_agent_indices(sw_batched, observation_name=observation_name, action_name=action_name)

    # dims
    B = sw_batched.alive_mask.shape[0]
    obs_dim = int(sw_batched.stores[idx[observation_name]].spec.shape[0])
    num_actions = 5  # Stay/Up/Down/Left/Right (env dependent)

    # init nets
    k1, k2, key = jax.random.split(key, 3)
    actor = _init_mlp(k1, (obs_dim, 64, 64, num_actions))
    critic = _init_mlp(k2, (obs_dim, 64, 64, 1))
    opt_actor = _adam_init(actor, config.actor_lr)
    opt_critic = _adam_init(critic, config.critic_lr)

    # rollout
    traj, sw_batched, key = _rollout_once(
        sw_batched, step_batched, actor, critic,
        idx[observation_name], idx[action_name], action_name,
        idx[reward_name], idx[termination_name],
        agent_idx, key, config.horizon_T,
    )

    adv, ret = _gae_adv(traj["rew"], traj["term"], traj["val"], traj["val_next"], config.gamma, config.lam)
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

    # flatten (T,B,...) → (N,...)
    flat = lambda x: x.reshape((-1,) + x.shape[2:]) if x.ndim >= 2 else x.reshape((-1,))
    obs_f, act_f, logp_old_f = flat(traj["obs"]), flat(traj["act"]), flat(traj["logp"])
    adv_f, ret_f = flat(adv_norm), flat(ret)

    # losses
    def actor_loss(p, obs, act, logp_old, adv):
        logits = _apply_mlp(p, obs)
        logp = _categorical_logprob(logits, act)
        ratio = jnp.exp(logp - logp_old)
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
        ent = -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1)
        return -jnp.mean(jnp.minimum(unclipped, clipped) + config.entropy_coef * ent)

    def critic_loss(p, obs, ret):
        v = _apply_mlp(p, obs).squeeze(-1)
        return jnp.mean((v - ret) ** 2)

    actor_update = jax.jit(lambda p, s, o, a, olp, ad: _adam_step(p, jax.grad(actor_loss)(p, o, a, olp, ad), s))
    critic_update = jax.jit(lambda p, s, o, rt: _adam_step(p, jax.grad(critic_loss)(p, o, rt), s))

    # SGD
    N = obs_f.shape[0]
    mb = max(1, int(config.minibatch_frac * N))
    perm_key = key
    for _ in range(config.train_iters):
        perm_key, sub = jax.random.split(perm_key)
        perm = jax.random.permutation(sub, N)
        for s in range(0, N, mb):
            sl = perm[s:s+mb]
            actor, opt_actor   = actor_update(actor, opt_actor, obs_f[sl], act_f[sl], logp_old_f[sl], adv_f[sl])
            critic, opt_critic = critic_update(critic, opt_critic, obs_f[sl], ret_f[sl])

    metrics = {
        "mean_return_agent": float(traj["rew"].sum(axis=0).mean()),
        "B": B, "T": config.horizon_T, "obs_dim": obs_dim, "num_actions": num_actions,
    }
    return sw_batched, actor, critic, metrics

