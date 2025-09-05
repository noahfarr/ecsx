# SPDX-License-Identifier: MIT
from __future__ import annotations
import time
import jax
import jax.numpy as jnp

from ecsx.algorithms.ppo_batched_discrete import (
    ppo_train_static_batched_discrete, PPOConfigDiscrete
)
from ecsx.systems.pipeline_batched_static import build_batched_step_static

from examples.ctf_gridworld import (
    CtfParams, build_world, freeze_batch,
    system_grid_movement_from_discrete, system_ctf_logic, system_observation,
    CtfRenderer, make_step
)

# ---------- helpers to compute per-team episodic returns ----------

def compute_team_returns(world):
    team = world.state.component_stores["Team"].data[:, 0]
    rew  = world.state.component_stores["Reward"].data[:, 0]
    r0 = float(jnp.sum(jnp.where(team == 0, rew, 0.0)))
    r1 = float(jnp.sum(jnp.where(team == 1, rew, 0.0)))
    return r0, r1

def zero_all_rewards(world):
    rew_store = world.state.component_stores["Reward"]
    world._world = world.state._with_store(
        "Reward",
        type(rew_store)(rew_store.spec,
                        jnp.zeros_like(rew_store.data),
                        rew_store.alive_mask),
    )
    return world

# ---------------- main ----------------

def main():
    params = CtfParams()
    B = 8               # batched envs
    H = 128             # rollout horizon
    UNROLL = 8

    # Build batched StaticWorld
    sw_batched = freeze_batch(B, n_per_team=1, params=params)

    systems = (
        system_grid_movement_from_discrete(params),
        system_ctf_logic(params),
        system_observation(params),
    )

    key = jax.random.PRNGKey(0)
    cfg = PPOConfigDiscrete(batch_envs=B, horizon_T=H, unroll_env=UNROLL, train_iters=4, minibatch_frac=0.5)

    # Train for a few epochs; after each, evaluate & print per-team episodic returns
    for epoch in range(3):
        t0 = time.time()
        sw_batched, actor, critic, metrics = ppo_train_static_batched_discrete(
            sw_batched, systems, key=key, config=cfg,
            observation_name="Observation",
            action_name="DiscreteAction",
            reward_name="Reward",
            termination_name="Termination",
        )
        key, _ = jax.random.split(key)
        dt = (time.time() - t0) * 1000
        print(f"[epoch {epoch}] train ms={dt:.1f}, mean_return_agent={metrics['mean_return_agent']:.3f}")

        # ---- Evaluation: run one batched rollout and print episodic returns per team ----
        step_b = build_batched_step_static(systems, unroll=UNROLL)
        # Track episodic returns per env; reset on termination
        team0_ret = jnp.zeros((B,), jnp.float32)
        team1_ret = jnp.zeros((B,), jnp.float32)

        # Unfreeze one env to compute rewards (we'll just use env 0 for per-step sampling)
        # Here we loop H/UNROLL calls; on each, step and accumulate team rewards.
        for _ in range(max(1, H // UNROLL)):
            # Step all envs with actions sampled from current policy (random for demo)
            # For simplicity, we keep random eval actions; you can plug the actor here.
            sw_batched = step_b(sw_batched, {})

            # Aggregate team rewards on host for *first* env only (demo); zero them
            # If you want per-env stats, thaw each and compute — omitted for brevity.
        # Instead, run a single env evaluation with rendering and accurate episodic returns:
        eval_single(params, actor)

def eval_single(params: CtfParams, actor_params, max_steps: int = 200):
    # Build one env and renderer
    w, t0, t1 = build_world(n_per_team=1, params=params)
    renderer = CtfRenderer(params)
    step = make_step(params, unroll=1)
    key = jax.random.PRNGKey(42)

    # Controlled agent is team-0 entity at index t0[0]
    agent = int(t0[0])
    team0_return = 0.0
    team1_return = 0.0

    try:
        for t in range(max_steps):
            # Build observation for controlled agent
            obs = w.state.component_stores["Observation"].read(jnp.asarray([agent]))[0]
            # Actor forward → logits → sample action
            logits = actor_forward(actor_params, obs[None, :])[0]
            a = int(jnp.argmax(logits))  # greedy for eval
            # Write action for both agents (opponent random)
            act_store = w.state.component_stores["DiscreteAction"]
            data = act_store.data
            data = data.at[agent].set(a)
            # Simple opponent: random move
            key, sub = jax.random.split(key)
            opp_id = int(t1[0])
            data = data.at[opp_id].set(int(jax.random.randint(sub, (), 0, 5)))
            w._world = w.state._with_store("DiscreteAction", type(act_store)(act_store.spec, data, act_store.alive_mask))

            # Step env
            w._world = step(w.state, {})

            # Accumulate team returns and zero rewards
            r0, r1 = compute_team_returns(w)
            team0_return += r0
            team1_return += r1
            w = zero_all_rewards(w)

            # Render
            renderer.draw(w)
            # Check termination
            term = bool(jnp.any(w.state.component_stores["Termination"].data[:, 0]))
            if term:
                print(f"[eval] episode end at t={t}  team0_return={team0_return:.2f}  team1_return={team1_return:.2f}")
                break
    finally:
        renderer.close()

def actor_forward(params, obs_batch):
    # obs_batch [B,D]
    x = obs_batch
    for i in range(len(params)):
        w, b = params[f"layer_{i}"]
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.tanh(x)
    return x

if __name__ == "__main__":
    main()

