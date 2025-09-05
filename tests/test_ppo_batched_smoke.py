# SPDX-License-Identifier: MIT
import jax
import jax.numpy as jnp

from ecsx.world.world_api import World
from ecsx.core.static_world import StaticWorld
from ecsx.utils.tree_batch import stack_trees

from ecsx.components.position import get_position_specification
from ecsx.components.velocity import get_velocity_specification
from ecsx.components.observation import get_observation_specification
from ecsx.components.action import get_action_specification
from ecsx.core.component_specification import ComponentSpecification

from ecsx.systems.action_application_system import action_application_system
from ecsx.systems.physics_2d import physics_2d
from ecsx.systems.observation_system import observation_system

from ecsx.algorithms.ppo_batched import ppo_train_static_batched, PPOConfig
from ecsx.integration.batched_io import find_agent_indices


def _build_world(capacity: int, key) -> World:
    """World with one agent entity having all required components."""
    w = World(capacity=capacity, key=key)
    w.register_component(get_position_specification())
    w.register_component(get_velocity_specification())
    w.register_component(get_observation_specification(4))
    w.register_component(get_action_specification(2))
    # Reward (scalar float) and Termination (scalar bool)
    rew_spec = ComponentSpecification("Reward", (1,), jnp.float32, jnp.array([0.0], jnp.float32))
    term_spec = ComponentSpecification("Termination", (1,), jnp.bool_, jnp.array([False]))
    w.register_component(rew_spec)
    w.register_component(term_spec)

    # Spawn exactly one agent with all components present
    w.spawn(
        Position=jnp.array([0.0, 0.0], jnp.float32),
        Velocity=jnp.array([0.0, 0.0], jnp.float32),
        Observation=jnp.zeros((4,), jnp.float32),
        Action=jnp.zeros((2,), jnp.float32),
        Reward=jnp.zeros((1,), jnp.float32),
        Termination=jnp.array([False]),
    )
    return w


def _build_batched_static_world(batch_envs: int, capacity: int) -> StaticWorld:
    keys = jax.random.split(jax.random.PRNGKey(0), batch_envs)
    sws = []
    for k in keys:
        w = _build_world(capacity, k)
        order = ("Position", "Velocity", "Observation", "Action", "Reward", "Termination")
        sws.append(StaticWorld.freeze(w.state, order))
    return stack_trees(sws)  # StaticWorld with leading batch axis


def test_ppo_batched_smoke_runs_and_updates_state():
    B, CAP = 4, 8
    swb = _build_batched_static_world(B, CAP)

    systems = (
        action_application_system(1.0),
        physics_2d(0.1),
        observation_system(True),
    )

    cfg = PPOConfig(
        batch_envs=B,
        horizon_T=16,     # short rollout for test speed
        unroll_env=4,     # reduce launch overhead
        train_iters=2,    # a couple of SGD passes
        minibatch_frac=0.5,
        gamma=0.99,
        lam=0.95,
    )

    key = jax.random.PRNGKey(42)

    # Baseline positions before training
    pos_idx = swb.store_names.index("Position")
    pos_before = swb.stores[pos_idx].data  # (B, CAP, 2)

    swb_out, actor, critic, metrics = ppo_train_static_batched(
        sw_batched=swb,
        systems=systems,
        key=key,
        config=cfg,
        observation_name="Observation",
        action_name="Action",
        reward_name="Reward",
        termination_name="Termination",
    )

    # --- Assertions ---
    # 1) Outputs exist and have sane shapes/types
    assert isinstance(swb_out, StaticWorld)
    assert "mean_return" in metrics and "obs_dim" in metrics and "act_dim" in metrics
    # values are finite
    for v in metrics.values():
        if isinstance(v, float):
            assert jnp.isfinite(jnp.array(v)), "metric contains NaN/Inf"

    # 2) Actor/critic params exist and match dims
    assert "policy" in actor and "log_std" in actor
    assert actor["log_std"].shape[-1] == 2  # Action(2)
    # critic final layer outputs scalar value
    sample_obs = jnp.zeros((B, metrics["obs_dim"]), jnp.float32)
    v_out = (lambda p, x: jax.nn.tanh(  # quick shape probe via apply function
        (lambda y: y)(x @ p["layer_0"][0] + p["layer_0"][1])
    ))(critic, sample_obs)  # not strictly necessary; just ensures params exist

    # 3) Environment state advanced: agent positions moved from zero
    agent_idx = find_agent_indices(swb_out, observation_name="Observation", action_name="Action")  # (B,)
    # Gather positions for each env's agent
    def _g(arr, i):
        return arr[i]
    pos_after_agents = jax.vmap(_g)(swb_out.stores[pos_idx].data, agent_idx)  # (B, 2)
    moved = jnp.any(jnp.abs(pos_after_agents) > 0.0)
    # Compare against original positions for same indices
    pos_before_agents = jax.vmap(_g)(pos_before, agent_idx)
    changed = jnp.any(jnp.abs(pos_after_agents - pos_before_agents) > 0.0)

    assert bool(moved) and bool(changed), "Agent positions did not change after rollout/training"


