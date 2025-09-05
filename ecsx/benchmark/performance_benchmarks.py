# SPDX-License-Identifier: MIT
"""
Performance benchmarks for ecsx:
- Single env: jitted + unrolled step (host world)
- Vector runner: N independent worlds stepped in a host loop (shared jitted step)
- Batched static: N StaticWorlds stepped in one vmapped + jitted call
"""

import time
import statistics as stats
import jax
import jax.numpy as jnp

from ecsx.world.world_api import World
from ecsx.systems.pipeline_builder import build_step_unrolled
from ecsx.components.position import get_position_specification
from ecsx.components.velocity import get_velocity_specification
from ecsx.components.observation import get_observation_specification
from ecsx.components.action import get_action_specification
from ecsx.systems.physics_2d import physics_2d
from ecsx.systems.observation_system import observation_system
from ecsx.systems.action_application_system import action_application_system

# vector runner (host loop)
from ecsx.integration.vector_env import VectorWorldRunner

# batched static step (single JIT over N worlds)
from ecsx.systems.pipeline_batched_static import build_batched_step_static
from ecsx.core.static_world import StaticWorld
from ecsx.utils.tree_batch import stack_trees


# ------------------------- helpers ------------------------- #

def _make_world(n_entities: int, key=jax.random.PRNGKey(0)) -> World:
    """Build a world with Position, Velocity, Observation(4), Action(2) on all entities."""
    w = World(capacity=n_entities, key=key)
    w.register_component(get_position_specification())
    w.register_component(get_velocity_specification())
    w.register_component(get_observation_specification(4))
    w.register_component(get_action_specification(2))
    # populate all rows
    for _ in range(n_entities):
        w.spawn(
            Position=jnp.array([0.0, 0.0], jnp.float32),
            Velocity=jnp.array([0.0, 0.0], jnp.float32),
            Observation=jnp.zeros((4,), jnp.float32),
            Action=jnp.zeros((2,), jnp.float32),
        )
    return w


def _bench_single_env(world: World, calls: int = 200, unroll: int = 8):
    """
    Measure compile time (first call) and steady-state per-step steps/sec for one world,
    using a jitted, unrolled step.
    """
    step_fn = build_step_unrolled(
        (action_application_system(1.0),
         physics_2d(0.1),
         observation_system(True)),
        unroll=unroll,
    )

    # compile
    t0 = time.perf_counter()
    world._world = step_fn(world.state, {})
    jax.block_until_ready(world.state.time_step)
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000.0

    # extra warmup
    world._world = step_fn(world.state, {})
    jax.block_until_ready(world.state.time_step)

    # measure
    times = []
    for _ in range(calls):
        t0 = time.perf_counter()
        world._world = step_fn(world.state, {})
        jax.block_until_ready(world.state.time_step)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    call_med = stats.median(times) if times else 1.0  # seconds per unrolled call
    per_step_sec = call_med / float(unroll)
    steps_per_sec = 1.0 / per_step_sec
    return compile_ms, steps_per_sec, unroll


def _bench_vector(num_envs: int, capacity: int, calls: int = 200, unroll: int = 8):
    """
    Build N worlds (each with `capacity` entities) and step them with a shared jitted/unrolled step.
    Reports compile_ms (first call) and per-step throughput in steps/sec/env and total steps/sec.
    Host loop over envs; good baseline before batched static.
    """
    def builder():
        return _make_world(capacity)

    systems = (action_application_system(1.0),
               physics_2d(0.1),
               observation_system(True))

    runner = VectorWorldRunner(builder, systems, num_envs=num_envs, unroll=unroll)

    # compile
    t0 = time.perf_counter()
    runner.step({})
    runner.block_until_ready()
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000.0

    # warmup
    runner.step({})
    runner.block_until_ready()

    # measure
    times = []
    for _ in range(calls):
        t0 = time.perf_counter()
        runner.step({})
        runner.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    call_med = stats.median(times) if times else 1.0
    per_step_sec = call_med / float(unroll)              # seconds per env-step
    steps_per_sec_per_env = 1.0 / per_step_sec
    total_steps_per_sec = steps_per_sec_per_env * num_envs
    return compile_ms, steps_per_sec_per_env, total_steps_per_sec, unroll, num_envs


# ------------------------- main ------------------------- #

def main():
    print("=== single-env (unrolled, jitted) ===")
    for n in [64, 256, 1024, 2048, 4096]:
        w = _make_world(n)
        compile_ms, sps, unroll = _bench_single_env(w, calls=200, unroll=8)
        print(f"entities={n:5d}  compile_ms={compile_ms:7.1f}  steps_per_sec={sps:10.1f}  (unroll={unroll})")

    print("\n=== vector runner (host loop; per-env capacity=1024) ===")
    for num_envs in [1, 4, 8, 16]:
        compile_ms, spp_env, total_sps, unroll, N = _bench_vector(
            num_envs=num_envs, capacity=1024, calls=200, unroll=8
        )
        print(f"N={N:2d}  compile_ms={compile_ms:7.1f}  steps/sec/env={spp_env:10.1f}  total_steps/sec={total_sps:12.1f}  (unroll={unroll})")

    print("\n=== batched static (single JIT over N worlds; per-env capacity=1024) ===")
    # Build N identical worlds, freeze to StaticWorld with a fixed order, stack, and vmapped step
    def build_world_and_freeze(cap=1024):
        w = _make_world(cap)
        order = ("Position", "Velocity", "Observation", "Action")
        return StaticWorld.freeze(w.state, order)

    systems = (action_application_system(1.0),
               physics_2d(0.1),
               observation_system(True))
    step_batched = build_batched_step_static(systems, unroll=8)

    for N in [1, 4, 8, 16]:
        sw_list = [build_world_and_freeze(1024) for _ in range(N)]
        sw_batched = stack_trees(sw_list)   # StaticWorld PyTree with leading batch axis

        # compile
        t0 = time.perf_counter()
        sw_batched = step_batched(sw_batched, {})
        jax.block_until_ready(sw_batched.time_step)  # time_step has leading batch axis
        t1 = time.perf_counter()
        compile_ms = (t1 - t0) * 1000.0

        # warmup
        sw_batched = step_batched(sw_batched, {})
        jax.block_until_ready(sw_batched.time_step)

        # measure (1 ECS step per call across all envs)
        calls = 200
        times = []
        for _ in range(calls):
            t0 = time.perf_counter()
            sw_batched = step_batched(sw_batched, {})
            jax.block_until_ready(sw_batched.time_step)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        med = stats.median(times) if times else 1.0
        steps_per_sec_env = unroll / med       # each call performs 1 step for each env
        total_steps_per_sec = steps_per_sec_env * N
        print(f"N={N:2d}  compile_ms={compile_ms:7.1f}  steps/sec/env={steps_per_sec_env:10.1f}  total_steps_sec={total_steps_per_sec:12.1f}  (batched, unroll={unroll})")

if __name__ == "__main__":
    main()

