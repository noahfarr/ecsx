import numpy as np
import jax
import jax.numpy as jnp

from ecsx.core.world import World
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.event_specification import EventSpecification
from ecsx.components import (
    get_position_specification as position_spec,
    get_velocity_specification as velocity_spec,
)


def crossed_x_event_spec(capacity: int) -> EventSpecification:
    """
    Emits (entity_id, step_index) as int32 pairs whenever x > threshold.
    Capacity is per-step buffer capacity; one event per entity per step fits with `world.capacity`.
    """
    return EventSpecification(
        name="CrossedX",
        shape=(2,),  # (entity_id, step_index)
        dtype=jnp.int32,
        capacity=capacity,
    )


# --- Utilities ---------------------------------------------------------------


def show_positions(world: World, title: str) -> None:
    pos_store = world.get_store("Position")
    idx = jnp.where(world.state.alive_mask)[0].astype(jnp.int32)
    if idx.size == 0:
        print(f"\n{title}\n  (no alive entities)")
        return
    pos = np.array(pos_store.read(idx))
    print(f"\n{title}")
    for i, eid in enumerate(np.array(idx)):
        print(f"  entity {int(eid)} -> Position {pos[i]}")


def print_events(world: World, name: str) -> None:
    buf = world.state.event_buffers[name]
    count = int(buf.count)
    if count == 0:
        print("  events: none")
        return
    payloads = np.array(buf.data[:count])
    print(f"  events ({name}, count={count}):")
    for i in range(count):
        eid, step = map(int, payloads[i])
        print(f"    [{i}] entity {eid} crossed at step {step}")


# --- Core demo logic ---------------------------------------------------------


def physics_step(world: World, dt: float) -> World:
    """x += v * dt for all alive entities (host-side, no jit; clear and explicit)."""
    pos_store = world.get_store("Position")
    vel_store = world.get_store("Velocity")

    idx = jnp.where(world.state.alive_mask)[0].astype(jnp.int32)
    if idx.size == 0:
        return world

    pos = pos_store.read(idx)  # (n, 2)
    vel = vel_store.read(idx)  # (n, 2)
    new_pos = pos + vel * dt  # (n, 2)

    # Write back via WorldState helper.
    updated_pos_store = pos_store.write(idx, new_pos)
    world._world = world.state._with_store("Position", updated_pos_store)
    return world


def detect_and_emit_crossed_x(
    world: World, step_index: int, threshold: float = 5.0
) -> World:
    """
    Populate the CrossedX event buffer with (entity_id, step_index) for all alive entities whose x > threshold.
    Uses overwrite() each step, then the caller can reset_event_buffers().
    """
    name = "CrossedX"
    pos_store = world.get_store("Position")
    idx = jnp.where(world.state.alive_mask)[0].astype(jnp.int32)
    if idx.size == 0:
        # still clear/overwrite to zero events
        zero_payloads = jnp.zeros((world.capacity, 2), dtype=jnp.int32)
        world._world = world.state.write_event_buffer(
            name, zero_payloads, jnp.array(0, jnp.int32)
        )
        return world

    pos = pos_store.read(idx)  # (n, 2)
    crossed_mask = pos[:, 0] > threshold
    crossed_idx = idx[crossed_mask]

    count = crossed_idx.shape[0]
    count_arr = jnp.array(int(count), dtype=jnp.int32)

    # Build full-sized payload matrix and fill first `count` rows.
    payloads = jnp.zeros((world.capacity, 2), dtype=jnp.int32)
    if count > 0:
        payloads = payloads.at[:count, 0].set(crossed_idx)
        payloads = payloads.at[:count, 1].set(
            jnp.full((count,), int(step_index), dtype=jnp.int32)
        )

    world._world = world.state.write_event_buffer(name, payloads, count_arr)
    return world


def main() -> None:
    rng = np.random.default_rng(42)

    world = World(capacity=8, key=jax.random.PRNGKey(0))
    world.register_component(position_spec()).register_component(velocity_spec())

    # Register the per-step event buffer.
    world._world = world.state.register_event_buffer(
        crossed_x_event_spec(world.capacity)
    )

    # Spawn a few entities with random positions in [-5, 5]^2 and velocities in [-1, 1]^2.
    print("Spawning entities:")
    for _ in range(3):
        p = jnp.array(rng.uniform(-5.0, 5.0, size=(2,)).astype(np.float32))
        v = jnp.array(rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32))
        eid = world.spawn(Position=p, Velocity=v)
        print(f"  id={eid} p={np.array(p)} v={np.array(v)}")

    show_positions(world, "Initial positions")

    # Simulate and emit events each step.
    dt = 0.5
    steps = 5
    for s in range(1, steps + 1):
        world = physics_step(world, dt)
        world = detect_and_emit_crossed_x(world, step_index=s, threshold=5.0)
        print(f"step {s}/{steps} (dt={dt})")
        print_events(world, "CrossedX")
        world._world = world.state.reset_event_buffers()  # clear counts for next frame

    show_positions(world, "After simulation")

    # Showcase: detach Velocity from the first alive entity (requires fixed ComponentStore.clear),
    # then advance one more second; that entity should no longer move.
    alive_ids = np.where(np.array(world.state.alive_mask))[0]
    if alive_ids.size > 0:
        first = int(alive_ids[0])
        world = world.detach_component(first, "Velocity")
        world = physics_step(world, dt=1.0)
        world = detect_and_emit_crossed_x(world, step_index=steps + 1, threshold=5.0)
        show_positions(world, "After detaching Velocity from first entity (+1s)")
        print_events(world, "CrossedX")


if __name__ == "__main__":
    main()
