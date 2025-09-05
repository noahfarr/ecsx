import jax
import jax.numpy as jnp

from ecsx.world.world_api import World
from ecsx.core.component_specification import ComponentSpecification
from ecsx.query.query_engine import select


def test_world_spawn_attach_step_and_despawn():
    w = World(capacity=6, key=jax.random.PRNGKey(0))
    value = ComponentSpecification("Value", (), jnp.int32, jnp.array(0, jnp.int32))
    pos = ComponentSpecification("Position", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32))
    w.register_component(value).register_component(pos)

    # spawn two entities
    e0 = w.spawn(Value=jnp.array(5, jnp.int32), Position=jnp.array([1.0, 2.0], jnp.float32))
    e1 = w.spawn(Position=jnp.array([0.0, 0.0], jnp.float32))

    # system: increment Value for all alive with Value
    def inc_value(world, key, inputs):
        mask = select(world, required=("Value",))
        idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
        cnt = jnp.sum(mask)
        vals = world.component_stores["Value"].read(idx[:cnt]) + 1
        world = world._with_store("Value", world.component_stores["Value"].write(idx[:cnt], vals))
        return world

    w.add_systems(inc_value).step()

    got_e0 = int(w.state.component_stores["Value"].read(jnp.asarray([e0]))[0])
    assert got_e0 == 6
    # e1 has no Value, selection should ignore it
    # despawn e0 and step again (no crash)
    w.despawn(e0).step()
    mask_after = select(w.state, required=("Value",))
    assert int(jnp.sum(mask_after)) == 0  # Value was only on e0 and it was cleared on despawn

