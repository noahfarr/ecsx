import jax
import jax.numpy as jnp

from ecsx.core.world_state import WorldState
from ecsx.core.entity_registry import EntityRegistry
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.static_world import StaticWorld
from ecsx.query.query_engine import select, select_static, compile_selector_for_static, indices_from_mask


def setup_world(capacity=8):
    key = jax.random.PRNGKey(0)
    w = WorldState.create(capacity=capacity, key=key)
    pos = ComponentSpecification("Position", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32))
    vel = ComponentSpecification("Velocity", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32))
    tag = ComponentSpecification("Tag", (), jnp.bool_, jnp.array(False))
    w = w.register_component(pos).register_component(vel).register_component(tag)
    return w

def test_static_world_select_and_compile():
    w = setup_world(capacity=6)
    reg = EntityRegistry.create(capacity=6)
    e0 = reg.spawn(); e1 = reg.spawn(); e2 = reg.spawn()
    w = w.with_alive_mask(reg.alive_mask)

    w = w.add_component_to_entity("Position", e0, jnp.array([1.0, 1.0], jnp.float32))
    w = w.add_component_to_entity("Velocity", e0, jnp.array([0.1, 0.0], jnp.float32))

    w = w.add_component_to_entity("Position", e1, jnp.array([2.0, 0.0], jnp.float32))
    w = w.add_component_to_entity("Tag", e1, jnp.array(True))

    w = w.add_component_to_entity("Position", e2, jnp.array([3.0, 0.0], jnp.float32))
    w = w.add_component_to_entity("Velocity", e2, jnp.array([0.0, 0.0], jnp.float32))
    w = w.add_component_to_entity("Tag", e2, jnp.array(True))

    # Dynamic (dict) path for correctness reference
    dyn_mask = select(w, required=("Position", "Velocity"), forbidden=("Tag",))
    dyn_idx = indices_from_mask(dyn_mask)
    assert dyn_idx.shape[0] == 1 and int(dyn_idx[0]) == e0

    # Static view (fixed store order)
    order = ("Position", "Velocity", "Tag")
    sw = StaticWorld.freeze(w, order)

    # 1) direct static select
    st_mask = select_static(sw, required=("Position", "Velocity"), forbidden=("Tag",))
    st_idx = indices_from_mask(st_mask)
    assert st_idx.shape[0] == 1 and int(st_idx[0]) == e0

    # 2) precompiled selector (index-resolved)
    fn = compile_selector_for_static(required=("Position", "Velocity"),
                                     forbidden=("Tag",),
                                     name_order=order)
    m2 = fn(sw)
    i2 = indices_from_mask(m2)
    assert i2.shape[0] == 1 and int(i2[0]) == e0

    # 3) jitted call (no Python dict lookups inside)
    jm = jax.jit(fn)(sw)
    ji = indices_from_mask(jm)
    assert ji.shape[0] == 1 and int(ji[0]) == e0

