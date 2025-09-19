import numpy as np
import pytest
import jax.numpy as jnp
from jax import tree_util as jtu

from ecsx.components import Position
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.component_store import ComponentStore


def _position_spec() -> ComponentSpecification:
    return ComponentSpecification(
        "Position",
        Position(
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        ),
    )


def test_from_spec_initializes_defaults():
    spec = _position_spec()
    store = ComponentStore.from_spec(spec, capacity=3)

    assert store.capacity == 3
    np.testing.assert_array_equal(np.asarray(store.alive_mask), np.zeros((3,), dtype=bool))

    leaves = [np.asarray(arr) for arr in jtu.tree_leaves(store.data)]
    assert all(leaf.shape[0] == 3 for leaf in leaves)


def test_write_and_read_roundtrip():
    spec = _position_spec()
    store = ComponentStore.from_spec(spec, capacity=3)

    indices = jnp.array([0, 2], dtype=jnp.int32)
    value = Position(
        jnp.array([1.0, -1.0], dtype=jnp.float32),
        jnp.array([2.0, -2.0], dtype=jnp.float32),
    )
    updated = store.write(indices, value)

    result = updated.read(indices)
    np.testing.assert_allclose(np.asarray(result.x), np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(result.y), np.array([2.0, -2.0], dtype=np.float32))
    np.testing.assert_array_equal(
        np.asarray(updated.alive_mask),
        np.array([True, False, True]),
    )


def test_write_validates_structure_and_shapes():
    spec = _position_spec()
    store = ComponentStore.from_spec(spec, capacity=2)

    with pytest.raises(ValueError):
        store.write(jnp.array([0], dtype=jnp.int32), jnp.array([1.0, 2.0], dtype=jnp.float32))

    with pytest.raises(ValueError):
        store.write(
            jnp.array([0, 1], dtype=jnp.int32),
            Position(jnp.array([1.0], dtype=jnp.float32), jnp.array([2.0], dtype=jnp.float32)),
        )


def test_clear_restores_default_values():
    spec = _position_spec()
    store = ComponentStore.from_spec(spec, capacity=2)
    store = store.write(
        jnp.array([0], dtype=jnp.int32),
        Position(jnp.array([5.0], dtype=jnp.float32), jnp.array([6.0], dtype=jnp.float32)),
    )

    cleared = store.clear(jnp.array([0], dtype=jnp.int32))
    result = cleared.read(jnp.array([0], dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(result.x), np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(result.y), np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(cleared.alive_mask), np.array([False, False]))


def test_component_store_tree_roundtrip():
    spec = _position_spec()
    store = ComponentStore.from_spec(spec, capacity=2)
    store = store.write(
        jnp.array([0], dtype=jnp.int32),
        Position(jnp.array([1.0], dtype=jnp.float32), jnp.array([2.0], dtype=jnp.float32)),
    )

    children, aux = store.tree_flatten()
    restored = ComponentStore.tree_unflatten(aux, children)

    original_leaves = [np.asarray(leaf) for leaf in jtu.tree_leaves(store.data)]
    restored_leaves = [np.asarray(leaf) for leaf in jtu.tree_leaves(restored.data)]
    for orig, new in zip(original_leaves, restored_leaves):
        np.testing.assert_allclose(orig, new)
    np.testing.assert_array_equal(np.asarray(store.alive_mask), np.asarray(restored.alive_mask))
