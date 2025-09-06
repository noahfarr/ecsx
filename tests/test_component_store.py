import jax.numpy as jnp
import pytest
from ecsx.core.component_specification import ComponentSpecification
from ecsx.core.component_store import ComponentStore


def make_spec():
    return ComponentSpecification(
        "Foo", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )


def test_from_spec_and_capacity():
    spec = make_spec()
    store = ComponentStore.from_spec(spec, capacity=4)
    assert store.capacity == 4
    assert store.data.shape == (4, 2)
    assert jnp.array_equal(store.data, jnp.zeros((4, 2), jnp.float32))
    assert jnp.array_equal(store.alive_mask, jnp.zeros((4,), bool))


def test_write_read_and_clear():
    spec = make_spec()
    store = ComponentStore.from_spec(spec, capacity=4)
    store = store.write(jnp.array([1]), jnp.array([[3.0, 4.0]], jnp.float32))
    read_back = store.read(jnp.array([1]))
    assert jnp.array_equal(read_back, jnp.array([[3.0, 4.0]], jnp.float32))
    assert bool(store.alive_mask[1])
    store = store.clear(jnp.array([1]))
    assert jnp.array_equal(
        store.read(jnp.array([1])), jnp.array([[0.0, 0.0]], jnp.float32)
    )
    assert not bool(store.alive_mask[1])
    with pytest.raises(ValueError):
        store.write(jnp.array([0, 1]), jnp.array([[1.0, 2.0]], jnp.float32))
