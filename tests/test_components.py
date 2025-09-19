import numpy as np
import pytest
import jax.numpy as jnp
from jax import tree_util as jtu

from ecsx import components as components_module
from ecsx.components import (
    Carriable,
    ContinuousAction,
    Direction,
    DiscreteAction,
    Flag,
    Goal,
    MultiDiscreteAction,
    Obstacle,
    Position,
    Renderable,
    Team,
    Velocity,
)


_COMPONENT_FACTORIES = [
    (
        "Position",
        lambda: Position(
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array(-2.5, dtype=jnp.float32),
        ),
    ),
    (
        "Velocity",
        lambda: Velocity(
            jnp.array(0.5, dtype=jnp.float32),
            jnp.array(-0.5, dtype=jnp.float32),
        ),
    ),
    (
        "Direction",
        lambda: Direction(jnp.array([1.0, 0.0], dtype=jnp.float32)),
    ),
    (
        "Team",
        lambda: Team(jnp.array(3, dtype=jnp.int32)),
    ),
    (
        "Carriable",
        lambda: Carriable(
            jnp.array(7, dtype=jnp.int32),
            jnp.array(False),
        ),
    ),
    (
        "Flag",
        lambda: Flag(
            Position(jnp.array(1.0, dtype=jnp.float32), jnp.array(2.0, dtype=jnp.float32)),
            Team(jnp.array(0, dtype=jnp.int32)),
            Carriable(jnp.array(5, dtype=jnp.int32), jnp.array(True)),
        ),
    ),
    (
        "Goal",
        lambda: Goal(
            Position(jnp.array(-3.0, dtype=jnp.float32), jnp.array(4.0, dtype=jnp.float32))
        ),
    ),
    (
        "ContinuousAction",
        lambda: ContinuousAction(
            value=jnp.array([0.1, -0.2], dtype=jnp.float32),
            shape=jnp.array([2], dtype=jnp.int32),
            low=jnp.array([-1.0, -1.0], dtype=jnp.float32),
            high=jnp.array([1.0, 1.0], dtype=jnp.float32),
            dtype=jnp.dtype("float32"),
        ),
    ),
    (
        "DiscreteAction",
        lambda: DiscreteAction(
            value=jnp.array(2, dtype=jnp.int32),
            n=jnp.array(5, dtype=jnp.int32),
            dtype=jnp.dtype("int32"),
        ),
    ),
    (
        "MultiDiscreteAction",
        lambda: MultiDiscreteAction(
            value=jnp.array([1, 2], dtype=jnp.int32),
            n=jnp.array([3, 4], dtype=jnp.int32),
            dtype=jnp.dtype("int32"),
        ),
    ),
    (
        "Obstacle",
        lambda: Obstacle(
            Position(jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
        ),
    ),
    (
        "Renderable",
        lambda: Renderable(jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)),
    ),
]


@pytest.mark.parametrize("name,factory", _COMPONENT_FACTORIES)
def test_component_tree_roundtrip(name, factory):
    instance = factory()
    children, aux = instance.tree_flatten()
    restored = type(instance).tree_unflatten(aux, children)
    assert isinstance(restored, type(instance))
    assert jtu.tree_structure(restored) == jtu.tree_structure(instance)

    original_leaves = [np.asarray(x) for x in jtu.tree_leaves(instance)]
    restored_leaves = [np.asarray(x) for x in jtu.tree_leaves(restored)]
    assert len(original_leaves) == len(restored_leaves)
    for original, new in zip(original_leaves, restored_leaves):
        np.testing.assert_equal(original, new)


def test_components_module_exports_expected_symbols():
    for name, _ in _COMPONENT_FACTORIES:
        assert hasattr(components_module, name)
    assert hasattr(components_module, "Goal")
