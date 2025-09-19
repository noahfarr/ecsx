import numpy as np
import pytest

from ecsx.core.entity_registry import EntityRegistry


def test_spawn_and_despawn_cycle():
    registry = EntityRegistry.create(2)

    first = registry.spawn()
    second = registry.spawn()
    assert {first, second} == {0, 1}
    np.testing.assert_array_equal(registry.alive_mask, np.array([True, True]))

    with pytest.raises(RuntimeError):
        registry.spawn()

    registry.despawn(first)
    np.testing.assert_array_equal(registry.alive_mask, np.array([False, True]))

    recycled = registry.spawn()
    assert recycled == first

    registry.despawn(recycled)
    registry.despawn(second)
    np.testing.assert_array_equal(registry.alive_mask, np.array([False, False]))


def test_despawn_of_dead_entity_is_noop():
    registry = EntityRegistry.create(1)
    registry.despawn(0)
    np.testing.assert_array_equal(registry.alive_mask, np.array([False]))
