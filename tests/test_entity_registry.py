import pytest
from ecsx.core.entity_registry import EntityRegistry


def test_spawn_and_despawn():
    reg = EntityRegistry.create(capacity=2)
    eid0 = reg.spawn()
    eid1 = reg.spawn()
    assert {eid0, eid1} == {0, 1}
    with pytest.raises(RuntimeError):
        reg.spawn()
    reg.despawn(eid0)
    assert not bool(reg.alive_mask[eid0])
    eid2 = reg.spawn()
    assert eid2 == eid0
