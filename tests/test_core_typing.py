import jax
from typing import Any

from ecsx.core import typing as ecsx_typing


def test_type_aliases_point_to_expected_types():
    assert ecsx_typing.Key is jax.Array
    assert ecsx_typing.Array is jax.Array
    assert ecsx_typing.ComponentName is str
    assert ecsx_typing.EntityId is int
    assert ecsx_typing.PyTree is Any
