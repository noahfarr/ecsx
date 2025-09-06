import jax.numpy as jnp
import pytest
from ecsx.core.component_specification import ComponentSpecification


def test_validate_value_success():
    spec = ComponentSpecification(
        "Foo", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )
    arr = spec.validate_value([1.0, 2.0])
    assert arr.shape == (2,)
    assert arr.dtype == jnp.float32


def test_validate_value_shape_mismatch():
    spec = ComponentSpecification(
        "Foo", (2,), jnp.float32, jnp.array([0.0, 0.0], jnp.float32)
    )
    with pytest.raises(ValueError):
        spec.validate_value([1.0, 2.0, 3.0])
