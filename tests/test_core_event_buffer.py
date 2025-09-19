import numpy as np
import jax.numpy as jnp

from ecsx.core.event_buffer import EventBuffer
from ecsx.core.event_specification import EventSpecification


def _spec() -> EventSpecification:
    return EventSpecification(
        name="Hit",
        shape=(2,),
        dtype=jnp.int32,
        capacity=3,
    )


def test_event_buffer_initial_state():
    buf = EventBuffer.from_specification(_spec())
    np.testing.assert_array_equal(buf.data, np.zeros((3, 2), dtype=np.int32))
    assert int(buf.count) == 0


def test_overwrite_and_clear():
    buf = EventBuffer.from_specification(_spec())
    payloads = jnp.arange(6, dtype=jnp.int32).reshape(3, 2)
    updated = buf.overwrite(payloads, jnp.array(2, dtype=jnp.int32))

    np.testing.assert_array_equal(updated.data, np.asarray(payloads))
    assert int(updated.count) == 2

    cleared = updated.clear()
    np.testing.assert_array_equal(cleared.data, np.asarray(payloads))
    assert int(cleared.count) == 0


def test_event_buffer_tree_roundtrip():
    buf = EventBuffer.from_specification(_spec())
    payloads = jnp.ones((3, 2), dtype=jnp.int32)
    buf = buf.overwrite(payloads, jnp.array(3, dtype=jnp.int32))

    children, aux = buf.tree_flatten()
    restored = EventBuffer.tree_unflatten(aux, children)

    np.testing.assert_array_equal(restored.data, np.asarray(payloads))
    assert int(restored.count) == 3
