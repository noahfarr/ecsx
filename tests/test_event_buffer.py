import jax.numpy as jnp
from ecsx.core.event_specification import EventSpecification
from ecsx.core.event_buffer import EventBuffer


def test_event_buffer_operations():
    spec = EventSpecification("Evt", (2,), jnp.int32, capacity=3)
    buf = EventBuffer.from_specification(spec)
    assert int(buf.count) == 0
    payloads = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.int32)
    buf = buf.overwrite(payloads, jnp.array(2, jnp.int32))
    assert int(buf.count) == 2
    assert jnp.array_equal(buf.data, payloads)
    buf = buf.clear()
    assert int(buf.count) == 0
