import numpy as np
import jax.numpy as jnp
from jax import tree_util as jtu

from ecsx.components import Position
from ecsx.core.component_specification import ComponentSpecification


def _position_proto() -> Position:
    return Position(
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
    )


def test_component_specification_reports_tree_metadata():
    spec = ComponentSpecification("Position", _position_proto())

    assert spec.treedef() == jtu.tree_structure(spec.prototype)

    leaves = [np.asarray(leaf) for leaf in spec.leaves()]
    assert len(leaves) == 2
    np.testing.assert_equal(leaves[0], np.asarray(0.0, dtype=np.float32))
    np.testing.assert_equal(leaves[1], np.asarray(1.0, dtype=np.float32))

    assert spec.leaf_shapes() == [(), ()]
    assert spec.leaf_dtypes() == [jnp.dtype("float32"), jnp.dtype("float32")]
