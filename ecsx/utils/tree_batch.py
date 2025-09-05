# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Iterable, List
import jax
from jax import tree_util
import jax.numpy as jnp

def stack_trees(trees: Iterable):
    """Stack a list/iterable of identical PyTrees along a new leading axis."""
    trees = list(trees)
    assert trees, "stack_trees: empty input"
    return tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees, is_leaf=None)

def unstack_tree(batched_tree):
    """Unstack a batched PyTree (leading axis B) into a list of PyTrees."""
    # infer B from first leaf
    leaves, treedef = tree_util.tree_flatten(batched_tree)
    B = leaves[0].shape[0]
    return [tree_util.tree_map(lambda x, i=i: x[i], batched_tree) for i in range(B)]

