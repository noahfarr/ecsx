from typing import TypeAlias, Protocol

import jax

Array: TypeAlias = jax.Array
Key: TypeAlias = jax.Array

ComponentName: TypeAlias = str
EntityId: TypeAlias = int


class PyTree(Protocol): ...
