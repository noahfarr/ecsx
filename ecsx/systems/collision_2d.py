import jax.numpy as jnp
from ecsx.query.query_engine import select

def collision_2d(collision_event_name: str = "Collision", event_capacity_multiplier: int = 4):
    def _system(world, key, inputs):
        # Entities that can collide: must have Position + Collider
        mask = select(world, required=("Position", "Collider"))
        idx = jnp.nonzero(mask, size=world.capacity, fill_value=-1)[0]
        n = jnp.sum(mask)

        pos = world.component_stores["Position"].read(idx[:n])         # (n,2)
        rad = world.component_stores["Collider"].read(idx[:n])[:, 0:1] # (n,1)

        # pairwise distances (naive O(n^2) – OK for small n)
        # Build (i<j) pairs
        ii = jnp.arange(n)
        I = jnp.repeat(ii, n)
        J = jnp.tile(ii, n)
        pair_mask = I < J

        pi = pos[I[pair_mask]]
        pj = pos[J[pair_mask]]
        ri = rad[I[pair_mask]]
        rj = rad[J[pair_mask]]
        d2 = jnp.sum((pi - pj) ** 2, axis=1)
        th = (ri + rj)[:, 0] ** 2
        hit = d2 <= th

        pairs_i = idx[I[pair_mask]][hit]
        pairs_j = idx[J[pair_mask]][hit]
        pairs = jnp.stack([pairs_i, pairs_j], axis=1) if pairs_i.shape[0] > 0 else jnp.zeros((0, 2), jnp.int32)

        # Write to event buffer, padded
        bufcap = world.event_buffers[collision_event_name].specification.capacity
        pad = jnp.full((bufcap, 2), -1, jnp.int32)
        k = jnp.minimum(pairs.shape[0], bufcap)
        pad = pad.at[:k].set(pairs[:k])
        world = world.write_event_buffer(collision_event_name, pad, jnp.asarray(k, jnp.int32))
        return world
    return _system

