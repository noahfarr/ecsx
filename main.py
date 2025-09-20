import numpy as np
import jax
import jax.numpy as jnp

from ecsx.core.world import World
from ecsx.core.component_specification import ComponentSpecification
from ecsx.components import (
    Position,
    Velocity,
    Move,
)
from ecsx.systems.intent import _move_system


def main() -> None:
    seed = 0

    key = jax.random.key(seed)

    position = ComponentSpecification(
        name="Position",
        prototype=Position(x=0.0, y=0.0),
    )
    velocity = ComponentSpecification(
        name="Velocity",
        prototype=Velocity(vx=0.0, vy=0.0),
    )
    move = ComponentSpecification(
        name="Move",
        prototype=Move(vx=0.0, vy=0.0),
    )

    key, world_key = jax.random.split(key)
    world = World(capacity=8, key=world_key)
    world.register_component(position).register_component(velocity).register_component(
        move
    )

    for i in range(3):
        position = Position(
            x=np.random.uniform(-1.0, 1.0), y=np.random.uniform(-1.0, 1.0)
        )
        velocity = Velocity(
            vx=np.random.uniform(-1.0, 1.0), vy=np.random.uniform(-1.0, 1.0)
        )
        move = Move(vx=1.0, vy=0.0)
        eid = world.spawn(Position=position, Velocity=velocity, Move=move)

    print("Positions:")
    print(world.get_store("Position").data)

    world.add_systems(_move_system)
    world = world.step(inputs={"move_store": "Move", "position_store": "Position"})

    print("Positions:")
    print(world.get_store("Position").data)


if __name__ == "__main__":
    main()
