import matplotlib.pyplot as plt
from ecsx.environments import build_grid_world
from ecsx.rendering import GridRenderer

# Build a simple gridworld
env = build_grid_world(num_agents=1, grid_size=(5, 5), goal_position=(4, 4))
renderer = GridRenderer(grid_size=(5, 5))

# Step/reset environment as needed
env.reset()
img = renderer.render(env.world, goal_position=(4, 4))

# Show on screen
plt.imshow(img)
plt.axis("off")
plt.show()
