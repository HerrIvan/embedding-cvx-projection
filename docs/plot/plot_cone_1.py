# Import matplotlib.pyplot for plotting
import matplotlib.pyplot as plt
import numpy as np
from src.embedding_cvx_projection import cone_projection
from icecream import ic

np.set_printoptions(precision=7, suppress=True)

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

# Define the vectors a, b, and c with origin at (0, 0, 0)
center = np.array([0.5, 0.5, 0.2])

factor = 0.5

a = factor * np.array([1, 0, 0]) + (1 - factor) * center
b = factor * np.array([0, 1, 0]) + (1 - factor) * center
c = factor * np.array([0, 0, 1]) + (1 - factor) * center

# normalize the vectors
a /= np.linalg.norm(a)
b /= np.linalg.norm(b)
c /= np.linalg.norm(c)


# Plot the vectors
for vec, color, label in zip([a, b, c], ['r', 'g', 'b'], ['a', 'b', 'c']):
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, arrow_length_ratio=0.1, label=label, alpha=0.4)


# generate a vector outside the convex cone
v = 0.8 * np.array([0.0, 1.0, 0.5]) + 0.2 * center

# normalize the vector
v /= np.linalg.norm(v)

# plot the vector in black
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='k',
          arrow_length_ratio=0.1, label='v')

# compute the projection coefficients of v on the convex cone
value, x = cone_projection(np.array([a, b, c]).T, v, beta=0)

print(np.array([a, b, c]) @ x)
print(value)

# get the projection of v on the convex cone
v_proj = x[0] * a + x[1] * b + x[2] * c

ic(x)

# plot the projection
ax.quiver(0, 0, 0, v_proj[0], v_proj[1], v_proj[2],
          color='grey', arrow_length_ratio=0.1, label='v_proj')


# plot the contribution of the vectors a, b, and c to the projection
for i, (vp, co) in enumerate(zip([a, b, c], ['r', 'g', 'b'])):
    print("--")
    v_c = vp*x[i]
    print("--")
    ax.quiver(0, 0, 0,
              v_c[0], v_c[1], v_c[2],
              color=co, arrow_length_ratio=0.1, alpha=0.7)




# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.legend()
ax.set_title('Convex Cone and projection of v to it')


ax.set_aspect('equal')

# set the elevation to 36 degrees and the azimuth to 96 degrees
ax.view_init(30, 110)

# set figure size
fig.set_size_inches(6, 6)

# Show the plot
plt.show()
# save the plot
fig.savefig('cone_1.png')