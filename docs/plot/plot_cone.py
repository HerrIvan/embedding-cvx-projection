# Import matplotlib.pyplot for plotting
import matplotlib.pyplot as plt
import numpy as np

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
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, arrow_length_ratio=0.1, label=label)


# plot random convex combinations of the vectors
for i in range(1000):
    # generate random coefficients
    coeffs = np.random.rand(3)
    coeffs /= np.sum(coeffs)
    # generate random convex combination
    vec = coeffs[0] * a + coeffs[1] * b + coeffs[2] * c

    # normalize the vector
    vec /= np.linalg.norm(vec)

    # plot the convex combination
    # set the marks without the edges but with fill
    ax.scatter(vec[0], vec[1], vec[2], color='k', marker='.', alpha=0.3, edgecolors='none')

    # ax.scatter(vec[0], vec[1], vec[2], color='k', marker='.', alpha=0.3)



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
fig.savefig('cone.png')