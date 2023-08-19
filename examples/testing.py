# from cube import get_cube
# X, smpl = get_cube()
# print(X.shape)
# print('yes')
# print(smpl.tri.shape)
# print(smpl.points)
# print(dir(smpl))
import numpy as np
from nutils import export, function, mesh, solver

def get_volumetric_spline(stddev=[0.1, 0.1, 0.1], seed=0):
    nelems = 4
    degree = 3

    # Create a unit cube.
    topo, cube = mesh.rectilinear([np.linspace(0, 1, nelems + 1)] * 3)

    # Define deformed geometry `geom` in terms of a spline basis and argument `geom`.
    basis = topo.basis('spline', degree=degree)
    geom = basis @ function.Argument('geom', shape=(len(basis), 3))

    # Initialize argument `geom` by projecting the initial geometry `cube` onto the basis.
    args = solver.optimize('geom,', topo.integral(np.sum((cube - geom)**2) * function.J(cube), degree=2 * degree))

    # Deform the geometry by adding a random offset to argument `geom`.
    rng = np.random.default_rng(seed=seed)  # `seed=0` for reproducibility
    comp = np.array([1.0 - i for i in stddev])
    args['geom'] = np.multiply(np.array(args['geom']), comp) + rng.normal(scale=stddev, size=args['geom'].shape)

    # Evaluate the basis functions at each point in the 3D grid and sum the weighted control points.
    volumetric_bspline = sum(args['geom'][i] * basis[i] for i in range(len(basis)))

    # Return the volumetric B-spline and the mesh topo.
    return volumetric_bspline, topo


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Call the function to get the volumetric B-spline and the mesh topo.
volumetric_bspline, topo = get_volumetric_spline()


# Generate a grid of points within the unit cube
num_points = 20
grid = np.linspace(0, 1, num_points)
grid_x, grid_y, grid_z = np.meshgrid(grid, grid, grid)

# Evaluate the volumetric B-spline at the grid points
evaluated_points = np.empty_like(grid_x)
for i in range(num_points):
    for j in range(num_points):
        for k in range(num_points):
            point = [grid_x[i, j, k], grid_y[i, j, k], grid_z[i, j, k]]
            evaluated_points[i, j, k] = np.dot(point, volumetric_bspline)

# Flatten the grid and evaluated points arrays for scatter plot
grid_x_flat = grid_x.flatten()
grid_y_flat = grid_y.flatten()
grid_z_flat = grid_z.flatten()
evaluated_points_flat = evaluated_points.flatten()

# Plot the volumetric spline
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use the flattened arrays for scatter plot and color mapping
sc = ax.scatter(grid_x_flat, grid_y_flat, grid_z_flat, c=evaluated_points_flat, cmap='coolwarm', marker='o')
fig.colorbar(sc, ax=ax, label='Volumetric Spline Value')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Volumetric Spline')

plt.show()