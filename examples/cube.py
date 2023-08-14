import numpy as np
from nutils import export, function, mesh, solver
from nutils.expression_v2 import Namespace
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable
from matplotlib import cm
def get_cube(stddev = [0.1, 0.1, 0.1]):
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
    rng = np.random.default_rng(seed=0) # `seed=0` for reproducibility
    comp = np.array([1.0-i for i in stddev])
    args['geom'] = np.multiply(np.array(args['geom']),comp) + rng.normal(scale=stddev, size=args['geom'].shape)

    # Plot the surface of the cube.
    smpl = topo.boundary.sample('bezier', 5)
    X = smpl.eval(geom, **args)
    return X, smpl
#export.triplot('surface.png', X, hull=smpl.hull, cmap='blue', linecolor = 'r')
"""with export.mplfigure('displacement.png') as fig:
    ax = fig.add_subplot(2,5,1, projection='3d')

    #meanU = np.array([np.mean(normU[t]) for t in bezier.tri])
    #norm = Normalize(np.min(meanU), np.max(meanU))
    surf = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=smpl.tri)
    #surf.set_fc(coolwarm(norm(meanU)))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    """
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting functionality
# import numpy as np


# # Create a 4x5 grid of subplots
# num_rows = 4
# num_cols = 5
# fig = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)

# # Fill in the rest of the subplots with the same 3D plot
# for i in range(1, num_rows * num_cols + 1):
    
#     ax = fig.add_subplot(4, 5, i, projection='3d')
#     surf = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=smpl.tri)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title('3D Plot')  # Add a title
    


# # Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# # Set a title for the entire figure

# # Display the plot
# plt.show()
