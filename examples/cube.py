import numpy as np
from nutils import export, function, mesh, solver
from nutils.expression_v2 import Namespace
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable
from matplotlib import cm

nelems = 10
stddev = 0.05
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
stddev = [0.1, 0.1, 0.1]
comp = np.array([1.0-i for i in stddev])
args['geom'] = np.multiply(np.array(args['geom']),comp) + rng.normal(scale=stddev, size=args['geom'].shape)

# Plot the surface of the cube.
smpl = topo.boundary.sample('bezier', 5)
X = smpl.eval(geom, **args)
#export.triplot('surface.png', X, hull=smpl.hull, cmap='blue', linecolor = 'r')
with export.mplfigure('displacement.png') as fig:
    ax = fig.add_subplot(111, projection='3d')

    #meanU = np.array([np.mean(normU[t]) for t in bezier.tri])
    #norm = Normalize(np.min(meanU), np.max(meanU))
    surf = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=smpl.tri)
    #surf.set_fc(coolwarm(norm(meanU)))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')