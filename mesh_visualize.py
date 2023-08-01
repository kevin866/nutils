import meshio
import matplotlib.pyplot as plt

# Load the MSH file
mesh = meshio.read("tests/test_mesh/mesh3d_p1_v4.msh")


# Extract vertices and cells information
points = mesh.points
cells = mesh.cells

import numpy as np
from mayavi import mlab
mlab.init_notebook()  # or mlab.init()

def plot_mesh_3d(points, cells):
    # Extract 3D vertices
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create triangular mesh
    triangles = []
    for cell in cells:
        if cell.type == "tetra":
            triangles.extend(cell.data)

    # Plot the 3D mesh using mayavi
    mlab.figure()
    mlab.triangular_mesh(x, y, z, triangles, representation="wireframe")
    mlab.show()

plot_mesh_3d(points, cells)
