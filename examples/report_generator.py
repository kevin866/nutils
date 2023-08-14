import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting functionality
import numpy as np
from cube import get_cube
from irregular_cube import main
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable
import itertools

# Create a 4x5 grid of subplots
stddev = [0.1, 0.1, 0.1]
# step = 0.1
# arr = np.linspace(0.1, 0.3, num=3)
# std = [[round(i,2)]*3 for i in arr]
arr = np.linspace(0.1, 0.4, num=4)
arr = [round(i,2) for i in arr]
std = list(itertools.permutations(arr, 3))
print(std)
num_rows = 6
num_cols = 4
fig = plt.figure(figsize=(8.27,11.69), constrained_layout=True)
normU, bezier, X = main()
x, smpl = get_cube()

# Fill in the rest of the subplots with the same 3D plot
for i in range(1, num_rows * num_cols + 1):
    stddev=std[(i-1)//2]
    if i % 2 !=0:
        x, smpl = get_cube(stddev=stddev)
        ax = fig.add_subplot(6, 4, i, projection='3d')
        surf = ax.plot_trisurf(x[:, 0], x[:, 1], x[:, 2], triangles=smpl.tri)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        ax.set_title('     X {X}, Y {Y}, Z {Z}'.format(X = stddev[0], Y = stddev[1], Z = stddev[2]))  # Add a title
    else:
        normU, bezier, X = main(stddev=stddev)
        ax = fig.add_subplot(6, 4, i, projection='3d')
        meanU = np.array([np.mean(normU[t]) for t in bezier.tri])
        norm = Normalize(np.min(meanU), np.max(meanU))
        surf = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=bezier.tri)
        surf.set_fc(coolwarm(norm(meanU)))

        cbar = fig.colorbar(ScalarMappable(cmap=coolwarm, norm=norm), ax=ax)
        cbar.set_label('Displacement')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
# Export the figure as a PNG image
plt.savefig("my_figure.png", dpi=300)

# Export the figure as a PDF file
plt.savefig("my_figure.pdf")
# Set a title for the entire figure

# Display the plot
plt.show()
# Define the number of rows and columns for subplots
