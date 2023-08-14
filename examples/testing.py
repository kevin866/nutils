import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting functionality
from cube import get_cube
from irregular_cube import main
from matplotlib.pyplot import Normalize
from matplotlib.cm import coolwarm, ScalarMappable
import itertools
# Define the number of rows and columns for the main subplot grid
num_rows = 6
num_cols = 2

# Calculate the figsize to fit within A4 paper size
a4_width = 8.27
a4_height = 11.69
a4_aspect_ratio = a4_width / a4_height
fig_aspect_ratio = num_cols / num_rows
fig_height = a4_height if a4_aspect_ratio >= fig_aspect_ratio else a4_width / fig_aspect_ratio

# Create a figure for the main subplot grid
fig = plt.figure(figsize=(a4_width, fig_height), constrained_layout=True)
gs = fig.add_gridspec(num_rows, num_cols)
normU, bezier, X = main()
x, smpl = get_cube()
# Common title for the 1 by 2 subplots
common_title = 'Common Title for 1 by 2 Subplots'

# Iterate through the main subplot grid
for i in range(num_rows):
    for j in range(num_cols):
        ax = fig.add_subplot(gs[i, j])

        # Create a 1 by 2 subplot within the current subplot
        inner_gs = gs[i, j].subgridspec(1, 2, wspace=0.4)
        ax1 = fig.add_subplot(inner_gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(inner_gs[0, 1], projection='3d')

        # Generate some sample data
        surf = ax1.plot_trisurf(x[:, 0], x[:, 1], x[:, 2], triangles=smpl.tri)
       
        meanU = np.array([np.mean(normU[t]) for t in bezier.tri])
        norm = Normalize(np.min(meanU), np.max(meanU))
        surf = ax2.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=bezier.tri)
        surf.set_fc(coolwarm(norm(meanU)))
        cbar = fig.colorbar(ScalarMappable(cmap=coolwarm, norm=norm), cax=ax2)
        cbar.set_label('Displacement')
        # Plot the data on the left subplot
        
        # ax1.set_title('Location: ({}, {}) - Subplot 1'.format(i, j))

        # Plot the data on the right subplot
        # ax2.plot(x, -y, color='orange')  # Plot negative y to show a different line
        # ax2.set_title('Location: ({}, {}) - Subplot 2'.format(i, j))
        
        # Add common title to the 1 by 2 subplot
        ax.set_title(common_title, fontsize=12, fontweight='bold')
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.savefig("my_figure.png", dpi=300)

# Show the plot
plt.show()