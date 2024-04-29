import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import os

def plot_errors(file_path):
    # Load data from the CSV file
   
    data_error = pd.read_csv(os.path.join(file_path, 'error_points.csv'))
    data_similarity = pd.read_csv(os.path.join(file_path, 'similarity_points.csv'))

    # Extract coordinates and error values
    x = data_error['x']
    y = data_error['y']
    z = data_error['z']
    error = data_error['error']
    similarity = data_similarity['similarity']

    # Create a 3D scatter plot with color-coded points based on error values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=error, cmap='viridis')

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Values')
    plt.title('Error HeatMap')

    plt.savefig(os.path.join(file_path, 'error_heatmap.png'))
    plt.close()


    # Create a 3D scatter plot with color-coded points based on similarity values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=similarity, cmap='viridis')

    # Add shading
    scatter.set_edgecolor('face')
    ax.set_axis_off()
    # Add color bar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Similarity Values')
    # plt.title('Similarity HeatMap')

    plt.savefig(os.path.join(file_path, 'similarity_heatmap.png'))
    plt.close()

    # CREATE A GIF FOR BOTH SIMILARITY AND ERROR HEATMAP
    # Create a 3D scatter plot with color-coded points based on custom conditions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Custom colormap with only red and blue colors
    cmap_custom = ListedColormap(['blue', 'red'])
    # Set color based on conditions
    scatter = ax.scatter(x, y, z, c=['red' if e > 1/256 else 'blue' for e in error], cmap=cmap_custom,alpha=0.7)

    # # Add shading
    # scatter.set_edgecolor('face')

    # # Set axis labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # Hide axes
    ax.set_axis_off()

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Values')
    plt.savefig(os.path.join(file_path, 'threshold_error.png'))
    plt.close()

    # Add shading
    scatter.set_edgecolor('face')

    ax.set_axis_off()

    # Add color bar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Error Values')
    plt.savefig(os.path.join(file_path, 'threshold_error.png'))
    plt.close()

    # # Function to update the plot for animation
    # def update(frame):
    #     ax.view_init(elev=10, azim=frame)

    # # Create the animation
    # rotation_animation = FuncAnimation(fig, update, frames=range(0, 360, 10), interval=5)

    # # Save the animation as a GIF
    # rotation_animation.save(os.path.join(file_path,"error_animation.gif"), writer='imagemagick', fps=15)

    # plt.close()


    # # create animation for similarity heatmap without custom conditions

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x, y, z, c=similarity, cmap='viridis')
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Similarity Values')
    # def update(frame):
    #     ax.view_init(elev=10, azim=frame)
    # rotation_animation = FuncAnimation(fig, update, frames=range(0, 360, 10), interval=5)
    # rotation_animation.save(os.path.join(file_path, "similarity_animation.gif"), writer='imagemagick', fps=15)
    # plt.close()

if __name__ == '__main__':
    import sys
    plot_errors(str(sys.argv[1]))
