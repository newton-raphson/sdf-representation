import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import argparse
import trimesh

def plot_stl(file_path, gif_path, num_frames=10, elevation=30, azimuth=45):
    # Load STL file
    mesh = trimesh.load(file_path)

    # Extract vertices and faces from the Trimesh object
    vertices = mesh.vertices
    faces = mesh.faces

    # Create a 3D plot without axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()  # Turn off axes display

    # Plot the STL mesh with solid gray color
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color='gray', edgecolor='none', shade=True)

    # Set view angle
    ax.view_init(elevation, azimuth)

    # Create GIF frames
    frames = []
    for i in range(num_frames):
        azimuth += 360 / num_frames
        ax.view_init(elevation, azimuth)
        plt.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

    # Save GIF
    imageio.mimsave(gif_path, frames, duration=0.1)

    # Close the plot
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='mesh.stl')
    parser.add_argument('--gif_path', type=str, default='mesh.gif')
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--elevation', type=int, default=30)
    parser.add_argument('--azimuth', type=int, default=45)
    args = parser.parse_args()
    plot_stl(args.file_path, args.gif_path, args.num_frames, args.elevation, args.azimuth)

