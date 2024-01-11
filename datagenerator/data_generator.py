import numpy as np
import trimesh
import igl
import pandas as pd
import os
import argparse
import glob
from utils.constants import RANDOM_SEED_DATA_GENERATION
def transform_to_triangle_space(lhs_point, triangle_coords):
    """
    Transforms a point in 3D space to a point in the triangle space defined by the given triangle coordinates.

    Args:
        lhs_point (tuple): A tuple of three floats representing the coordinates of the point in 3D space.
        triangle_coords (tuple): A tuple of three tuples, each containing three floats representing the coordinates of the vertices of the triangle in 3D space.
        dense_width (int): The width of the dense grid.

    Returns:
        tuple: points inside the triangle space in 3D space.
    """
    u, v, w = lhs_point
    return u * triangle_coords[0] + v * triangle_coords[1] + w * triangle_coords[2]
    
def change_format(csv_path):
    """
    Changes the format of the CSV file to the format required by the model.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        pandas.DataFrame: DataFrame containing the data in the required format.
    """
    df = pd.read_csv(csv_path)
    
    return df

def create_narrow_band(uniform_points, triangle_coords, width=0.1):
    """
    Creates a narrow band around a given triangle using uniform points and triangle coordinates.

    Args:
        uniform_points (tuple): A tuple of three uniform points.
        triangle_coords (list): A list of three coordinates representing the triangle.
        width (float, optional): The width of the narrow band. Defaults to 0.1.

    Returns:
        numpy.ndarray: narrow band point in both directions from normal of the triange.
    """
    normal_width = calculate_normal(triangle_coords) * width
    u, v, w = uniform_points
    point = u * triangle_coords[0] + v * triangle_coords[1] + w * triangle_coords[2]
    return point + normal_width

def calculate_normal(triangle):
    """
    Calculates the normal vector of a triangle defined by three points.

    Args:
        triangle (numpy.ndarray): A 3x3 array of points defining the triangle.

    Returns:
        numpy.ndarray: A 1x3 array representing the normal vector of the triangle.
    """
    AB = triangle[1] - triangle[0]
    AC = triangle[2] - triangle[0]
    normal = np.cross(AB, AC)
    return normal / np.linalg.norm(normal)
# Not used 
# Tested but didn't work

# def calculate_mean_curvature(mesh, v, f, radius):
#     mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, v, radius)
#     # Average mean curvature values for vertices that belong to each triangle
#     triangle_mean_curvature = [np.mean(mean_curvature[triangle]) for triangle in f]
#     return np.array(triangle_mean_curvature)


def generate_test_points(v,f,size_cube):
    np.meshgrid()
    pass

def generate_occupancy(cube_size, geometry_path):
    """
    Generate occupancy data for a given mesh geometry.

    Args:
    - cube_size (int): Size of the cube.
    - geometry_path (str): Path to the mesh geometry file.

    Returns:
    - df (pandas.DataFrame): DataFrame containing occupancy values for points sampled in the cube.
    """

    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, cube_size),
                             np.linspace(-1, 1, cube_size),
                             np.linspace(-1, 1, cube_size))
    query_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    mesh = trimesh.load(geometry_path)
    v, f = mesh.vertices, mesh.faces
    if query_points.shape[0] == 0:
        query_points = np.array([[0, 0, 0]], dtype=np.float32)  # Convert to float32
        S, _, _,n = igl.signed_distance(query_points, v, f, return_normals=True)
        S=np.array([-0.5])
        n=np.array([[0,0,0]])
    else:
        if len(query_points)> 500000:
            print("The points are more than 500000")
            batch_size = 100000
            S=np.array([])
            for i in range(0, len(query_points), batch_size):
                S_in, _, _ = igl.signed_distance(query_points[i:i+batch_size], v, f, return_normals=False)
                S=np.append(S,S_in)
                # if i==0:
                #     n=np.array(n_in)
                # else:
                #     n_in = np.array(n_in)
                #     n= np.concatenate((n,n_in),axis=0)

        else:
            S, _, _= igl.signed_distance(query_points, v, f, return_normals=False)
    # print(f"Shapes of S is {S.shape} and {n.shape}")
    data = np.column_stack((query_points, np.sign(S)))
    df = pd.DataFrame(data, columns=['x', 'y', 'z','S'])
    df.to_csv("occupancy.csv",index=False)
    return df
# TO GENERAATE MISMATCH DATA AFTER POST PROCESSING
def write_signed_distance_mismatch(query_points, geometry_path):
    mesh = trimesh.load(geometry_path)
    v, f = mesh.vertices, mesh.faces
    if query_points.shape[0] == 0:
        query_points = np.array([[0, 0, 0]], dtype=np.float32)  # Convert to float32
        S, _, _,n = igl.signed_distance(query_points, v, f, return_normals=True)
        S=np.array([-0.5])
        n=np.array([[0,0,0]])
    else:
        if len(query_points)> 500000:
            print("The points are more than 500000")
            batch_size = 100000
            S=np.array([])
            for i in range(0, len(query_points), batch_size):
                S_in, _, _,n_in = igl.signed_distance(query_points[i:i+batch_size], v, f, return_normals=True)
                print(f"Shapes of S in is {S_in.shape} and n is {n_in.shape}")
                S=np.append(S,S_in)
                if i==0:
                    n=np.array(n_in)
                else:
                    n_in = np.array(n_in)
                    n= np.concatenate((n,n_in),axis=0)

        else:
            S, _, _,n = igl.signed_distance(query_points, v, f, return_normals=True)
    print(f"Shapes of S is {S.shape} and {n.shape}")
    data = np.column_stack((query_points, S,n))
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'S','nx','ny','nz'])
    return df
# handle distributed file system
# specifically created to handle something like 
# DAVID with 1 billion triangles
def write_signed_distance_distributed(geometry_path,data_path,num_points_uniform, num_points_surface, num_points_narrow_band,dense_width=0.1,additional_points=0,path=None):
    # list all the files in the directory
    files = glob.glob(geometry_path+"/*")
    print(f"Number of files is {len(files)}")

    # go inside the file 
    for file in files:
        # get the .ply file inside it 
        ply_file = glob.glob(file+"/*.ply")
        if len(ply_file) == 0:
            continue
        # for each ply file generate the signed distance data
        for file in ply_file:
            # generate the signed distance data
            df_uniform_points, df_on_surface, df_narrow_band = generate_signed_distance_data(
                file,
                num_points_uniform,
                num_points_surface,
                num_points_narrow_band,
                dense_width,
                additional_points,
                distributed=True
            )
            # append the data to the csv file
            df_uniform_points.to_csv(os.path.join(data_path,"uniform.csv"),mode='a',index=False)
            df_on_surface.to_csv(os.path.join(data_path,"surface.csv"),mode='a',index=False)
            df_narrow_band.to_csv(os.path.join(data_path,"narrow.csv"),mode='a',index=False)
    return True

# main function to generate signed distance data
def generate_signed_distance_data(geometry,num_points_uniform, num_points_surface, num_points_narrow_band,dense_width=0.1,additional_points=0,path=None,distributed=False):
    """
    Generate signed distance data for a given mesh geometry.

    Args:
    - geometry (str): Path to the mesh geometry file.
    - num_points_uniform (int): Number of uniformly sampled points inside the bounding box of the mesh.
    - num_points_surface (int): Number of points to sample on the surface of the mesh.
    - num_points_narrow_band (int): Number of points to sample in the narrow band region of the mesh.
    - dense_width (float): Width of the narrow band region.

    Returns:
    - df_uniform_points (pandas.DataFrame): DataFrame containing signed distance values for uniformly sampled points.
    - df_on_surface (pandas.DataFrame): DataFrame containing signed distance values for points sampled on the surface of the mesh.
    - df_narrow_band (pandas.DataFrame): DataFrame containing signed distance values for points sampled in the narrow band region of the mesh.
    """
    # Load the mesh
    mesh = trimesh.load(geometry)

    # # Check if mesh is within bounds
    # if not np.all(np.logical_and(mesh.vertices >= -1, mesh.vertices <= 1)):
    #     raise ValueError("Mesh is out of bounds. Please ensure that the mesh is bounded between -1 and 1.")
    # Get mesh data
    v, f = mesh.vertices, mesh.faces
    # using constant seed for reproducibility
    np.random.seed(RANDOM_SEED_DATA_GENERATION)
    # Generate random points
    points_inside_box = num_points_uniform
    print("Generating points")

    query_uniform_points = np.random.uniform(-1, 1, size=(int(points_inside_box), 3))
    print("uniform points generated")
    if distributed:
        min_value = np.min(np.min(mesh.vertices, axis=0))*1.1
        max_value = np.max(np.max(mesh.vertices, axis=0))*1.1
        query_uniform_points = np.random.uniform(min_value, max_value, size=(int(points_inside_box), 3))
    # Calculate mean curvature for each triangle
    # triangle_mean_curvature = calculate_mean_curvature(mesh, v,f, 0.1)

    # # Define adaptive sampling factor based on mean curvature
    # sampling_det = triangle_mean_curvature > np.mean(triangle_mean_curvature) 
    # Sample points

    query_on_surface = []
    query_narrow_band = []
    inside_points = points_inside_box

    for i,triangle in enumerate(v[f]):  
        if(i%100==0):
            print(f"This is the {i}th triangle")
        # points_on_surface = num_points_surface + sampling_det[i]*additional_points
        points_on_surface = num_points_surface
        uniform_surface_points = np.random.uniform(0, 1, size=(points_on_surface, 3))
        uniform_surface_points /= np.sum(uniform_surface_points, axis=1, keepdims=True)

        for uniform_surface_point in uniform_surface_points:
            point = transform_to_triangle_space(uniform_surface_point, triangle)
            query_on_surface.append(point)

        uniform_narrow_points = np.random.uniform(0, 1, size=(num_points_surface, 3))
        uniform_narrow_points /= np.sum(uniform_narrow_points, axis=1, keepdims=True)
        uniform_width = np.random.uniform(-dense_width, dense_width, size=num_points_narrow_band)

        for width, point_uniform in zip(uniform_width, uniform_narrow_points):
            if width < 0:
                inside_points += 1
            point = create_narrow_band(point_uniform, triangle, width)
            query_narrow_band.append(point)

    query_on_surface = np.array(query_on_surface)
    query_narrow_band = np.array(query_narrow_band)

    print(query_on_surface.shape)
        # Write signed distance data to CSV
    def write_signed_distance(query_points, v, f,key):
        if query_points.shape[0] == 0:
            query_points = np.array([[0, 0, 0]], dtype=np.float32)  # Convert to float32
            S, _, _,n = igl.signed_distance(query_points, v, f, return_normals=True)
            S=np.array([-0.5])
            n=np.array([[0,0,0]])
        else:
            if len(query_points)> 500000:
                print("The points are more than 500000")
                batch_size = 100000
                S=np.array([])
                for i in range(0, len(query_points), batch_size):
                    S_in, _, _,n_in = igl.signed_distance(query_points[i:i+batch_size], v, f, return_normals=True)
                    print(f"Shapes of S in is {S_in.shape} and n is {n_in.shape}")
                    S=np.append(S,S_in)
                    if i==0:
                        n=np.array(n_in)
                    else:
                        n_in = np.array(n_in)
                        n= np.concatenate((n,n_in),axis=0)

            else:
                S, _, _,n = igl.signed_distance(query_points, v, f, return_normals=True)
        print(f"Shapes of S is {S.shape} and {n.shape}")
        data = np.column_stack((query_points, S,n))
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'S','nx','ny','nz'])
        return df
    df_on_surface = write_signed_distance(query_on_surface, v, f,"surface")
    df_uniform_points = write_signed_distance(query_uniform_points, v, f,"uniform")
    # print("uniform points generated and signed distance computed")
    # df_on_surface = write_signed_distance(query_on_surface, v, f,"surface")
    df_narrow_band = write_signed_distance(query_narrow_band, v, f,"narrow_band")
    print("point generation completed")
    return df_uniform_points, df_on_surface, df_narrow_band

def parse_args():
    parser = argparse.ArgumentParser(description="Generate signed distance data for a mesh geometry.")
    parser.add_argument("geometry", type=str, help="Path to the mesh geometry file")
    parser.add_argument("--num_uniform", type=int, default=10, help="Number of uniformly sampled points inside the bounding box of the mesh")
    parser.add_argument("--num_surface", type=int, default=1, help="Number of points to sample on the surface of the mesh")
    parser.add_argument("--num_narrow_band", type=int, default=1, help="Number of points to sample in the narrow band region of the mesh")
    parser.add_argument("--dense_width", type=float, default=0.1, help="Width of the narrow band region")

    return parser.parse_args()

def main():
    args = parse_args()
    df_uniform_points, df_on_surface, df_narrow_band = generate_signed_distance_data(
        args.geometry,
        args.num_uniform,
        args.num_surface,
        args.num_narrow_band,
        args.dense_width
    )
    print(df_uniform_points)
    # Optionally, you can save the DataFrames to CSV files or perform other operations here.
    dataframes = [("uniform", df_uniform_points), ("on_surface", df_on_surface), ("narrow_band", df_narrow_band)]

    for name, df in dataframes:
        df.to_csv(f"{name}.csv", index=False)

if __name__ == "__main__":
    main()