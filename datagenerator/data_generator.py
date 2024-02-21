import numpy as np
import trimesh
import igl
import pandas as pd
import os
import argparse
import glob
from utils.constants import RANDOM_SEED_DATA_GENERATION
##############################################################################################################
#####################################UTILITY FUNCTIONS########################################################
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

def create_narrow_band_distribute(uniform_points, triangle_coords, width):
    """
    Creates a narrow band around a given triangle using uniform points and triangle coordinates.

    Args:
        uniform_points (tuple): A tuple of three uniform points.
        triangle_coords (list): A list of three coordinates representing the triangle.
        width (float, optional): The width of the narrow band. Defaults to 0.1.

    Returns:
        numpy.ndarray: narrow band point in both directions from normal of the triange.
        numpy.ndarrat: corresponding norma
    """
    # print(triangle_coords.shape)
    normal = calculate_normal(triangle_coords)

    # print({normal.shape,width.shape})
    normal_width = normal*width
    u, v, w = uniform_points
    point = u * triangle_coords[0] + v * triangle_coords[1] + w * triangle_coords[2]
    return point + normal_width,normal

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
    mag = np.linalg.norm(normal)
    if mag ==0:
        return 
    return normal / np.linalg.norm(normal)
# let's define a function which takes a triangle 
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

def compute_min_max(geometry_path):
    """
    Computes the minimum and maximum values of vertices in the given geometry path.

    Args:
        geometry_path (str): The path to the directory containing the geometry files.

    Returns:
        tuple: A tuple containing the maximum and minimum values of vertices.
    """
    files = glob.glob(geometry_path+"/*")
    max_val=0
    min_val=0
    print(f"Number of files is {len(files)}")
    if os.path.exists(os.path.join(geometry_path,"max_min.txt")):
        # read the max and min values from the file
        with open(os.path.join(geometry_path,"max_min.txt"),"r") as f:
            max_val = float(f.readline())
            min_val = float(f.readline())
        return max_val,min_val

    # go inside the file 
    for file in files:
        # get the .ply file inside it 
        ply_file = glob.glob(file+"/*.ply")
        if len(ply_file) == 0:
            continue
        # for each ply file generate the signed distance data
        for file in ply_file:
            mesh = trimesh.load(file)
            if np.min(mesh.vertices) < min_val:
                min_val = np.min(mesh.vertices)
            if np.max(mesh.vertices) > max_val:
                max_val = np.max(mesh.vertices)
    if not os.path.exists(os.path.join(geometry_path,"max_min.txt")):
        with open(os.path.join(geometry_path,"max_min.txt"),"w") as f:
            f.write(str(max_val)+"\n")
            f.write(str(min_val)+"\n")
    return max_val,min_val
##############################################################################################################
##################################### MAIN FUNCTIONS ##########################################################


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


def write_signed_distance_distributed(geometry_path, data_path, num_points_uniform=0, num_points_surface=0, num_points_narrow_band=0, dense_width=0.1, additional_points=0, path=None):
    """
    Writes signed distance data to CSV files for a given set of geometry files.

    Parameters:
    - geometry_path (str): The path to the directory containing the geometry files.
    - data_path (str): The path to the directory where the CSV files will be saved.
    - num_points_uniform (int): The number of uniformly distributed points to generate for each geometry file.
    - num_points_surface (int): The number of points to generate on the surface of each geometry file.
    - num_points_narrow_band (int): The number of points to generate within the narrow band of each geometry file.
    - dense_width (float, optional): The width of the dense region around the surface. Default is 0.1.
    NOT USED 
    - additional_points (int, optional): The number of additional points to generate for each geometry file. Default is 0. 
    - path (str, optional): The path to a specific geometry file. If provided, only this file will be processed. Default is None.

    Returns:
    - bool: True if the signed distance data is successfully written to the CSV files, otherwise function fails
    """
    ####### set the randomizer #####################
    np.random.seed(RANDOM_SEED_DATA_GENERATION)
    ################################################
    # list all the files in the directory
    files = glob.glob(geometry_path+"/*")
    print(f"Number of files is {len(files)}")
    min_val,max_val = compute_min_max(geometry_path)
    # apply a certain increment to the min and max values
    # to make sure that the points are not on the surface 
    # of the bounding box
    
    # increment by 40% of the max value
    min_val = min_val - 0.4*max_val
    max_val = max_val + 0.4*max_val

    log_file_path = os.path.join(data_path, "processed_files.log")

    # If the log file exists, read the processed files into a set
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            processed_files = set(line.strip() for line in log_file)
        print(f"Number of processed files is {len(processed_files)}")
    else:
        processed_files = set()
    # go inside the file 
    for file in files:
        # get the .ply file inside it 
        ply_file = glob.glob(file+"/*.ply")
        if len(ply_file) == 0:
            continue
        # for each ply file generate the signed distance data
        for file in ply_file:
            if file in processed_files:
                continue
            print("\n##############################################################################################################\n")

            print(f"Processing file {file}")

            print("\n##############################################################################################################\n")

            # rescale the .ply file vertices to be between -1 and 1
            mesh = trimesh.load(file)
            # check point if the mesh is corrupted or not
            if len(mesh.vertices)==1 or len(mesh.faces)==1:
                print(f"\n Continuing: veritces {len(mesh.vertices)},faces {len(mesh.faces)}  \n")
                continue
            # mesh.vertices = (mesh.vertices - min_val)/(max_val-min_val)
            v,f = mesh.vertices, mesh.faces
            # for on surface
            query_on_surface = np.array(v)
            df_on_surface = pd.DataFrame(query_on_surface, columns=['x', 'y', 'z'])
            # df_on_surface=df_on_surface.apply(pd.to_numeric, errors='coerce')
            # df_on_surface=df_on_surface.dropna()
            # S_surface = np.zeros(len(query_on_surface))
            # # no normal at all
            # n_surface = np.zeros_like(query_on_surface)
            # print("Saving")
            # data_surface = np.column_stack((query_on_surface, S_surface,n_surface))
            # df_on_surface = pd.DataFrame(data_surface, columns=['x', 'y', 'z', 'S','nx','ny','nz'])
            
            # # take each triangle and get a point  
            # uniform_narrow_points = np.random.uniform(0, 1, size=(len(f), 3))
            # # to get to the barycentric co-ordinate system
            # uniform_narrow_points /= np.sum(uniform_narrow_points, axis=1, keepdims=True)
            # print(f"The length of narrow_points is {len(uniform_narrow_points)}")
            # # let's take it as 10% of the width
            # width_threshold = 0.1 *np.abs(np.max(v)-np.min(v))
            # uniform_width = np.random.uniform(-width_threshold, width_threshold, size=len(f))
            # query_points_narrow=[]
            # normal_narrow=[]
            # total_triangle = len(f)
            # for i,triangle in enumerate(v[f]):  
            #     print(f"\n The Triangles is {i}/{total_triangle}")
            #     points_narrow, normal=create_narrow_band_distribute(uniform_narrow_points[i],triangle,uniform_width[i])
            #     print(points_narrow)
            #     print("\n")
            #     query_points_narrow.append(points_narrow)
            #     normal_narrow.append(normal)
            #     if i==50:
            #         break
            # # get the point the Signed distance value is the value of the normal
            # # the normal is that value calculated 
            # # create df for this
            # query_on_surface = np.array(query_points_narrow)
            # S_narrow = np.array(uniform_width).reshape(len(query_on_surface),1)
            # # no normal at all
            # n_narrow = np.array(normal_narrow)
            # data_narrow = np.column_stack((query_on_surface, S_narrow,n_narrow))
            # df_narrow_band = pd.DataFrame(data_narrow, columns=['x', 'y', 'z', 'S','nx','ny','nz'])

            # mesh.export(file)
            # we just want to take the vertices and corresponding normals

            # # generate the signed distance data
            # df_uniform_points, df_on_surface, df_narrow_band = generate_signed_distance_data(
            #     file,
            #     num_points_uniform,
            #     num_points_surface,
            #     num_points_narrow_band,
            #     dense_width,
            #     additional_points,
            #     distributed=True
            # )
            # append the data to the csv file
            # df_uniform_points.to_csv(os.path.join(data_path,"uniform.csv"),mode='a',index=False)
            df_on_surface.to_csv(os.path.join(data_path,"surface.csv"),mode='a',index=False)
            # df_narrow_band.to_csv(os.path.join(data_path,"narrow.csv"),mode='a',index=False)
            # add the file to the processed files
            with open(log_file_path, "a") as log_file:
                log_file.write(file + "\n")

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