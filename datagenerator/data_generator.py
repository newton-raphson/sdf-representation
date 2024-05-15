import numpy as np
import trimesh
import igl
import pandas as pd
import os
import argparse
import glob
from utils.constants import RANDOM_SEED_DATA_GENERATION
import gmsh
import math
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
    normal = calculate_normal(triangle_coords)
    if normal is None:
        normal = np.array([0,0,0])
    normal_width = normal* width
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
    # replace NoneType with 0
    if normal is None:
        normal = np.array([0,0,0])

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
class KDTree:
    def __init__(self, polygon_points):
        self.data = np.array([(polygon_points[i] + polygon_points[i + 1]) / 2 for i in range(len(polygon_points) - 1)])
        self.data = np.append(self.data, [(polygon_points[-1] + polygon_points[0]) / 2], axis=0)
    
    def knnSearch(self, query_pt, num_results, ret_index, out_dist_sqr):
        distances = np.sum((self.data - query_pt[:-1] * np.ones(self.data.shape)) ** 2, axis=1)
        sorted_indices = np.argsort(distances)
        ret_index[:num_results] = sorted_indices[:num_results]
        out_dist_sqr[:num_results] = distances[ret_index[:num_results]]
        return num_results

def extract_polygon_from_gmsh(filename):
    gmsh.initialize()
    gmsh.open(filename)

    polygon_points = []
    node_tag = 1

    while True:
        try:
            while True:
                v = gmsh.model.mesh.getNode(node_tag)
                polygon_points.append(np.array([v[0][0], v[0][1]]))
                node_tag += 1
        except Exception as e:
            break

    gmsh.finalize()

    polygon_points.append(polygon_points[0])

    return polygon_points

def compute_distance_vector(pt, m_lines, kd_trees, d):
    x = pt[0]
    y = pt[1]
    d = [0.0] * 3
    num_results = 2
    ret_index = np.zeros(num_results, dtype=np.uint32)
    out_dist_sqr = np.zeros(num_results)

    query_pt = np.array([x, y, 0.0])

    num_results = kd_trees.knnSearch(query_pt, num_results, ret_index, out_dist_sqr)

    OnePointVector = np.zeros(len(pt))
    OnePointVectorOtherEnd = np.zeros(len(pt))
    OnePointVector2 = np.zeros(len(pt))
    OnePointVectorOtherEnd2 = np.zeros(len(pt))

    for dim in range(len(pt)-1):
        OnePointVector[dim] = m_lines[ret_index[0]][0][dim] - pt[dim]
        OnePointVectorOtherEnd[dim] = m_lines[ret_index[0]][1][dim] - pt[dim]
        OnePointVector2[dim] = m_lines[ret_index[1]][0][dim] - pt[dim]
        OnePointVectorOtherEnd2[dim] = m_lines[ret_index[1]][1][dim] - pt[dim]

    PickNormalVector = np.array([m_lines[ret_index[0]][2][0], m_lines[ret_index[0]][2][1], 0.0])
    PickNormalVector2 = np.array([m_lines[ret_index[1]][2][0], m_lines[ret_index[1]][2][1], 0.0])

    scale = 0.0
    scale2 = 0.0
    for dim in range(len(pt)-1):
        scale += OnePointVector[dim] * PickNormalVector[dim]
        scale2 += OnePointVector2[dim] * PickNormalVector2[dim]

    DistanceVector = PickNormalVector * scale
    DistanceVector2 = PickNormalVector2 * scale2

    for dim in range(len(pt)-1):
        d[dim] = DistanceVector2[dim] if np.linalg.norm(DistanceVector) > np.linalg.norm(DistanceVector2) else DistanceVector[dim]

    PickClosestNumber = 1 if np.linalg.norm(DistanceVector) > np.linalg.norm(DistanceVector2) else 0

    Distance2EndPoint = [np.linalg.norm(OnePointVector), np.linalg.norm(OnePointVectorOtherEnd), np.linalg.norm(OnePointVector2), np.linalg.norm(OnePointVectorOtherEnd2)]

    if not isPointOnLineSegment(m_lines[ret_index[PickClosestNumber]][0], m_lines[ret_index[PickClosestNumber]][1], [pt[0] + d[0], pt[1] + d[1], 0.0]):
        IndexOfSmallest = Distance2EndPoint.index(min(Distance2EndPoint))

        if IndexOfSmallest == 0:
            for dim in range(len(pt)-1):
                d[dim] = OnePointVector[dim]
        elif IndexOfSmallest == 1:
            for dim in range(len(pt)-1):
                d[dim] = OnePointVectorOtherEnd[dim]
        elif IndexOfSmallest == 2:
            for dim in range(len(pt)-1):
                d[dim] = OnePointVector2[dim]
        elif IndexOfSmallest == 3:
            for dim in range(len(pt)-1):
                d[dim] = OnePointVectorOtherEnd2[dim]
       # Ray casting algorithm to determine if the point is inside or outside the polygon
    # magnitude of the distance 

    distnace_magnitude = np.linalg.norm(d)
    d = d/distnace_magnitude
    num_intersections = 0
    for line in m_lines:
        if ((line[0][1] <= pt[1] and line[1][1] > pt[1]) or (line[0][1] > pt[1] and line[1][1] <= pt[1])) and \
                (pt[0] < (line[1][0] - line[0][0]) * (pt[1] - line[0][1]) / (line[1][1] - line[0][1]) + line[0][0]):
            num_intersections += 1

    if num_intersections % 2 == 1:
        # Odd number of intersections means the point is inside the polygon
        distnace_magnitude=-1*distnace_magnitude
    else:
        # Even number of intersections means the point is outside the polygon
        distnace_magnitude = 1*distnace_magnitude

    return distnace_magnitude,d

def compute_distances_kdtree(query_pt,m_lines,kd_tree):
    distances = []
    normals = []

    for pt in query_pt:
        distnace_magnitude,d = compute_distance_vector(pt, m_lines, kd_tree, [])  # Compute distance vector
        distances.append(distnace_magnitude)  # Extract only the distance components
        normals.append(d)    # Extract the normal vector components

    return np.array(distances), np.array(normals)


def isPointOnLineSegment(p1, p2, p):
    totalDist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    dist1 = math.sqrt((p[0] - p1[0]) ** 2 + (p[1] - p1[1]) ** 2)
    dist2 = math.sqrt((p2[0] - p[0]) ** 2 + (p2[1] - p[1]) ** 2)
    return abs(dist1 + dist2 - totalDist) < 1e-9


# Function to generate points near the axes intersection
def generate_points_near_axes(radius, num_points):
    # Define the axes intersection points
    axes_points = np.array([
        [0, 0, radius],   # Positive z-axis
        [0, 0, -radius],  # Negative z-axis
        [0, radius, 0],   # Positive y-axis
        [0, -radius, 0],  # Negative y-axis
        [radius, 0, 0],   # Positive x-axis
        [-radius, 0, 0]   # Negative x-axis
    ])

    # Generate additional points near the axes intersection
    additional_points = np.random.normal(loc=0, scale=0.001, size=(num_points, 3))
    additional_points *= radius / np.linalg.norm(additional_points, axis=1)[:, np.newaxis]

    # Repeat the axes points for each additional point
    axes_points_repeated = np.repeat(axes_points, num_points, axis=0)

    # Reshape the axes_points_repeated array to match the shape of additional_points for proper addition
    axes_points_reshaped = np.reshape(axes_points_repeated, (len(axes_points), num_points, 3))

    # Add the additional points to the axes points
    axes_nearby_points = axes_points_reshaped + additional_points

    # Reshape the axes_nearby_points array to match the desired output shape
    axes_nearby_points = np.reshape(axes_nearby_points, (-1, 3))

    return axes_nearby_points
# let's define a function which takes a triangle 
# Not used 
# Tested but didn't work

# def calculate_mean_curvature(mesh, v, f, radius):
#     mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, v, radius)
#     # Average mean curvature values for vertices that belong to each triangle
#     triangle_mean_curvature = [np.mean(mean_curvature[triangle]) for triangle in f]
#     return np.array(triangle_mean_curvature)
def generate_signed_distance(query_points, geometry_path):
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

def generate_analytical_sphere(uniform_points,narrow_points,on_surface_points,save_path):

    # take a radius equal to our previous radius
    radius = 0.5

    # generate random uniform points between -1 and 1 using r,theta,phi and convert to cartesian
    r = np.random.uniform(-1, 1, size=uniform_points)
    theta = np.random.uniform(0, 2 * np.pi, size=uniform_points)
    phi = np.random.uniform(0, np.pi, size=uniform_points)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    uniform_points = np.column_stack((x, y, z))
    # save the uniform points with the signed distance 
    S_uniform = np.linalg.norm(uniform_points, axis=1) - radius
    # normal at the particular point is the point itself
    n_uniform = uniform_points
    data_uniform = np.column_stack((uniform_points, S_uniform,n_uniform))
    df_uniform = pd.DataFrame(data_uniform, columns=['x', 'y', 'z', 'S','nx','ny','nz'])
    
    # generate random narrow points between -1 and 1 using r,theta,phi and convert to cartesian]
    r = np.random.uniform(0.846, 0.854, size=narrow_points)
    theta = np.random.uniform(0, 2 * np.pi, size=narrow_points)
    phi = np.random.uniform(0, np.pi, size=narrow_points)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    narrow_points = np.column_stack((x, y, z))
    # save the narrow points with the signed distance
    S_narrow = np.linalg.norm(narrow_points, axis=1) - radius
    # normal at the particular point is the point itself
    n_narrow = narrow_points
    data_narrow = np.column_stack((narrow_points, S_narrow,n_narrow))
    df_narrow = pd.DataFrame(data_narrow, columns=['x', 'y', 'z', 'S','nx','ny','nz'])

    # generate random on surface points between -1 and 1 using r,theta,phi and convert to cartesian
    # add 10% more onsurface points
    # corresponding to the intersection of the sphere with the axes
    # θ = 0, φ = 0: Corresponds to the positive z-axis.
    # θ = π, φ = 0: Corresponds to the negative z-axis.
    # θ = π/2, φ = 0: Corresponds to the positive y-axis.
    # θ = 3π/2, φ = 0: Corresponds to the negative y-axis.
    # θ = π/2, φ = π/2: Corresponds to the positive x-axis.
    # θ = 3π/2, φ = π/2: Corresponds to the negative x-axis.


    additional_points = int(0.1*on_surface_points)

    r = radius * np.ones(on_surface_points)
    theta = np.random.uniform(0, 2 * np.pi, size=on_surface_points)
    # generate theta corresponding to the intersection of the sphere with the axes
    theta_axis = np.random.uniform(0, 2 * np.pi, size=on_surface_points)

    phi = np.random.uniform(0, np.pi, size=on_surface_points)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    on_surface_points = np.column_stack((x, y, z))
    # Generate additional points on the axes
    axes_nearby_points = generate_points_near_axes(radius, additional_points)
    on_surface_points = np.vstack((on_surface_points, axes_nearby_points))
    # save the on surface points with the signed distance
    S_on_surface = np.linalg.norm(on_surface_points, axis=1) - radius
    # normal at the particular point is the point itself
    n_on_surface = on_surface_points
    data_on_surface = np.column_stack((on_surface_points, S_on_surface,n_on_surface))
    df_on_surface = pd.DataFrame(data_on_surface, columns=['x', 'y', 'z', 'S','nx','ny','nz'])

    dataframes = [("uniform", df_uniform), ("surface", df_on_surface), ("narrow", df_narrow)]

    for name, df in dataframes:
        path = os.path.join(save_path, f"{name}.csv")
        df.to_csv(path,index=True)

    return df_uniform,df_narrow,df_on_surface

def generate_points_circle(uniform_points,on_surface_points,narrow_points,width,save_path):
    """
    Generate points on a circle with a given radius.

    Args:
        radius (float): The radius of the circle.
        num_points (int): The number of points to generate.

    Returns:
        numpy.ndarray: An array of points on the circle.
    """
    # circle radius is set here 
    radius = np.sqrt(2/np.pi)
    # generate random uniform points between -1 and 1 using r,theta,phi and convert to cartesian
    x = np.random.uniform(-1, 1, size=uniform_points)
    y = np.random.uniform(-1, 1, size=uniform_points)
    z = np.zeros(x.shape)

    uniform_points = np.column_stack((x, y,z))
    # save the uniform points with the signed distance 
    S_uniform = np.linalg.norm(uniform_points, axis=1) - radius
    # normal at the particular point is the point itself
    norms=np.linalg.norm(uniform_points, axis=1)
    n_uniform = uniform_points/ norms[:, np.newaxis]
    data_uniform = np.column_stack((uniform_points, S_uniform,n_uniform))
    df_uniform = pd.DataFrame(data_uniform, columns=['x', 'y','z','S','nx','ny','nz'])
    
    # generate random narrow points between -1 and 1 using r,theta,phi and convert to cartesian]
    
    r = np.random.uniform(radius+width, radius-width, size=narrow_points)
    theta = np.random.uniform(0, 2 * np.pi, size=narrow_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(x.shape)
    narrow_points = np.column_stack((x, y,z))
    # save the narrow points with the signed distance
    S_narrow = np.linalg.norm(narrow_points, axis=1) - radius
    
    # normal at the particular point is the point itself
    norms = np.linalg.norm(narrow_points, axis=1)
    n_narrow = narrow_points/ norms[:, np.newaxis]

    data_narrow = np.column_stack((narrow_points, S_narrow,n_narrow))
    df_narrow = pd.DataFrame(data_narrow, columns=['x', 'y','z', 'S','nx','ny','nz'])

    ###################GENERATE POINTS ON SURFACE

    r = radius * np.ones(on_surface_points)
    theta = np.random.uniform(0, 2 * np.pi, size=on_surface_points)

    x = r *  np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(x.shape)
    on_surface_points = np.column_stack((x, y,z))
    # save the on surface points with the signed distance
    S_on_surface = np.linalg.norm(on_surface_points, axis=1) - radius
    # normal at the particular point is the point itself
    norm = np.linalg.norm(on_surface_points, axis=1)
    n_on_surface = on_surface_points/ norm[:, np.newaxis]
    data_on_surface = np.column_stack((on_surface_points, S_on_surface,n_on_surface))
    df_on_surface = pd.DataFrame(data_on_surface, columns=['x', 'y','z', 'S','nx','ny','nz'])

    dataframes = [("uniform", df_uniform), ("surface", df_on_surface), ("narrow", df_narrow)]

    for name, df in dataframes:
        path = os.path.join(save_path, f"{name}.csv")
        df.to_csv(path,index=True)

    return df_uniform,df_narrow,df_on_surface
##############################################################################################################
##################################### MAIN FUNCTIONS ##########################################################

def generate_signed_distance_2D_msh(uniform_points, narrow_points, on_surface_points, width, geometry_path, save_path):
    """
    Generate signed distance data for a given 2D mesh geometry.

    Args:
    - uniform_points (int): The number of uniformly distributed points to generate.
    - narrow_points (int): The number of points to generate within the narrow band.
    - on_surface_points (int): The number of points to generate on the surface.
    - width (float): The width of the dense region around the surface.
    - geometry_path (str): The path to the 2D mesh geometry file.
    - save_path (str): The path to save the generated signed distance data.

    Returns:
    - bool: True if the signed distance data is successfully written to the CSV files, otherwise function fails
    """
    # Extract the polygon from the Gmsh file
    polygon_points = extract_polygon_from_gmsh(geometry_path)

    # Create a KDTree for the polygon points
    kd_tree = KDTree(polygon_points)
    # Generate the signed distance data for the uniform points
    m_lines = [np.array([np.array(polygon_points[i]), np.array(polygon_points[i + 1]), np.zeros(2)]) for i in range(len(polygon_points) - 1)]
    m_lines.append(np.array([np.array(polygon_points[0]), np.array(polygon_points[-1]), np.zeros(2)]))
    number_of_lines = len(m_lines)

        ###################GENERATE POINTS ON SURFACE
    points_per_line = on_surface_points // number_of_lines
    points = np.array([])  # Initialize an empty array to store points

    for line in m_lines:
        dt = line[1] - line[0]
        samples = np.random.uniform(0, 1, points_per_line)  # Generate random samples for each line
        point = line[0] + samples[:, np.newaxis] * dt
      # Compute the points along the line
        if points.size == 0:
            points = point
        else:
            points = np.vstack((points, point))
    print(f"Shapes of points is {points.shape}")
    x = points[:, 0]
    y = points[:, 1]
    z = np.zeros(x.shape)
    on_surface_points = np.column_stack((x, y,z))
    S_on_surface = np.zeros(len(points))
    norm = np.linalg.norm(on_surface_points, axis=1)
    n_on_surface = on_surface_points/ norm[:, np.newaxis]
    data_on_surface = np.column_stack((on_surface_points, S_on_surface,n_on_surface))
    df_on_surface = pd.DataFrame(data_on_surface, columns=['x', 'y','z', 'S','nx','ny','nz'])
        # circle radius is set here 
    path = os.path.join(save_path, "surface.csv")
    df_on_surface.to_csv(path,index=True)
    exit(1)
    # Uniform points
    radius = np.sqrt(2/np.pi)
    # generate random uniform points between -1 and 1 using r,theta,phi and convert to cartesian
    x = np.random.uniform(-1, 1, size=uniform_points)
    y = np.random.uniform(-1, 1, size=uniform_points)
    z = np.zeros(x.shape)
    uniform_points = np.column_stack((x, y,z))
    S_uniform = np.linalg.norm(uniform_points, axis=1) - radius
    # normal at the particular point is the point itself
    norms=np.linalg.norm(uniform_points, axis=1)
    n_uniform = uniform_points/ norms[:, np.newaxis]
    data_uniform = np.column_stack((uniform_points, S_uniform,n_uniform))
    df_uniform = pd.DataFrame(data_uniform, columns=['x', 'y','z','S','nx','ny','nz'])
    
    # generate random narrow points between -1 and 1 using r,theta,phi and convert to cartesian]
    
    points_per_line = narrow_points // number_of_lines
    points = np.array([])  # Initialize an empty array to store points

    for line in m_lines:
        dt = line[1] - line[0]
        samples = np.random.uniform(0, 1, points_per_line)  # Generate random samples for each line
        point = line[0] + samples[:, np.newaxis] * dt
        narrow_widths = np.random.uniform(-width,width,)
      # Compute the points along the line
        if points.size == 0:
            points = point
        else:
            points = np.vstack((points, point))
    print(f"Shapes of points is {points.shape}")
    x = points[:, 0]
    y = points[:, 1]
    z = np.zeros(x.shape)
    narrow_points = np.column_stack((x, y,z))
    # save the narrow points with the signed distance
    S_narrow,n_narrow = compute_distances_kdtree(narrow_points,m_lines,kd_tree)

    data_narrow = np.column_stack((narrow_points, S_narrow,n_narrow))
    df_narrow = pd.DataFrame(data_narrow, columns=['x', 'y','z', 'S','nx','ny','nz'])



    dataframes = [("uniform", df_uniform), ("surface", df_on_surface), ("narrow", df_narrow)]

    for name, df in dataframes:
        path = os.path.join(save_path, f"{name}.csv")
        df.to_csv(path,index=True)

    return df_uniform,df_narrow,df_on_surface
    
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