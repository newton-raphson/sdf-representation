import pandas as pd
import torch
import trimesh
import os
from torch.cuda.amp import autocast
from skimage.measure import marching_cubes
import numpy as np
import igl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch.nn as nn
import pickle
import time
from models import ImplicitNet
import argparse

def generate_coordinates(volume_size):
    device = torch.device("cuda")
    x = torch.linspace(-1, 1, volume_size[0], device=device)
    xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
    coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    return coordinates

def generate_sdf_values(model, coordinates, batch_size):
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    sdf_values = []
    with torch.no_grad():
        for i in range(0, coordinates.shape[0], batch_size):
            batch_coordinates = coordinates[i:i + batch_size]
            batch_sdf = model(batch_coordinates)
            sdf_values.append(batch_sdf)
    sdf_values = torch.cat(sdf_values)
    return sdf_values

def extract_mesh(sdf_values, volume_size, cube_size):
    spacing = (2/volume_size[0], 2/volume_size[0], 2/volume_size[0])
    if cube_size == 256:
        verts, faces, normals, _ = marching_cubes(sdf_values, level=0.0,spacing=spacing)
        centroid = np.mean(verts)
        verts -= centroid
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    else:
        sdf_values_check = sdf_values[::256//cube_size, ::256//cube_size, ::256//cube_size]
        verts_check, faces_check, normals_check, _ = marching_cubes(sdf_values_check, level=0.0,spacing=spacing)
        centroid = np.mean(verts_check)
        verts_check -= centroid
        mesh = trimesh.Trimesh(vertices=verts_check, faces=faces_check)
    return mesh

def calculate_signed_distance(geometry_path, coordinates):
    mesh = trimesh.load(geometry_path)
    v, f = mesh.vertices, mesh.faces
    S, I, C = igl.signed_distance(coordinates.cpu().numpy(), v, f, return_normals=False)
    return S

def calculate_loss(S, sdf_values):
    mse_loss = nn.MSELoss()
    loss = mse_loss(torch.FloatTensor(S), torch.FloatTensor(sdf_values.ravel()))
    value_loss = loss/2
    return value_loss

def calculate_accuracy(S, sdf_values):
    predicted_labels = np.sign(sdf_values)
    actual_labels = np.sign(S)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    return accuracy

def generate_iteration_df(i, value_loss, accuracy):
    iteration_df = pd.DataFrame({
        'Iteration': [i],
        'NMSELoss': [value_loss],
        'Accuracy' :[accuracy]
    })
    return iteration_df

def save_iteration_df(iteration_df, main_path):
    iteration_df.to_csv( os.path.join(main_path,"results.csv"), mode='a', header=False, index=False)

def generate_classification_report(actual_labels1, predicted_labels1, save_directory,key):
    class_report = classification_report(actual_labels1, predicted_labels1, output_dict=True)
    # Convert the classification report to a DataFrame
    df = pd.DataFrame(class_report).transpose()
    # Save the DataFrame to a CSV file
    csv_filename = os.path.join(save_directory,f"classification_report{key}.csv")
    df.to_csv(csv_filename, index=True)
    print(f"Classification report saved to {csv_filename}")

def generate_confusion_matrix(actual_labels, predicted_labels, save_directory,key):
    # Create a confusion matrix
    cm = confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = (cm / row_sums) * 100
    sns.heatmap(cm_percent, annot=True, cmap='inferno', fmt='.1f')
    plt.xlabel("Prediction")
    plt.ylabel("Actual Value")
    plt.savefig(os.path.join(save_directory, f"cm{key}.png"))
    plt.close()

def generate_mismatching_coordinates(S, sdf_values, coordinates, threshold_low, threshold_high, save_directory):
    actual_labels = np.sign(S)
    predicted_labels = np.sign(sdf_values)
    inside_range_indices = np.where((S >= threshold_low) & (S <= threshold_high))[0]
    mismatching_coordinates = []
    coordinates=coordinates.cpu().numpy()
    for index in inside_range_indices:
        if actual_labels[index] != predicted_labels[index]:
            mismatching_coordinates.append(coordinates[index])
    header_row = np.array([['x', 'y', 'z']])
    mismatching_coordinates =np.array(mismatching_coordinates)
    data_with_header = np.vstack((header_row, mismatching_coordinates))
    csv_file_path = os.path.join(save_directory,'mismatching_co-ordinates.csv')
    np.savetxt(csv_file_path, data_with_header, delimiter=',', fmt='%s')
def reconstruct_mesh(save_directory,model_path,cube,num_hidden_layers,hidden_dim, geometry_path):
    
    dims = [hidden_dim for i in range(num_hidden_layers)]
    skip_in = [num_hidden_layers//2]
    print(f"hidden dim is {num_hidden_layers}:{dims}")
    model = ImplicitNet(d_in=3,dims=dims,skip_in=skip_in)
    print(f"the skip in is {skip_in} ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i=200
    path_chk=model_path
    with open(path_chk, 'rb') as resume_file:
            saved_data = pickle.load(resume_file)
    epoch = saved_data['epoch']+1
    print(f"....Loading model from epoch {epoch}...")
    # Remove the "module." prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in saved_data['model_state_dict'].items()}

    # Load the modified state_dict into your model
    model.load_state_dict(new_state_dict)
    model.eval()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    volume_size = (cube, cube, cube)  # Adjust this based on your requirements
    spacing = (2/volume_size[0], 2/volume_size[1], 2/volume_size[2])
    torch.cuda.empty_cache()
    x = torch.linspace(-1, 1, volume_size[0], device=device)
    xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
    # Reshape the coordinates to create a DataFrame
    coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).to(device)
    batch_size = 655360  # Adjust based on available memory
    sdf_values = []
    with torch.no_grad():
        for i in range(0, coordinates.shape[0], batch_size):
            batch_coordinates = coordinates[i:i + batch_size]
            batch_sdf = model(batch_coordinates).to(torch.float32)
            # type(batch_sdf)
            sdf_values.append(batch_sdf)
            batch_sdf = batch_sdf.ravel()

    sdf_values = torch.cat(sdf_values)
    # Reshape the SDF values array to match the volume shape
    sdf_values = sdf_values.cpu().numpy().reshape(volume_size)
    verts, faces, normals, _ = marching_cubes(sdf_values, level=0.0,spacing=spacing)
    print(f"Mesh generated for cube size {cube}")
    print(f"Saving mesh to {os.path.join(save_directory, f'{os.path.basename(geometry_path)}_resconstructed_{epoch}.stl')}")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(os.path.join(save_directory, f"{os.path.basename(geometry_path)}_resconstructed_{epoch}_cube_{cube}.stl"), file_type='stl') 
    # save the mesh
    centroid = np.mean(verts)
    verts -= centroid
    # save the mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    # mesh.export(os.path.join(save_directory, f"output_mesh{i}.stl"), file_type='stl')
    print(f"Saving mesh to {os.path.join(save_directory, f'{os.path.basename(geometry_path)}_resconstructed_{epoch}.stl')}")
    mesh.export(os.path.join(save_directory, f"{os.path.basename(geometry_path)}_resconstructed_{epoch}_cube_{cube}.stl"), file_type='stl') 
def post_process(model,save_directory,geometry_path,num_points_uniform,num_points_surface,num_points_narrow_band,total_points,totalthreshold=0.01,second=None):
    start_time = time.time()
    try:
        volume_size = (1024, 1024, 1024)  # Adjust this based on your requirements
        spacing = (2/volume_size[0], 2/volume_size[0], 2/volume_size[0])
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        x = torch.linspace(-1, 1, volume_size[0], device=device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        # Reshape the coordinates to create a DataFrame
        coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        path_chk=os.path.join(save_directory, "best_model_epoch.pkl")            
        with open(path_chk, 'rb') as resume_file:
            saved_data = pickle.load(resume_file)
        epoch = saved_data['epoch']+1
        print(f"....Loading model from epoch {epoch}...")
        model.load_state_dict(saved_data['model_state_dict'])
        model.eval()
        model.to(device)
        mesh = trimesh.load(geometry_path)
        # Get mesh data
        v, f = mesh.vertices, mesh.faces
        sdf_values = []
        print(f"Generating SDF values for cube size {volume_size[0]}")
        S=np.array([])
        threshold1 = 0.01
        threshold2 = 0.00025
        predicted_labels1 = []
        actual_labels1 =[]
        predicted_labels2 = []
        actual_labels2 = []
        mismatching_coordinates1 = []
        mismatching_coordinates2 =[]
        loss_threshold_1 = torch.zeros(1, device=device)
        loss_threshold_2 = torch.zeros(1, device=device)
        indices1_len = 0
        indices2_len = 0
        mse_loss = nn.MSELoss()
        batch_size = 655360  # Adjust based on available memory
        with torch.no_grad():
            for i in range(0, coordinates.shape[0], batch_size):
                batch_coordinates = coordinates[i:i + batch_size]
                batch_sdf = model(batch_coordinates)
                sdf_values.append(batch_sdf)
                batch_sdf = batch_sdf.ravel()
                # if(i%10==0):
                # print(f"Generating SDF values for {i} points")
                S_in, _, _ = igl.signed_distance(batch_coordinates.cpu().numpy(), v, f, return_normals=False)
                print(f"the length of S_in is{len(S_in)}")
                indices=np.where((S_in >= -threshold1) & (S_in <= threshold1))[0]
                if len(indices)==0:
                    continue
                print(f"Indices for {i} points is {len(indices)}")
                # compute the loss for those indices
                
                loss = mse_loss(torch.FloatTensor(S_in[indices]).to(device), batch_sdf[indices])*len(indices) # multiply by the number of points to find sum of squared error
                loss_threshold_1 = torch.add(loss_threshold_1, loss)
                indices1_len += len(indices)
                predicted_labels = torch.sign(batch_sdf[indices]).cpu().numpy()
                # find the actual labels
                actual_labels = np.sign(S_in[indices])
                # find the mismatching indices
                # mismatching_indices1 = np.where(actual_labels != predicted_labels)[0]
                
                # find the mismatching coordinates
                # mismatching_coordinates1.append(batch_coordinates[mismatching_indices1].cpu().numpy())
                print("1")
                # append the predicted and actual labels for the first threshold
                predicted_labels1.append(predicted_labels)
                print("2")
                actual_labels1.append(actual_labels)
                print("3")
                mismatching_indices = np.where(actual_labels != predicted_labels)[0]
                cx= batch_coordinates[indices].cpu().numpy()
                # Extract the coordinates corresponding to the mismatching indices
                mismatching_coordinates1.append(cx[mismatching_indices])
                # find indices where S_in < 0.00025
                indices2=np.where((S_in >= -threshold2) & (S_in <= threshold2))[0]
                if len(indices2)==0:
                    continue
                # compute the loss for those indices
                loss2 = mse_loss(torch.FloatTensor(S_in[indices2]).to(device), batch_sdf[indices2])*len(indices2) # multiply by the number of points to find sum of squared error
                print(f"The loss 2 is {loss2}")
                print(f"The length of indices2 is {len(indices2)}")
                loss_threshold_2 = torch.add(loss_threshold_2, loss2)
                print(f"The loss threshold 2 is {loss_threshold_2}")
                indices2_len += len(indices2)
                # find the predicted labels

                # find the predicted labels for the second threshold
                predictedlabels2 = torch.sign(batch_sdf[indices2]).cpu().numpy()
                print("3.1")
                # find the actual labels for the second threshold
                actuallabels2 = np.sign(S_in[indices2])
                print("3.2")
                # find the mismatching indices for the second threshold
                # mismatching_indices2 = np.where(actual_labels2 != predicted_labels2)[0]

                print("3.3")
                # find the mismatching coordinates for the second threshold
                print("4")
                # mismatching_coordinates2.append(batch_coordinates[mismatching_indices2].cpu().numpy())
                print("5")
                # append the predicted and actual labels for the second threshold
                predicted_labels2.append(predictedlabels2)
                print("6")
                actual_labels2.append(actuallabels2)
                # for index in indices2:
                #     if actual_labels2[index] != predicted_labels2[index]:
                #         mismatching_coordinates2.append(coordinates[index])
                mismatching_indices2 = np.where(actuallabels2 != predictedlabels2)[0]

                # Extract the coordinates corresponding to the mismatching indices
                cx= batch_coordinates[indices2].cpu().numpy()
                
                mismatching_coordinates2.append(cx[mismatching_indices2])

            # Calculate the MSE loss
            n_mse_mismatch_loss1 = (loss_threshold_1/indices1_len).cpu().numpy()/(2*threshold1)
            n_mse_mismatch_loss2 = (loss_threshold_2/indices2_len).cpu().numpy()/ (2*threshold2)
            print(f"n_mse_mismatch_loss1 is {n_mse_mismatch_loss1}")
            print(f"n_mse_mismatch_loss2 is {n_mse_mismatch_loss2}")
            actual_labels1 = np.concatenate(actual_labels1)
            predicted_labels1 = np.concatenate(predicted_labels1)
            actual_labels2 = np.concatenate(actual_labels2)
            predicted_labels2 = np.concatenate(predicted_labels2)
            # Calculate accuracy
            accuracy1 = accuracy_score(actual_labels1, predicted_labels1)
            accuracy2 = accuracy_score(actual_labels2, predicted_labels2)
            print(f"accuracy1 is {accuracy1}")
            print(f"accuracy2 is {accuracy2}")
            generate_classification_report(actual_labels1, predicted_labels1, save_directory,"1")
            generate_classification_report(actual_labels2, predicted_labels2, save_directory,"2")
            generate_confusion_matrix(actual_labels1, predicted_labels1, save_directory,"1")
            generate_confusion_matrix(actual_labels2, predicted_labels2, save_directory,"2")
            # print(mismatching_coordinates1)
            header_row = np.array([['x', 'y', 'z']])
            # Concatenate the header row with your data
            data_with_header = np.vstack((header_row, *mismatching_coordinates1))

            csv_file_path = os.path.join(save_directory,'mismatching_co-ordinates1.csv')
            np.savetxt(csv_file_path, data_with_header, delimiter=',', fmt='%s')

            data_with_header = np.vstack((header_row, *mismatching_coordinates2))

            csv_file_path = os.path.join(save_directory,'mismatching_co-ordinates2.csv')
            np.savetxt(csv_file_path, data_with_header, delimiter=',', fmt='%s')
            end_time = time.time()

            iteration_df = pd.DataFrame({
                'Start Time': [start_time],
                'End Time': [end_time],
                'Time Taken': [end_time - start_time],
                'Epoch': [epoch],
                'total_points': [total_points],
                'Resolution': [volume_size[0]],
                'Geometry': [os.path.basename(geometry_path)],
                'Points_Uni': [num_points_uniform],
                'Points_Surface': [num_points_surface],
                'Points_Narrow_Band': [num_points_narrow_band],
                'NMSELoss_Mismatch 0.01': [n_mse_mismatch_loss1],
                'NMSELoss_Mismatch 0.00025': [n_mse_mismatch_loss2],
                'Accuracy' :[accuracy1],
                'Accuracy2' :[accuracy2]
        })
        
        # Save the DataFrame to the CSV file in "append" mode
        iteration_df.to_csv( os.path.join(os.path.dirname(save_directory),"resultsfull.csv"), mode='a', header=False, index=False)     
        sdf_values = torch.cat(sdf_values)

        # Reshape the SDF values array to match the volume shape
        sdf_values = sdf_values.cpu().numpy().reshape(volume_size)
        # cube_sizes = [256, 128, 64, 32]
        # for cube_size in cube_sizes:
        recon_size=256
        spacing = (2/recon_size, 2/recon_size, 2/recon_size)
        sdf_values_check = sdf_values[::volume_size[0]//recon_size, ::volume_size[0]//recon_size, ::volume_size[0]//recon_size]
        # sdf_values_check=sdf_values
        print(f"Generating mesh for cube size {recon_size}")
        verts, faces, normals, _ = marching_cubes(sdf_values_check, level=0.0,spacing=spacing)
        print(f"Mesh generated for cube size {recon_size}")
        # save the mesh
        centroid = np.mean(verts)
        verts -= centroid
        # save the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        # mesh.export(os.path.join(save_directory, f"output_mesh{i}.stl"), file_type='stl')
        print(f"Saving mesh to {os.path.join(save_directory, f'{os.path.basename(geometry_path)}_resconstructed_{recon_size}.stl')}")
        mesh.export(os.path.join(save_directory, f"{os.path.basename(geometry_path)}_resconstructed_{recon_size}.stl"), file_type='stl') 
        # else:
        #     sdf_values_check = sdf_values[::256//cube_size, ::256//cube_size, ::256//cube_size]
            #     verts_check, faces_check, normals_check, _ = marching_cubes(sdf_values_check, level=0.0,spacing=spacing)
            #     # save the mesh
            #     centroid = np.mean(verts_check)
            #     verts_check -= centroid
            #     # save the mesh
            #     mesh = trimesh.Trimesh(vertices=verts_check, faces=faces_check)
            #     # mesh.export(os.path.join(save_directory, f"output_mesh{i}.stl"), file_type='stl')
            #     mesh.export(os.path.join(save_directory, f"{os.path.basename(geometry_path)}_resconstructed_{cube_size}.stl"), file_type='stl') 

        # mesh = trimesh.load(geometry_path)

        # # Get mesh data
        # v, f = mesh.vertices, mesh.faces
        # # Calculate signed distance values
        # print("Calculating signed distance values")
        # cpu_coordinates = coordinates.cpu().numpy()
        # S=np.array([])
        # mse_loss = nn.MSELoss()
        # print("Calculating true signed distance values")
        # for i in range(0, len(cpu_coordinates), batch_size):
        #     S_in, _, _ = igl.signed_distance(cpu_coordinates[i:i+batch_size], v, f, return_normals=False)
        #     S=np.append(S,S_in)
        #     # finding if the points are 
        #     indices=np.where(S.abs()<=0.01)[0]

        # # Calculate the MSE loss
        # import torch.nn as nn
        # # Calculate the MSE loss
        # mse_loss = nn.MSELoss()
        # loss = mse_loss(torch.FloatTensor(S), torch.FloatTensor(sdf_values.ravel()))
        # # find the mse-error of vertices
        # # v= torch.FloatTensor(v).to(device)
        # # pred= model(v)
        # # actual = torch.zeros_like(pred,device=device)
        # # loss_v = mse_loss(pred, actual)
        # # normalized_vertices_loss = loss_v.cpu().detach().numpy()/2
        # # Calculate predicted labels (+1 for positive, -1 for negative, 0 for zero)
        # predicted_labels = np.sign(sdf_values)

        # # Calculate actual labels (+1 for positive, -1 for negative, 0 for zero)
        # actual_labels = np.sign(S)

        # # Generate the classification report
        # class_report = classification_report(actual_labels1, predicted_labels1, output_dict=True)
        # # Convert the classification report to a DataFrame
        # df = pd.DataFrame(class_report).transpose()
        # # Save the DataFrame to a CSV file
        # csv_filename = os.path.join(save_directory,"classification_report.csv")
        # df.to_csv(csv_filename, index=True)
        # print(f"Classification report saved to {csv_filename}")
        # Create a confusion matrix
        # cm = confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)
        # # Calculate row-wise sums
        # row_sums = cm.sum(axis=1, keepdims=True)
        # # Normalize the confusion matrix to percentages
        # cm_percent = (cm / row_sums) * 100
        # # Plot the confusion matrix as a heatmap
        # sns.heatmap(cm_percent, annot=True, cmap='inferno', fmt='.1f')  # fmt='.1f' formats the percentages to one decimal place
        # plt.xlabel("Prediction")
        # plt.ylabel("Actual Value")
        # plt.savefig(os.path.join(save_directory, "cm.png"))
        # plt.close()
        # # Find indices where label mismatch occurs within the specified threshold
        # inside_range_indices = np.where((S >= -threshold) & (S <= threshold))[0]
        # # Check if the actual and predicted labels match within the specified range of S
        # mismatching_coordinates = []
        # coordinates=coordinates.cpu().numpy()
        # for index in inside_range_indices:
        #     if actual_labels[index] != predicted_labels[index]:
        #         mismatching_coordinates.append(coordinates[index])
        # inside_range_indices = np.where((S >= -threshold) & (S <= threshold))[0]

        # # Check if the actual and predicted labels match within the specified range of S
        # mismatching_indices = inside_range_indices[(actual_labels[inside_range_indices] != predicted_labels[inside_range_indices])]

        # # Extract the mismatching coordinates
        # mismatching_coordinates = coordinates[mismatching_indices]
        # # Save the mismatched coordinates to a CSV file
        # header_row = np.array([['x', 'y', 'z']])
        # mismatching_coordinates =np.array(mismatching_coordinates)
        # # Calculate the MSE mismatch loss
        # mse_mismatch_loss = mse_loss(torch.FloatTensor(S[inside_range_indices]), torch.FloatTensor(sdf_values.ravel()[inside_range_indices]))
        # n_mse_mismatch_loss=mse_mismatch_loss/2
        # First, calculate n_mse_mismatch_loss for the first threshold (0.01)
        # Calculate n_mse_mismatch_loss for the first threshold (0.01)
        # threshold1 = 0.01
        # inside_range_indices1 = np.where(S.abs() <= threshold1)[0]

        # # Check if the actual and predicted labels match within the specified range of S
        # mismatching_indices1 = inside_range_indices1[(actual_labels[inside_range_indices1] != predicted_labels[inside_range_indices1])]

        # # Extract the mismatching coordinates
        # mismatching_coordinates1 = coordinates[mismatching_indices1]

        # # Calculate the MSE mismatch loss for the first threshold
        # mse_mismatch_loss1 = mse_loss(torch.FloatTensor(S[inside_range_indices1]), torch.FloatTensor(sdf_values.ravel()[inside_range_indices1]))
        # n_mse_mismatch_loss1 = mse_mismatch_loss1 / (2*threshold1)

        # # Calculate n_mse_mismatch_loss for the second threshold (0.00025) using the same indices
        # threshold2 = 0.00025
        # inside_range_indices2 = inside_range_indices1  # Reuse the same indices
        # inside_range_indices2 = inside_range_indices2[(S[inside_range_indices2].abs() <= threshold2)]

        # # Calculate the MSE mismatch loss for the second threshold
        # mse_mismatch_loss2 = mse_loss(torch.FloatTensor(S[inside_range_indices2]), torch.FloatTensor(sdf_values.ravel()[inside_range_indices2]))
        # n_mse_mismatch_loss2 = mse_mismatch_loss2 / (2*threshold2)

        return n_mse_mismatch_loss2
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1000

    args = parse_args()
    print(args.num_hidden_layers)
    reconstruct_mesh(args.directory,args.model,args.cube,args.num_hidden_layers,args.hidden_dim,args.geometry)