import pandas as pd
import torch
import trimesh
import os
from skimage.measure import marching_cubes
import numpy as np
import igl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch.nn as nn
import time







# helper Functions
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
# main function
def post_process(executor):
    # if not isinstance(executor, Executor):
    #     raise ValueError("executor must be an instance of Executor")
    
    start_time = time.time()
    try:
        volume_size = (executor.config.cubesize, executor.config.cubesize, executor.config.cubesize) 
        spacing = (2/volume_size[0], 2/volume_size[0], 2/volume_size[0])
        device = executor.device
        torch.cuda.empty_cache()
        x = torch.linspace(-1, 1, volume_size[0], device=device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        # Reshape the coordinates to create a DataFrame
        coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        # Load the model
        model = executor.model
        optimizer = torch.optim.Adam(model.parameters(), lr=executor.config.learning_rate)
        model,epoch = executor.load_model(model,optimizer,executor.train_path,True)
        if executor.config.rescale:
            geom_path = executor.rescaled_path
        else:
            geom_path = executor.config.geometry_path
        # Load the geometry
        mesh = trimesh.load(geom_path)
        # Get mesh data
        v, f = mesh.vertices, mesh.faces
        sdf_values = []
        print(f"Generating SDF values for cube size {volume_size[0]}")
        S=np.array([])
        executor.config.threshold1 = 0.01
        executor.config.threshold2 = 0.00025
        predicted_labels1 = []
        actual_labels1 =[]
        predicted_labels2 = []
        actual_labels2 = []
        mismatching_coordinates1 = []
        mismatching_coordinates2 =[]
        loss_threshold_1 = torch.zeros(1, device=executor.device)
        loss_threshold_2 = torch.zeros(1, device=executor.device)
        indices1_len = 0
        indices2_len = 0
        mse_loss = nn.MSELoss()
        batch_size = executor.config.ppbatchsize # Adjust based on available memory
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
            save_directory = executor.postprocess_save_path
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
                'Resolution': [volume_size[0]],
                'NMSELoss_Mismatch 0.01': [n_mse_mismatch_loss1],
                'NMSELoss_Mismatch 0.00025': [n_mse_mismatch_loss2],
                'Accuracy' :[accuracy1],
                'Accuracy2' :[accuracy2]
        })
        
        # Save the DataFrame to the CSV file in "append" mode
        iteration_df.to_csv( os.path.join(os.path.dirname(save_directory),"results.csv"), mode='a', header=True, index=False)     
        return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1000

    args = parse_args()
    print(args.num_hidden_layers)
    reconstruct_mesh(args.directory,args.model,args.cube,args.num_hidden_layers,args.hidden_dim,args.geometry)