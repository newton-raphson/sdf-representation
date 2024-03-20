from executor.executor import Executor
from model.networks import ImplicitNet
from model.losses import compute_normal
import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datagenerator.data_generator import generate_signed_distance
import glob
import sys
from utils.visualize_errors import plot_errors
import time

def compute_normal_for_model(model_path, hidden_dim, num_hidden_layers, save_path):
    model = ImplicitNet(d_in=3, dims=[hidden_dim for i in range(num_hidden_layers)], skip_in=[num_hidden_layers//2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model,epoch = Executor.load_model(model, optimizer, model_path)
    print(f"Model loaded from epoch {epoch}")
    model.eval()
    # sys.exit(0)

    # use pandas to read the csv
    df = pd.read_csv(os.path.join(save_path, 'nodes_coordinates.csv'))

    # load points from the csv file
    input_tensor = torch.FloatTensor(df[["x", "y", "z"]].values)
    input_tensor = input_tensor.view(-1, 3)

    # check if the save_path has any stl files
    files = glob.glob(os.path.join(save_path, '*.stl'))
    if(len(files) != 0):
        # load the stl file
        print(f"Loading the stl file{files[0]}")
        df = generate_signed_distance(df[["x", "y", "z"]].values,files[0])
        # save the df to a csv file
        df.to_csv(os.path.join(save_path, 'igl_wf.csv'), index=True)
    
    start = time.time()

    # compute the model output
    output = model(input_tensor)

    # compute the normal for the particular input
    normal = compute_normal(model, input_tensor)
    end = time.time()
    print(f"Time taken to compute the normal: {end-start}")
    print("Normal shape: ", normal.shape)
    # print(normal)
    # print(normal.shape)
    # normalize the normal
    # normal = output / torch.norm(output, dim=1).view(-1, 1)
    normal=normal.detach().numpy()

    output = output.detach().numpy()

    print(output.shape, normal.shape, input_tensor.shape)
    # # # create a df like the input df
    # # df_op = pd.DataFrame(np.hstack((df["x"].values,df["y"].values,df["z"].values, output.detach().numpy(), normal[0,:],normal[1,:],normal[2,:])), columns=df.columns.tolist())

    # # save the df to a csv file
    data = np.column_stack((df["x"].values,df["y"].values,df["z"].values, output,normal))
    # print(output.shape, normal.shape, input.shape)
    df_op = pd.DataFrame(data, columns=['x', 'y', 'z', 'S','nx','ny','nz'])

    df_op.to_csv(os.path.join(save_path, 'computed.csv'), index=True)
    exit()

    # compute the error in computed S and the original S
    error = np.abs(df["S"].values - df_op["S"].values)
    # save the co-ordinate and the error to a csv file
    data = np.column_stack((df["x"].values,df["y"].values,df["z"].values, error))
    df_error = pd.DataFrame(data, columns=['x', 'y', 'z', 'error'])
    df_error.to_csv(os.path.join(save_path, 'error_points.csv'), index=True)
    # mean square error
    mse = np.mean(error**2)
    # compute the root mean square error
    rmse = np.sqrt(np.mean(error**2))
    print("Root mean square error: ", rmse)

    # Compute cosine similarity
    v1 = df[["nx", "ny", "nz"]].values
    v2 = df_op[["nx", "ny", "nz"]].values
    # v1_2d = v1.reshape(1, -1)
    # v2_2d = v2.reshape(1, -1)
    # print(v1_2d.shape, v2_2d.shape)
    # Compute cosine similarity for each pair of corresponding vectors
    similarity = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))


    # save the similarity with the co-ordinates to a csv file
    data = np.column_stack((df["x"].values,df["y"].values,df["z"].values, similarity))
    df_similarity = pd.DataFrame(data, columns=['x', 'y', 'z', 'similarity'])
    df_similarity.to_csv(os.path.join(save_path, 'similarity_points.csv'), index=True)



    print("Cosine similarity: ", similarity)
    print("Similarity shape", similarity.shape)
    print("Mean similarity: ", np.mean(similarity))
    print("Median similarity: ", np.median(similarity))
    print("Standard deviation of similarity: ", np.std(similarity))
    print("Minimum similarity: ", np.min(similarity))
    print("Maximum similarity: ", np.max(similarity))

    # save all this statistics to a file to similarity.csv

    with open(os.path.join(save_path, 'similarity.csv'), 'w') as f:
        f.write("Cosine similarity: \n")
        f.write(str(similarity))
        f.write("\nMean similarity: ")
        f.write(str(np.mean(similarity)))
        f.write("\nMedian similarity: ")
        f.write(str(np.median(similarity)))
        f.write("\nStandard deviation of similarity: ")
        f.write(str(np.std(similarity)))
        f.write("\nMinimum similarity: ")
        f.write(str(np.min(similarity)))
        f.write("\nMaximum similarity: ")
        f.write(str(np.max(similarity)))

        # save the rmse and mse

        f.write("\nRoot mean square error: ")
        f.write(str(rmse))
        f.write("\nMean square error: ")
        f.write(str(mse))
    plot_errors(save_path)


def compute_all():
    # main_path = "/work/mech-ai-scratch/samundra/experiments/sdf_representation_test"
    # for file_path in ["cube","cone","cylinder","sphere","tractor","turbine","tetrakis"]:
    #     save_string =file_path+"/normal_outside"
    #     model_path = file_path+"/models"
    #     compute_normal_for_model(os.path.join(main_path,model_path),512,8,os.path.join(main_path,save_string))
    # for file_path in ["tractor"]:
    #     save_string ="/work/mech-ai-scratch/samundra/experiments/sdf_representation_test/turbine/r_kaplan_turbine/config_uniform100000,surface_40,narrowband_40,narrowband_width_0.0001/ImplicitNet,hidden_dim_512,num_hidden_layers_8,skip_connection_(4,),beta_100.0,geometric_init_True/loss_IGRLOSS/lr_0.0004,epochs_20000,min_epochs_1000,batch_size_8192/postprocess"
    #     model_path = file_path+"/models"
    #     compute_normal_for_model(os.path.join(main_path,model_path),512,10,os.path.join(main_path,save_string))
    
    postprocess_path = "/work/mech-ai-scratch/samundra/experiments/sdf_representation_test/turbine/r_kaplan_turbine/config_uniform100000,surface_40,narrowband_40,narrowband_width_0.0001/ImplicitNet,hidden_dim_512,num_hidden_layers_8,skip_connection_(4,),beta_100.0,geometric_init_True/loss_IGRLOSS/lr_0.0004,epochs_20000,min_epochs_1000,batch_size_8192/postprocess"
    model_path = "/work/mech-ai-scratch/samundra/experiments/sdf_representation_test/turbine/r_kaplan_turbine/config_uniform100000,surface_40,narrowband_40,narrowband_width_0.0001/ImplicitNet,hidden_dim_512,num_hidden_layers_8,skip_connection_(4,),beta_100.0,geometric_init_True/loss_IGRLOSS/lr_0.0004,epochs_20000,min_epochs_1000,batch_size_8192/models"


    compute_normal_for_model(model_path,512,8,postprocess_path)