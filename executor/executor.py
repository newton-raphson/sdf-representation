import optuna
from train import train_model,train_model_implicit
import trimesh
import numpy as np
from utils import create_directory
import os
import sys
import argparse
from config_generator import Configuration
from datagenerator import data_generator
import torch
def testing_different_stl(config):
    geometry_name = os.path.basename(config.geometry)
    #  this creates a main-directory for the geometry
    # this folder will contain everything related to that geometry
    main_path = create_directory(os.path.join(config.directory,f"r_{geometry_name[:-4]}"))
    # this folder will contain all the models for particular number of data-points
    data_path = create_directory(os.path.join(main_path,f"config_{config.uniform_points},surface_{config.surface},narrowband_{config.narrowband},narrowband_width_{config.narrowband_width}"))

    # create a .txt to save the information about the data_path
    with open(os.path.join(data_path,"info.txt"),"w") as f:
        f.write(f"config_uniform{config.uniform_points},surface_{config.surface},narrowband_{config.narrowband},narrowband_width_{config.narrowband_width}")

    if config.samplingonly:
        print("Sampling only")
        if config.distributed:
            data_generator.write_signed_distance_distributed(config.geometry,data_path,config.uniform_points,config.surface,config.narrowband,config.narrowband_width)
            print("Distributed sampling done")
            return
    if config.rescale:
        rescaled_path=os.path.join(main_path,geometry[:-4]+"_rescaled.stl")
        # if the rescaled file exists then use that
        if not os.path.exists(rescaled_path):
            # Load the mesh
            geometry = geometry_path
            mesh = trimesh.load(geometry)
                
            # Rescale and center the mesh
            desired_volume = 0.5 * (1 - (-1)) ** 3
            # while 
            scaling_factor = (desired_volume / mesh.volume) ** (1/3)
            # Center the mesh vertices around the origin
            mesh.vertices -= np.mean(mesh.vertices, axis=0)
            # Rescale the mesh
            # scaling_factor = min(1.0, scaling_factor / max_abs_coord)
            mesh.vertices *= scaling_factor
            while (np.max(np.abs(mesh.vertices))+0.1+0.05)>1:
                mesh.vertices*=0.99999
    
            mesh.export(rescaled_path,file_type='stl')
            print("Rescaling done")
    
    if os.path.exists(os.path.join(data_path,"uniform_points.csv")):
        print("Sampling already done")
    else:
        geometry_path = config.geometry if not config.rescale else rescaled_path
        df_uniform_points,df_on_surface,df_narrow_band, = data_generator.write_signed_distance(geometry_path,data_path,config.uniform_points,config.surface,config.narrowband,config.narrowband_width)
        # save the dataframes to CSV
        df_uniform_points.to_csv(os.path.join(data_path,"uniform_points.csv"))
        df_on_surface.to_csv(os.path.join(data_path,"on_surface.csv"))
        df_narrow_band.to_csv(os.path.join(data_path,"narrow_band.csv"))
        print("Sampling done")
        if config.samplingonly:
            return
    # get the device to be used for training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    if not config.ppo:
        #  this sets the post process to false 
        #  we create a folder inside the data path on the basis of 
        #  the model parameters supplied 

        #  this creates a folder inside the data path on the basis of
        #  the model parameters supplied
        model_path = create_directory(os.path.join(data_path,f"model_{config.model.__name__},hidden_dim_{config.hidden_dim},num_hidden_layers_{config.num_hidden_layers},skip_connection_{config.skip_connection},beta_{config.beta},geometric_init_{config.geometric_init}"))
        #  this creates a folder inside the model path on the basis of
        #  the loss parameters supplied
        loss_path = create_directory(os.path.join(model_path,f"loss_{config.loss.__name__}"))
        #  this creates a folder inside the loss path on the basis of
        #  the training parameters supplied
        train_path = create_directory(os.path.join(loss_path,f"lr_{config.lr},epochs_{config.epochs},min_epochs_{config.minepochs},batch_size_{config.batchsize}"))
        #  this creates a folder inside the train path on the basis of
        #  the sampling parameters supplied

        # let's get the dataloader from the data_path
        




        


        
    
if __name__ == '__main__':
    # pass the config file path to the function
    config_file_path = sys.argv[1]
    config = Configuration(config_file_path)
    testing_different_stl(config)