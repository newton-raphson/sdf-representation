import optuna
from train import train_model,train_model_implicit
import trimesh
import numpy as np
from utils import create_directory
import os
import sys
import argparse
from configgen.config_generator import Configuration
from datagenerator import data_generator
import torch
from dataloader import load_data
import matplotlib.pyplot as plt
import pickle

class Executor:
    def __init__(self, config):
        self.config = config
        self.geometry_name = os.path.basename(config.geometry)
        #  this creates a main-directory for the geometry
        # this folder will contain everything related to that geometry
        self.main_path = create_directory(os.path.join(config.directory,f"r_{self.geometry_name[:-4]}"))
        # this folder will contain all the models for particular number of data-points
        self.data_path = create_directory(os.path.join(self.main_path,f"config_{config.uniform_points},surface_{config.surface},narrowband_{config.narrowband},narrowband_width_{config.narrowband_width}"))
        with open(os.path.join(data_path,"info.txt"),"w") as f:
            f.write(f"config_uniform{config.uniform_points},surface_{config.surface},narrowband_{config.narrowband},narrowband_width_{config.narrowband_width}")
        self.model_path = create_directory(os.path.join(data_path,f"model_{config.model.__name__},hidden_dim_{config.hidden_dim},num_hidden_layers_{config.num_hidden_layers},skip_connection_{config.skip_connection},beta_{config.beta},geometric_init_{config.geometric_init}"))
        #  this creates a folder inside the model path on the basis of
        #  the loss parameters supplied
        self.loss_path = create_directory(os.path.join(model_path,f"loss_{config.loss.__name__}"))
        #  this creates a folder inside the loss path on the basis of
        #  the training parameters supplied
        self.train_path = create_directory(os.path.join(loss_path,f"lr_{config.lr},epochs_{config.epochs},min_epochs_{config.minepochs},batch_size_{config.batchsize}"))

        # inside the train_path create a folder to save the models for checkpointing
        self.model_save_path = create_directory(os.path.join(self.train_path,"models"))

        # inside the train_path create a folder to save the  post processing results
        self.postprocess_save_path = create_directory(os.path.join(self.train_path,"postprocess"))

        # inside the train_path create a folder to save the  plots
        self.plot_save_path = create_directory(os.path.join(self.train_path,"plots"))
    def rescale(self):
        rescaled_path=os.path.join(self.main_path,self.geometry_name[:-4]+"_rescaled.stl")
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
        else :
            print("Rescaled file already exists")
        return rescaled_path
    def sampling(self):
        if os.path.exists(os.path.join(self.data_path,"uniform_points.csv")):
            print("Sampling already done")
        elif self.config.distributed:
            data_generator.write_signed_distance_distributed(self.config.geometry,self.data_path,self.config.uniform_points,self.config.surface,self.config.narrowband,self.config.narrowband_width)
            print("Distributed sampling done")
        else:
            geometry_path = self.config.geometry if not self.config.rescale else self.rescale()
            df_uniform_points,df_on_surface,df_narrow_band, = data_generator.write_signed_distance(geometry_path,self.data_path,self.config.uniform_points,self.config.surface,self.config.narrowband,self.config.narrowband_width)
            # save the dataframes to CSV
            df_uniform_points.to_csv(os.path.join(self.data_path,"uniform_points.csv"))
            df_on_surface.to_csv(os.path.join(self.data_path,"on_surface.csv"))
            df_narrow_band.to_csv(os.path.join(self.data_path,"narrow_band.csv"))
            print("Sampling done")
    def train(self):
        # making sure that the sample exists 
        self.sampling()
        # already in the class variables
        training_dataloader, validation_dataloader = load_data.load_data(self.data_path,self.config,self.device)
        # optimizer is used as provided in the config
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if device == 'cuda':
            torch.cuda.empty_cache()
        if self.config.contd:
            # load the best model from the model_save_path
            self.model, optimizer, epoch, loss_per_epoch, best_val_loss,val_loss_per_epoch =\
                self.load_model(self.model,optimizer,self.model_save_path,best=True)  
        else:
            start_epoch = 0
            loss_per_epoch = []
            val_loss_per_epoch = []
            best_val_loss = float('inf')
        # counter for early stopping
        # train the model
        self.model.train()
        counter = 0
        for i in range(start_epoch, int(num_epochs)):
            train_loss = 0
            torch.cuda.empty_cache()
            for batch, (x_batch, y_batch) in enumerate(training_dataloader):
                optimizer.zero_grad()
                loss = self.loss(x_batch, y_batch, self.model,i)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(training_dataloader)
            loss_per_epoch.append(train_loss)
            val_loss = 0
            for batch, (x_batch, y_batch) in enumerate(validation_dataloader):
                loss = self.loss(x_batch, y_batch, self.model,i)
                val_loss += loss.item()
            val_loss /= len(validation_dataloader)
            val_loss_per_epoch.append(val_loss)
            # write this to a file 
            with open(os.path.join(self.train_path,"train_loss.txt"),"w") as f:
                f.write(f"Epoch {i+1}/{num_epochs}: train loss {train_loss} validation loss {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # save the model
                # code to save the model 
                self.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=True)
                pass
            else:
                # increase the counter by 1
                counter += 1
            if counter >= self.config.patience:
                print("Early stopping: No improvement for the last {} epochs".format(self.config.patience))
                self.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=False)
                break
            if i%self.config.checkpointing == 0:
                # save the model every 
                self.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=False)
                fig, ax = plt.subplots()
                ax.plot(loss_per_epoch, label='train_loss')
                ax.plot(val_loss_per_epoch, label='val_loss')
                ax.set_title('Loss vs Epochs')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                fig.savefig(os.path.join(self.plot_save_path, f"loss{i}.png"))
                plt.close(fig)

        # save the model every 
    def save_model(model, optimizer, loss_per_epoch, epoch,best_val_loss,val_loss_per_epoch,save_path,best=False):
        if best:
            checkpoint_data = {
                'epoch': epoch,
                'loss_per_epoch': loss_per_epoch,
                'best_val_loss': best_val_loss,
                'val_loss_per_epoch': val_loss_per_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_path = os.path.join(save_path, f"best_model.pkl")
        else:
            # just save the model state dict and epoch 
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            checkpoint_path = os.path.join(save_path, f"model_epoch{epoch}.pkl")
        with open(checkpoint_path, 'wb') as checkpoint_file:
            pickle.dump(checkpoint_data, checkpoint_file)   
    def load_model(model,optimizer,save_path,best = False):
        if best:
            checkpoint_path = os.path.join(save_path, "best_model.pkl")
            with open(latest_model_file, 'rb') as checkpoint_file:
                checkpoint_data = pickle.load(checkpoint_file)
            model = model_device_handler(model,checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            epoch = checkpoint_data['epoch']+1
            best_val_loss = checkpoint_data['best_val_loss']
            val_loss_per_epoch = checkpoint_data['val_loss_per_epoch']
            loss_per_epoch = checkpoint_data['loss_per_epoch']
            return model, optimizer, epoch, loss_per_epoch, best_val_loss, val_loss_per_epoch
        else:
            checkpoint_path = save_path
            with open(checkpoint_path, 'rb') as checkpoint_file:
                checkpoint_data = pickle.load(checkpoint_file)
            model = model_device_handler(model,checkpoint_data['model_state_dict'])
            epoch = checkpoint_data['epoch']+1
            return model, epoch
    def model_device_handler(model,model_state_dict):
        """ Handle model for different device configurations

        Args:
            model: model to be handled

            model_state_dict: model state dict to be loaded
        """
        # if the model is single gpu model and state dict is multi-gpu
        # then remove the module from the keys
        # this will work if the model is not multi-gpu model
        # as well 
        if not isinstance(model, torch.nn.DataParallel):
            new_state_dict = OrderedDict()
            # Create a new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model
        
        # if the model is multi-gpu model and state dict is single-gpu
        # then add the module from the keys
        elif isinstance(model, torch.nn.DataParallel):
            new_state_dict = OrderedDict()
            # Create a new OrderedDict that does not contain `module.`
            for k, v in model_state_dict.items():
                name = 'module.' + k if not k.startswith('module.') else k  # add `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model
    def reconstruct_only(self):
        volume_size = (self.cubesize, self.cubesize, self.cubesize)  # Adjust this based on your requirements
        spacing = (2/volume_size[0], 2/volume_size[1], 2/volume_size[2])
        x = torch.linspace(-1, 1, volume_size[0], device=device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        # Reshape the coordinates to create a DataFrame
        coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).to(self.device)
        batch_size = self.conf.ppbatchsize  # Adjust based on available memory
        sdf_values = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model, epoch = self.load_model(self.model,optimizer,self.model_save_path,best=True)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, coordinates.shape[0], batch_size):
                batch_coordinates = coordinates[i:i + batch_size]
                batch_sdf = self.model(batch_coordinates).to(torch.float32)
                # type(batch_sdf)
                sdf_values.append(batch_sdf)
                batch_sdf = batch_sdf.ravel()

        sdf_values = torch.cat(sdf_values)
        # Reshape the SDF values array to match the volume shape
        sdf_values = sdf_values.cpu().numpy().reshape(volume_size)
        verts, faces, normals, _ = marching_cubes(sdf_values, level=0.0,spacing=spacing)
        print(f"Mesh generated for cube size {cube}")
        if self.config.rescale:
            # save the mesh
            centroid = np.mean(verts)
            verts -= centroid
        # save the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        # mesh.export(os.path.join(save_directory, f"output_mesh{i}.stl"), file_type='stl')
        print(f"Saving mesh to {os.path.join(self.postprocess_save_path, f'{os.path.basename(geometry_path)}_resconstructed_{epoch}.stl')}")
        mesh.export(os.path.join(self.postprocess_save_path, f"{os.path.basename(geometry_path)}_resconstructed_{epoch}_cube_{cube}.stl"), file_type='stl') 

    def postprocess(self):
        # already in the class variables
        volume_size = (self.cubesize, self.cubesize, self.cubesize)
        spacing = (2/volume_size[0], 2/volume_size[1], 2/volume_size[2])
        x = torch.linspace(-1, 1, volume_size[0], device=device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        # Reshape the coordinates to create a DataFrame
        coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).to(self.device)
        batch_size = self.conf.ppbatchsize  # Adjust based on available memory
        sdf_values = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model, epoch = self.load_model(self.model,optimizer,self.model_save_path,best=True)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, coordinates.shape[0], batch_size):
                batch_coordinates = coordinates[i:i + batch_size]
                batch_sdf = self.model(batch_coordinates).to(torch.float32)
                # type(batch_sdf)
                sdf_values.append(batch_sdf)
                batch_sdf = batch_sdf.ravel()
        
    def run(self):
        if self.config.samplingonly:
            print("Sampling only")
            self.sampling()
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = config.model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(model)
        self.model.to(device)
        self.loss = self.config.loss
        if self.config.ppo:
            print("PPO only")
            self.postprocess()
            return
        self.train()
        return
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

        training_dataloader, validation_dataset = load_data.load_data(data_path,config,device)

        # optimizer is used as provided in the config 

        # train the model
    
if __name__ == '__main__':
    # pass the config file path to the function
    config_file_path = sys.argv[1]
    config = Configuration(config_file_path)
    testing_different_stl(config)