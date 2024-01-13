import trimesh
import numpy as np
from utils.files import create_directory
import os
import sys
from configgen.config_reader import Configuration
from datagenerator import data_generator
import torch
from dataloader import load_data
import matplotlib.pyplot as plt
import pickle
from evaluations import post_process
from collections import OrderedDict
import igl
class Executor:
    def __init__(self, config):
        self.config = config
        
        self.geometry_name = config.name

        self.main_path = create_directory(os.path.join(config.directory,f"r_{self.geometry_name}"))

        # this folder will contain all the models for particular number of data-points
        self.data_path = create_directory(os.path.join(self.main_path,f"config_uniform{config.uniform_points},surface_{config.surface},narrowband_{config.narrowband},narrowband_width_{config.narrowband_width}"))
        with open(os.path.join(self.data_path,"info.txt"),"w") as f:
            f.write(f"config_uniform{config.uniform_points},surface_{config.surface},narrowband_{config.narrowband},narrowband_width_{config.narrowband_width}")
        self.model_path = create_directory(os.path.join(self.data_path,f"{config.model._get_name()},hidden_dim_{config.hidden_dim},num_hidden_layers_{config.num_hidden_layers},skip_connection_{config.skip_connection},beta_{config.beta},geometric_init_{config.geometric_init}"))
        #  this creates a folder inside the model path on the basis of
        #  the loss parameters supplied
        self.loss_path = create_directory(os.path.join(self.model_path,f"loss_{config.loss._get_name()}"))
        #  this creates a folder inside the loss path on the basis of
        #  the training parameters supplied
        self.train_path = create_directory(os.path.join(self.loss_path,f"lr_{config.lr},epochs_{config.epochs},min_epochs_{config.minepochs},batch_size_{config.batchsize}"))

        # inside the train_path create a folder to save the models for checkpointing
        self.model_save_path = create_directory(os.path.join(self.train_path,"models"))

        # inside the train_path create a folder to save the  post processing results
        self.postprocess_save_path = create_directory(os.path.join(self.train_path,"postprocess"))

        # inside the train_path create a folder to save the  plots
        self.plot_save_path = create_directory(os.path.join(self.train_path,"plots"))
    def rescale(self):
        self.rescaled_path=os.path.join(self.main_path,self.geometry_name+"_rescaled.stl")
        # if the rescaled file exists then use that
        if not os.path.exists(self.rescaled_path):
            # Load the mesh
            geometry = self.config.geometry
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
    
            mesh.export(self.rescaled_path,file_type='stl')
            print("Rescaling done")
        else :
            print("Rescaled file already exists")
        return self.rescaled_path
    def sampling(self):
        if os.path.exists(os.path.join(self.data_path,"uniform.csv")) or os.path.exists(os.path.join(self.data_path,"surface.csv")) or os.path.exists(os.path.join(self.data_path,"narrow")):
            print("Sampling already done")
            return
        elif self.config.distributed:
            data_generator.write_signed_distance_distributed(self.config.geometry,self.data_path,self.config.uniform_points,self.config.surface,self.config.narrowband,self.config.narrowband_width)
            print("Distributed sampling done")
        else:
            geometry_path = self.config.geometry if not self.config.rescale else self.rescale()
            df_uniform_points,df_on_surface,df_narrow_band = data_generator.generate_signed_distance_data(geometry_path,self.config.uniform_points,self.config.surface,self.config.narrowband,self.config.narrowband_width)
            # save the dataframes to CSV
            df_uniform_points.to_csv(os.path.join(self.data_path,"uniform.csv"))
            df_on_surface.to_csv(os.path.join(self.data_path,"surface.csv"))
            df_narrow_band.to_csv(os.path.join(self.data_path,"narrow.csv"))
            print("Sampling done")
    def train(self):
        # making sure that the sample exists 
        self.sampling()
        # already in the class variables
        training_dataloader, validation_dataloader = load_data.load_data(self.data_path,self.config,self.device)
        # optimizer is used as provided in the config
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        if self.config.contd:
            # load the best model from the model_save_path
            self.model, optimizer, start_epoch, loss_per_epoch, best_val_loss,val_loss_per_epoch =\
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
        for i in range(start_epoch, int(self.config.epochs)):
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
            with open(os.path.join(self.train_path,"train_loss.txt"),"a") as f:
                f.write(f"Epoch {i+1}/{self.config.epochs}: train loss {train_loss} validation loss {val_loss}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # save the model
                # code to save the model 
                self.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=True)
            else:
                # increase the counter by 1
                counter += 1
            if counter >= self.config.patience and i >= self.config.minepochs:
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
    def save_model(self,model, optimizer, loss_per_epoch, epoch,best_val_loss,val_loss_per_epoch,save_path,best=False):
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
    def load_model(self,model,optimizer,save_path,best = False):
        if best:
            checkpoint_path = os.path.join(save_path, "best_model.pkl")
            with open(checkpoint_path, 'rb') as checkpoint_file:
                checkpoint_data = pickle.load(checkpoint_file)
            model = self.model_device_handler(model,checkpoint_data['model_state_dict'])
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
            model = self.model_device_handler(model,checkpoint_data['model_state_dict'])
            epoch = checkpoint_data['epoch']+1
            return model, epoch
    def model_device_handler(self,model,model_state_dict):
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
        volume_size = (self.config.cubesize, self.config.cubesize, self.config.cubesize)  # Adjust this based on your requirements
        spacing = (2/volume_size[0], 2/volume_size[1], 2/volume_size[2])
        x = torch.linspace(-1, 1, volume_size[0], device=self.device)
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
        verts, faces, normals, _ = igl.marching_cubes(sdf_values, level=0.0,spacing=spacing)
        print(f"Mesh generated for cube size {self.config.cubesize}")
        if self.config.rescale:
            # save the mesh
            centroid = np.mean(verts)
            verts -= centroid
        # save the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        # mesh.export(os.path.join(save_directory, f"output_mesh{i}.stl"), file_type='stl')
        print(f"Saving mesh to {os.path.join(self.postprocess_save_path, f'{ self.geometry_name}_resconstructed_{epoch}.stl')}")
        mesh.export(os.path.join(self.postprocess_save_path, f"{self.geometry_name}_resconstructed_{epoch}_cube_{self.config.cubesize}.stl"), file_type='stl') 
    def run(self):
        if self.config.samplingonly:
            print("Sampling only")
            self.sampling()
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.config.model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.loss = self.config.loss
        if self.config.ppo:
            print("PPO only")
            post_process.post_process(self)
            return
        self.train()
        return

