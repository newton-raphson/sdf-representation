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
from skimage.measure import marching_cubes
from skimage import measure
from evaluations.generate_gif import plot_stl
import torch.nn as nn
from model.networks import FeedForwardNetwork,ImplicitNet
from model.losses import WeightedSmoothL2Loss
from utils.pickling import CPU_Unpickler
import glob
from torch.optim.lr_scheduler import StepLR
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
        self.plot_save_path = create_directory(os.path.join(self.train_path,"plots"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cuda is {torch.cuda.is_available()}")
        self.model = self.config.model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.loss = self.config.loss
        print("\nExecutor initialized\n")
    def rescale(self):
        # rescale the geometry to fit in the unit cube
        print("Rescaling the geometry")
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
        # if not self.config.continue_sampling:
        if "pcd" in self.config.name:
            return
        if not self.config.two_dim:
            geometry_path = self.config.geometry if not self.config.rescale else self.rescale()
        else:
            geometry_path = self.config.geometry 
        if os.path.exists(os.path.join(self.data_path,"uniform.csv")) or os.path.exists(os.path.join(self.data_path,"surface.csv")) or os.path.exists(os.path.join(self.data_path,"narrow")):
            print("Sampling already done")
            return
        if self.config.two_dim:
           print("Circle Generation Is Running")
        #    df_uniform,df_narrow,df_on_surface = data_generator.generate_signed_distance_2D_msh(self.config.uniform_points,self.config.narrowband,self.config.surface,self.config.narrowband_width,self.config.geometry,self.data_path)
           data_generator.generate_points_circle(self.config.uniform_points,self.config.surface,self.config.narrowband,self.config.narrowband_width,self.data_path)
           print("Sampling done")
           return
        else:
            print("geometry_path",geometry_path)
            df_uniform_points,df_on_surface,df_narrow_band = data_generator.generate_signed_distance_data(geometry_path,self.config.uniform_points,self.config.surface,self.config.narrowband,self.config.narrowband_width)
            # save the dataframes to CSV
            df_uniform_points.to_csv(os.path.join(self.data_path,"uniform.csv"))
            df_on_surface.to_csv(os.path.join(self.data_path,"surface.csv"))
            df_narrow_band.to_csv(os.path.join(self.data_path,"narrow.csv"))
            print("Sampling done")
            return
    def train(self):
        # making sure that the sample exists 
        #  add a check here to see if the sampling is done
        try:
            print(" Trying Sampling .... ")
            self.sampling()
        except:
            print("Sampling failed")
            return
        # already in the class variables
        training_dataloader, validation_dataloader = load_data.load_data(self.data_path,self.config,self.device)     
        # model as follow
        model = self.model
        # euler characteristict as follow
        # Define your Euler characteristic as a trainable parameter
        euler_characteristic = nn.Parameter(torch.tensor(0.0))

        # use optimizer according to the loss function
        print(f"Loss function is {self.loss.__name__}")
        if self.loss.__name__()=="GaussBonnetLoss":
            optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': [euler_characteristic]}  # Euler characteristic as a separate parameter
        ], lr=self.config.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        # ##################### ADDED FOR TESTING ############################
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        # ####################################################################
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        if self.config.contd:
            # load the best model from the model_save_path
            model, optimizer, start_epoch, loss_per_epoch, best_val_loss,val_loss_per_epoch =\
                Executor.load_model(model,optimizer,self.model_save_path,best=True)  
        else:
            start_epoch = 0
            loss_per_epoch = []
            val_loss_per_epoch = []
            best_val_loss = float('inf')
        # counter for early stopping
        # train the model
        model = model.to(self.device)
        model = model.train()

        counter = 0
        loss_fn = self.loss
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        for i in range(start_epoch, int(self.config.epochs)):
            loss=0
            train_loss = 0
            val_loss = 0
            torch.cuda.empty_cache()

            for batch, (x_batch, y_batch) in enumerate(training_dataloader):
                optimizer.zero_grad()
                # euler_characteristic = torch.tensorint(0)
                # if self.loss.__name__()=="GaussBonnetLoss":

                #     loss = loss_fn(x_batch.to(self.device), y_batch.to(self.device), model,i,torch.round(euler_characteristic))
                # else:
                loss = loss_fn(x_batch.to(self.device), y_batch.to(self.device), model,i)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                del x_batch
                del y_batch
                exit(1)
            # take a step in the scheduler
            # scheduler.step()
            torch.cuda.empty_cache()
            train_loss = train_loss/len(training_dataloader)
            loss_per_epoch.append(train_loss)
            # model.eval()
            # for batch, (x_batch, y_batch) in enumerate(validation_dataloader):
            #     if self.loss.__name__()=="GaussBonnetLoss":
            #         loss = self.loss(x_batch.to(self.device), y_batch.to(self.device), model,i,euler_characteristic)
            #     else:
            #         loss = self.loss(x_batch.to(self.device), y_batch.to(self.device), model,i)
            #     # if model.__name__=="KAN":
            #     #     loss+=model.regularization_loss(1,0)
            #     val_loss += loss.item()
            #     del x_batch
            #     del y_batch
            val_loss = train_loss
            val_loss_per_epoch.append(val_loss)
            model.train()

            # write this to a file 
            str_to_write = f"Epoch {i+1}/{self.config.epochs}: train loss {train_loss} validation loss {val_loss}\n"
            with open(os.path.join(self.train_path,"train_loss.txt"),"a") as f:
                f.write(str_to_write)
            if i%self.config.checkpointing==0:
                print(str_to_write)
                # print(f"\n Euler Characterisiticts is {euler_characteristic.item()}\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # save the model
                # code to save the model 
                Executor.save_model(model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=True)
            else:
                # increase the counter by 1
                counter += 1
            if counter >= self.config.patience and i >= self.config.minepochs:
                print("Early stopping: No improvement for the last {} epochs".format(self.config.patience))
                Executor.save_model(model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=False)
                break
            if i%self.config.checkpointing == 0:
                
                # save the model every 
                Executor.save_model(model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=False)
                fig, ax = plt.subplots()
                ax.plot(loss_per_epoch, label='train_loss')
                ax.plot(val_loss_per_epoch, label='val_loss')
                ax.set_title('Loss vs Epochs')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend() 
                fig.savefig(os.path.join(self.plot_save_path, f"loss{i}.png"))
                plt.close(fig)

        # save the model every 
    @staticmethod
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
    @staticmethod
    def load_model(model,optimizer,save_path,best = False):
        if best:
            checkpoint_path = os.path.join(save_path, "best_model.pkl")
            with open(checkpoint_path, 'rb') as checkpoint_file:
                if not torch.cuda.is_available():
                    print("No CUDA Function")
                    checkpoint_data = CPU_Unpickler(checkpoint_file).load()
                else:
                    checkpoint_data = pickle.load(checkpoint_file)
            model = Executor.model_device_handler(model,checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            epoch = checkpoint_data['epoch']+1
            best_val_loss = checkpoint_data['best_val_loss']
            val_loss_per_epoch = checkpoint_data['val_loss_per_epoch']
            loss_per_epoch = checkpoint_data['loss_per_epoch']
            return model, optimizer, epoch, loss_per_epoch, best_val_loss, val_loss_per_epoch
        else:
            print(f"Looking for model in {save_path}")
            model_files = glob.glob(save_path + '/model_epoch*.pkl')
            # print(f"Model files found {model_files}")
            if len(model_files) == 0:
                print("No model found")
                raise FileNotFoundError
            # pick the last model file
            # Sort the list of model files based on modification time
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            # Pick the last (most recent) model file
            last_model_file = model_files[0]

            checkpoint_path = last_model_file
            print(f"Loading model from {checkpoint_path}")
            print("Here")
            with open(checkpoint_path, 'rb') as checkpoint_file:
                if not torch.cuda.is_available():
                    print("No CUDA Function")
                    checkpoint_data = CPU_Unpickler(checkpoint_file).load()
                else:
                    checkpoint_data = pickle.load(checkpoint_file)
            model = Executor.model_device_handler(model,checkpoint_data['model_state_dict'])
            epoch = checkpoint_data['epoch']+1
            return model, epoch
    @staticmethod
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
        if not torch.cuda.is_available():
            if next(model.parameters()).is_cuda:
                model = model.to('cpu')

            # Remove 'module.' prefix from keys if necessary
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model
        
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
        if self.config.two_dim:
            self.two_dim_contour()
            return

        volume_size = (self.config.cubesize, self.config.cubesize, self.config.cubesize)  # Adjust this based on your requirements
        spacing = (2/volume_size[0], 2/volume_size[1], 2/volume_size[2])
        x = torch.linspace(-1, 1, volume_size[0], device=self.device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        # Reshape the coordinates to create a DataFrame
        coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


        ###########################################################################################
        volume_size = (self.config.cubesize, self.config.cubesize, self.config.cubesize)  # Adjust this based on your requirements

        ###########################################################################################
        batch_size = self.config.ppbatchsize  # Adjust based on available memory
        sdf_values = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        print("Loading the best model\n\n")
        print(f"Loading model from {self.model_save_path}")
        print("\n\n")
        # self.model,_, epoch, _, _, _ = Executor.load_model(self.model,optimizer,self.model_save_path,True)
        self.model,epoch= Executor.load_model(self.model,optimizer,self.model_save_path,False)
        print(f"Model loaded from epoch {epoch}")
        # self.model.eval()
        # batch_size=1000
        with torch.no_grad():
            for i in range(0, coordinates.shape[0], batch_size):
                print(f"\nProcessing batch {i//batch_size}")
                batch_coordinates = coordinates[i:i + batch_size]
                batch_sdf = self.model(batch_coordinates.to(self.device)).to(torch.float32)
                # type(batch_sdf)
                sdf_values.append(batch_sdf)
                batch_sdf = batch_sdf.ravel()
                # how to reduce ram consumption here
                del batch_coordinates

        sdf_values = torch.cat(sdf_values)
        # Reshape the SDF values array to match the volume shape
        sdf_values = sdf_values.cpu().numpy().reshape(volume_size)
        verts, faces, normals, _ = marching_cubes(sdf_values, level=0.0,spacing=spacing)
        print(f"Mesh generated for cube size {self.config.cubesize}")
        if self.config.rescale:
            # save the mesh
            centroid = np.mean(verts)
            verts -= centroid
        # save the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        # mesh.export(os.path.join(save_directory, f"output_mesh{i}.stl"), file_type='stl')
        path_to_save = os.path.join(self.postprocess_save_path, f"{self.geometry_name}_resconstructed_{epoch}_cube_{self.config.cubesize}.stl")
        print(f"Saving mesh to {path_to_save}")
        mesh.export(path_to_save, file_type='stl') 
        plot_stl(path_to_save,os.path.join(self.postprocess_save_path, f"{self.geometry_name}_resconstructed_{epoch}_cube_{self.config.cubesize}.gif"))
    
    def two_dim_contour(self):
        volume_size = (self.config.cubesize, self.config.cubesize, self.config.cubesize)  # Adjust this based on your requirements
        spacing = (2/volume_size[0], 2/volume_size[1], 2/volume_size[2])
        x = torch.linspace(-1, 1, volume_size[0], device=self.device)
        y = torch.linspace(-1, 1, volume_size[1], device=self.device)
        xx, yy = torch.meshgrid(x, y)
        zz = torch.zeros_like(xx)
        coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        sdf_values = []
        batch_size = self.config.ppbatchsize
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        print("Loading the best model\n\n")
        print(f"Loading model from {self.model_save_path}")
        print("\n\n")
        # self.model,_, epoch, _, _, _ = Executor.load_model(self.model,optimizer,self.model_save_path,True)
        model,epoch= Executor.load_model(self.model,optimizer,self.model_save_path,False)
        print(f"Model loaded from epoch {epoch}")
        with torch.no_grad():
            for i in range(0, coordinates.shape[0], batch_size):
                print(f"\nProcessing batch {i//batch_size}")
                batch_coordinates = coordinates[i:i + batch_size]
                batch_sdf = model(batch_coordinates.to(self.device)).to(torch.float32)
                sdf_values.append(batch_sdf)
                del batch_coordinates

        sdf_values = torch.cat(sdf_values)
        sdf_values = sdf_values.cpu().numpy().reshape(volume_size[0], volume_size[1])

        # contours = measure.find_contours(sdf_values, level=0.0)

        # # Plot the contours
        # for contour in contours:
        #     plt.plot(contour[:, 1], contour[:, 0], 'k')
        # Define the threshold
        threshold = 1 / (2 ** 10)

        # Compute the radius of the contour
        contour_points = []
        for i in range(len(xx)):
            for j in range(len(yy)):
                if abs(sdf_values[i, j]) < threshold:
                    contour_points.append((xx[i, j].cpu().numpy(), yy[i, j].cpu().numpy()))
        # Calculate the distance of each point on the contour from the origin (0,0)
        distances_from_origin = [np.sqrt(x**2 + y**2) for x, y in contour_points]
        np_distances = np.array(distances_from_origin)
        # save the distances to a file as csv
        np.savetxt(os.path.join(self.postprocess_save_path, f"{self.geometry_name}_distances_{epoch}_cube_{self.config.cubesize}.csv"), np_distances, delimiter=",")

        plt.figure()
        plt.contourf(xx.cpu().numpy(), yy.cpu().numpy(), sdf_values, levels=0, cmap='coolwarm')
        plt.colorbar()
        circle_radius = np.sqrt(2 / np.pi)
        circle = plt.Circle((0, 0), circle_radius, color='r', fill=False)
        plt.gca().add_patch(circle)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        path_to_save = os.path.join(self.postprocess_save_path, f"{self.geometry_name}_contours_{epoch}_cube_{self.config.cubesize}.png")

        # Set the domain size
        # domain_size = (-1, 1)  # adjust this based on your requirements
        # plt.xlim(domain_size)
        # plt.ylim(domain_size)

        # Show the plot
        path_to_save = os.path.join(self.postprocess_save_path, f"{self.geometry_name}_contours_{epoch}_cube_{self.config.cubesize}.png")
        plt.savefig(path_to_save)

        

        # # Plot the circle
        # circle_radius = np.sqrt(2 / np.pi)
        # circle = plt.Circle((0, 0), circle_radius, color='r', fill=False)
        # plt.gca().add_patch(circle)
        # # Set equal aspect ratio
        # plt.gca().set_aspect('equal', adjustable='box')
        # Save the plot
        # path_to_save = os.path.join(self.postprocess_save_path, f"{self.geometry_name}_contours_{epoch}_cube_{self.config.cubesize}.png")
        # plt.savefig(path_to_save)
    def run(self):
        print("Running the executor")
        # get the name of the loss function
        loss_name = self.config.loss.__name__()
        if self.config.samplingonly and not loss_name=="IGRLOSSPCD":
            print("Sampling only")
            self.sampling()
            return        # inside the train_path create a folder to save the  plots

        if self.config.ppo:
            print("PPO only")
            if self.config.reconstruct:
                print("Reconstructing only")
                self.reconstruct_only()
                return
            post_process.post_process(self)
            return
        self.train()
        return

