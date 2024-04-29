# This file reads config.ini and returns a Configuration object
# The Configuration object contains all the parameters needed for the training

import configparser
from model import networks,losses

class Configuration:
    def __init__(self, file_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(file_path)

        # FILE
        self.geometry = self.config.get("Files", "geometry")
        self.directory = self.config.get("Files","directory")
        self.name = self.config.get("Files","name")
        
        # MODEL PARAMS 
        model = getattr(networks,self.config.get("Model","model"))
        if self.config.get("Model","model")=="ImplicitNet" :
            # for model
            self.hidden_dim = self.config.getint("Model","hidden_dim")
            self.num_hidden_layers = self.config.getint("Model","num_hidden_layers")
            dims = [self.hidden_dim for i in range(self.num_hidden_layers)]
            self.input_dim = self.config.getint("Model","input_dim")
            val = self.config.getint("Model","skip_connection")
            if val == 0:
                self.skip_connection = ()
                self.beta = 0
            else:
                self.skip_connection = (val,)
                self.beta = self.config.getfloat("Model","beta")
            self.geometric_init = self.config.getboolean("Model","geometric_init")
            print("skip_connection",self.skip_connection)
            self.model = model(self.input_dim,dims,self.skip_connection,self.beta,self.geometric_init)
        if self.config.get("Model","model") == "FeedForwardNetwork":
            self.hidden_dim = self.config.getint("Model","hidden_dim")
            self.num_hidden_layers = self.config.getint("Model","num_hidden_layers")
            self.input_dim = self.config.getint("Model","input_dim")
            self.model = model(self.input_dim,self.hidden_dim,self.num_hidden_layers,)
            self.skip_connection = ()
            self.beta = 0
            self.geometric_init= False
        print(self.model)
        # LOSS PARAMS
        self.loss = self.get_loss_function()

        # TRAIN PARAMS
        self.lr = self.config.getfloat("Training", "lr")
        self.epochs = self.config.getint("Training","epochs")
        self.minepochs = self.config.getint("Training","min_epochs")
        self.batchsize = self.config.getint("Training","batch_size")
        self.checkpointing = self.config.getint("Training","checkpointing")
        self.contd = self.config.getboolean("Training","continue")
        self.patience = self.config.getint("Training","patience")
        if self.config.has_option("Training", "two_dim"):
            self.two_dim = self.config.getboolean("Training", "two_dim")
        else:
            self.two_dim = False  # or a default value
        
    
        # SAMPLING PARAMS
        self.samplingonly = self.config.getboolean("Sampling","samplingonly")
        self.continue_sampling = self.config.getboolean("Sampling","continue_sampling")
        self.rescale = self.config.getboolean("Sampling","rescale")
        self.distributed = self.config.getboolean("Sampling","distributed")
        self.uniform_points = self.config.getint("Sampling","uniform_points")
        self.surface = self.config.getint("Sampling","surface")
        self.narrowband = self.config.getint("Sampling","narrowband")
        self.narrowband_width = self.config.getfloat("Sampling","narrowband_width")
        self.mismatchuse = self.config.getboolean("Sampling","mismatchuse")
        self.train_test_split = self.config.getfloat("Sampling","train_test_split")

        #  SETTINGS
        self.ppo = self.config.getboolean("Optional","ppo") # if true only post processing is done
        self.reconstruct = self.config.getboolean("Optional","reconstruct") # if true only reconstruction is done
        self.cubesize = self.config.getint("Optional","cubesize")
        self.ppbatchsize = self.config.getint("Optional","postprocessbatchsize")

    def get_loss_function(self):
        loss_function_name = self.config.get('Loss', 'loss_function')

        # Check if the loss function name is valid and available
        if hasattr(losses, loss_function_name):
            # Get the loss function class dynamically using getattr
            loss_function_class = getattr(losses, loss_function_name)

            # Get parameters from the config
            parameters = {}
            for key in self.config.options('Loss'):
                if key != 'loss_function':
                    parameters[key] = float(self.config.get('Loss', key))

            # Instantiate the loss function with parameters
            return loss_function_class(**parameters)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")


