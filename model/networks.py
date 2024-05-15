import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.utils import weight_norm

class ImplicitNet(nn.Module):
    """_summary_
    This class defines the architecture of the implicit network which can be 
        converted to a Fully Connected Network if desired.
    _description_
    The implicit network consists of a series of fully connected layers with
    ReLU activations. The number of layers and the number of hidden units per
    layer are specified by the user. The input dimension is 3, corresponding to
    the x, y, and z coordinates of a point in 3D space. The output dimension is
    1, corresponding to the SDF value at that point.
    _attributes_
    num_hidden_layers (int): The number of hidden layers in the network.
    hidden_dim (int): The number of hidden units per hidden layer.
    input_dim (int): The dimension of the input to the network.
    skip_connection (tuple): A tuple of integers specifying the layers that
        should have skip connections.
    beta (float): The beta parameter for the softplus activation function.
    geometric_init (bool): Whether to use geometric initialization.
    _methods_
    forward(input): Forward pass through the network.
    """  
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        beta=100,
        geometric_init=True,
        radius_init=1,
        lipsitch= False,
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        #  if skip_in is empty then this network will
        #  be a fully connected network
        #  last layer is activated by tanh
        #  making the network a FCN as described
        #  in the paper: https://arxiv.org/pdf/1901.05103.pdf
        self.skip_in = skip_in
        print("skip_in",skip_in)
        # not tested for this project
        # implementated on the basis of following paper
        # https://arxiv.org/pdf/2202.08345.pdf
        self.lipsitch = lipsitch
        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta,threshold=0)

        else:
            self.activation = nn.ReLU()

    def __name__(self):
        return "ImplicitNet"
    def forward(self, input):

        x = input
        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            # if self.lipsitch:
            #     self.normalization(lin.parameters(),1/self.ci[layer]*nn.softplus(beta=self.ci[layer]))
            
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
            # save each layer into npy files 
            
        return x
    # normalization for lipsitch implementation 
    # not tested for this project
    def normalization(W,ci):
        row_sum = torch.sum(W, dim=1)
        scale = torch.min(1, ci / row_sum)
        return W * scale

class ImplicitNetCompatible(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=1,
        lipsitch= False,
        beta=99
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.lipsitch = lipsitch
        self.layers = nn.ModuleList()
        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:
                    # nn.init.normal_(lin.weight, mean=torch.sqrt(torch.tensor(np.pi)) / torch.sqrt(torch.tensor(dims[layer])), std=0.00001)
                    torch.nn.init.normal_(lin.weight,  mean=torch.sqrt(torch.tensor(math.pi)) / torch.sqrt(torch.tensor(dims[layer])), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))
                
            self.layers.append(lin)
            

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()
        

    def forward(self, input):

        x = input

        for layer, lin in enumerate(self.layers):
            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / math.sqrt(2)
            # if self.lipsitch:
            #     self.normalization(lin.parameters(),1/self.ci[layer]*nn.softplus(beta=self.ci[layer]))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return x
    def normalization(W,ci):
        row_sum = torch.sum(W, dim=1)
        scale = torch.min(1, ci / row_sum)
        return W * scale

# Tested but did not work
# class SinActivation(nn.Module):
#     def forward(self, x):
#         return torch.sin(torch.pi*x)

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512, num_layers=8):
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                weight_norm(nn.Linear(input_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=0.5)  # You can adjust the dropout probability as needed
            ))
            input_dim = hidden_dim  # Update input dimension for subsequent layers

        self.output_layer = nn.Sequential(
            weight_norm(nn.Linear(hidden_dim, 1)),  # Output a single SDF value
            nn.Tanh()  # Apply tanh activation
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        sdf_value = self.output_layer(x)
        return sdf_value
    def __name__(self):
        return "FeedForwardNetwork"

