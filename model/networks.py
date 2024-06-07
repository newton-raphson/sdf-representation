import torch
import torch.nn as nn
import torch.nn.functional as F
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
            self.activation = nn.Softplus(beta=beta)

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
            elif isinstance(self.activation, nn.ReLU):  # If the activation is ReLU
                x = torch.tanh(x)  #
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



class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=256,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    def __name__(self):
        return "KAN"

