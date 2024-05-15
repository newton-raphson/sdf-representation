# loss Function to test 
import torch
import torch.nn as nn

# Every Loss functions def forward(self,x_batch,y_batch,model,epoch):

################################################EXAMPLE CLASS FOR THE LOSS FUNCTION############################################################
# class ExampleLoss(nn.Module):
#     def __init__(self, args):'
#         super(ExampleLoss, self).__init__()
#         self.args = args
#     def __name__(self):
#         return "ExampleLoss"
#     def forward(self, x_batch,y_batch,model,epoch):
#         # Implement loss function here using PyTorch operations
#         return loss
############################################################################################################################################
# create a loss function class here with just mean squared error
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def __name__(self):
        return "MSELoss"
    def forward(self, x_batch,y_batch,model,epoch):
        # Convert inputs to PyTorch tensors if they are not already
        y_true = y_batch[:,0]
        y_pred = model(x_batch)
        # Implement mean squared error here using PyTorch operations
        loss = (y_true - y_pred)**2
        total_loss = loss.mean()
        return total_loss
# implemented as per the DeepSDF paper
class CustomSDFLoss(nn.Module):
    # delta is the threshold value which is used to clamp the predicted and target SDF values
    def __init__(self, delta):
        super(CustomSDFLoss, self).__init__()
        self.delta =delta
    def __name__(self):
        return "CustomSDFLoss"
    # x_batch, y_batch,model,epoch
    def forward(self,x_batch,y_batch,model,epoch):
        # Apply the clamp operation to predicted and target SDF values
        predicted_sdf = torch.clamp(model(x_batch), -self.delta, self.delta)
        target_sdf = torch.clamp(y_batch[:,0], -self.delta, self.delta)

        # Calculate the L2 loss between the clamped SDF values
        loss = nn.functional.mse_loss(predicted_sdf, target_sdf)
        return loss

class WeightedSmoothL2Loss(nn.Module):
    def __init__(self, weight_factor=0.5,delta=0.1):
        super(WeightedSmoothL2Loss, self).__init__()
        self.weight_factor = weight_factor
        self.delta = delta
    def __name__(self):
        return "WeightedSmoothL2Loss"
    
    def forward(self, x_batch,y_batch,model,epoch):
        # Convert inputs to PyTorch tensors if they are not already
        y_true = torch.clamp(y_batch[:,0], -self.delta, self.delta)
        y_pred = torch.clamp(model(x_batch), -self.delta, self.delta)
        error = y_true - y_pred
        absolute_error = torch.abs(error)

        # Calculate the weight based on the proximity of y_true to zero
        weight = 1.0 + self.weight_factor * torch.exp(-torch.abs(y_true))

        l2_loss = torch.mean(weight * absolute_error**2) 
        return l2_loss
  
class CombinedLoss(nn.Module):
    def __init__(self, weight_factor=0.5,delta=0.1,alpha = 0.8):
        super(CombinedLoss, self).__init__()
        self.weight_factor = weight_factor
        self.delta = delta
        self.alpha = alpha
    def __name__(self):
        return "CombinedLoss"
    def forward(self, x_batch,y_batch,model,epoch):
        # Convert inputs to PyTorch tensors if they are not already
        y_true = torch.clamp(y_batch[:,0], -self.delta, self.delta)
        y_pred = torch.clamp(model(x_batch), -self.delta, self.delta)
        error = y_true - y_pred
        absolute_error = torch.abs(error)+ torch.FloatTensor([1e-8]).cuda()
        # calculate the l1 loss function as well here 
        l1_loss = torch.mean(torch.abs(error))
        # Calculate the weight based on the proximity of y_true to zero
        weight = 1.0 + self.weight_factor * torch.exp(-torch.abs(y_true)/self.delta)

        # Implement mean squared error here using PyTorch operations and use alpha to combine 
        # the two loss functions
        l2_loss = torch.mean(weight * absolute_error**2) 
        total_loss = self.alpha*l1_loss + (1-self.alpha)*l2_loss
        return total_loss

class IGRLOSS(nn.Module):
    # delta is the threshold value which is used to clamp the predicted and target SDF values
    # tau is the regularizer weight, regularization is the normal similarity (less rigorous then the direct normal loss)
    # lambda_g is the weight for the eikonal loss
    def __init__(self,delta=0.1, tau=1, lambda_g=0.1,regularizer_threshold=1):
        super(IGRLOSS, self).__init__()
        self.delta = delta
        self.tau = tau ###normal_lambda
        self.lambda_g = lambda_g ###normal_lambda eikonal
        self.regularizer_threshold= regularizer_threshold
    def __name__(self):
        return "IGRLOSS"
        # y, y_predicted,model,inputs,true_surface_normal,epoch,surface_normal
    def forward(self, x_batch,y_batch,model,epoch):
        # Apply the clamp operation to predicted and target SDF values
        
        predicted_sdf = torch.clamp(model(x_batch), -self.delta, self.delta)
        target_sdf = torch.clamp(y_batch[:,0], -self.delta, self.delta)

        # Calculate the  loss between the clamped SDF values
        loss = (predicted_sdf - target_sdf)**2
        regularizer_loss =  torch.zeros_like(loss)

        # calculate the surface normal
        surface_normal = compute_normal(model, x_batch)
        gradient_norm = surface_normal.norm(2,dim=-1) 
        true_surface_normal = y_batch[:,1:].view(-1, 3)
        surface_normal = surface_normal.view(-1, 3)
        # regularizer loss is the normal similarity
        regularizer_loss = torch.where(
            target_sdf.abs() < self.regularizer_threshold,
            (1 - torch.nn.functional.cosine_similarity(true_surface_normal, surface_normal/surface_normal.norm(), dim=-1))**2,
            1e-8*torch.ones_like(loss)
        )

        # gradient loss is the eikonal loss
        gradient_loss = torch.where((target_sdf.abs() < self.regularizer_threshold) , ((gradient_norm-1)**2), 1e-8*torch.ones_like(loss))

        total_loss = loss.mean() +  self.tau*regularizer_loss.mean() + self.lambda_g*gradient_loss.mean()
        return total_loss
    
class RegularizedCustomSDFLoss(nn.Module):
    def __init__(self, delta,threshold=1):
        super(RegularizedCustomSDFLoss, self).__init__()
        self.delta =delta
        self.epsilon = 1e-3
        self.regularizer_weight = 1e2
        self.regularizer_threshold = threshold
    
    
    def forward(self,x_batch,y_batch,model,epoch):
        # Apply the clamp operation to predicted and target SDF values
        predicted_sdf = torch.clamp(model(x_batch), -self.delta, self.delta)
        target_sdf = torch.clamp(y_batch[:,0], -self.delta, self.delta)

        # Calculate the L1 loss between the clamped SDF values
        loss = (predicted_sdf - target_sdf)**2
        regularizer_loss =  torch.zeros_like(loss)
        regularizer_loss = torch.where(target_sdf.abs() < self.regularizer_threshold, (true_surface_normal-surface_normal)**2 , torch.zeros_like(loss))
        total_loss = loss.mean() +  self.regularizer_weight*regularizer_loss.mean()
        return total_loss

def compute_normal(model, pj):
    # Make predictions using the model
    pj.requires_grad = True
    phi_pj = model(pj)
    # # Create a tensor of ones with the same shape as phi_pj
    d_points = torch.ones_like(phi_pj, requires_grad=False, device=phi_pj.device)
    points_grad = torch.autograd.grad(
        outputs=phi_pj,
        inputs=pj,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad



