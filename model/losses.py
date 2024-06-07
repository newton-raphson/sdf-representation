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
        print("X_batch",x_batch.shape)
        print("Y_batch",y_batch.shape)
        # print all the 
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
class IGRLOSSPCD(nn.Module):
    # delta is the threshold value which is used to clamp the predicted and target SDF values
    # tau is the regularizer weight, regularization is the normal similarity (less rigorous then the direct normal loss)
    # lambda_g is the weight for the eikonal loss
    def __init__(self,delta=0.1, tau=1, lambda_g=0.1,regularizer_threshold=1,local_sigma=0.01,global_sigma=0.1):
        super(IGRLOSSPCD, self).__init__()
        self.delta = delta
        self.tau = tau ###normal_lambda
        self.lambda_g = lambda_g ###normal_lambda eikonal
        self.regularizer_threshold= regularizer_threshold
        self.local_sigma = local_sigma
        self.global_sigma = global_sigma
    def __name__(self):
        return "IGRLOSSPCD"
        # y, y_predicted,model,inputs,true_surface_normal,epoch,surface_normal
    def forward(self, x_batch,y_batch,model,epoch):
        # Apply the clamp operation to predicted and target SDF values
        
        predicted_sdf = model(x_batch)

        # calculate the mnfld loss
        loss = (predicted_sdf)**2
        loss = loss.mean()

        non_mnfld_points = self.get_points(x_batch)
        surface_normal = compute_normal(model, x_batch)
        gradient_norm = surface_normal.norm(2,dim=-1) 

        # compute the eikonol loss
        non_mnfld_loss = ((gradient_norm-1)**2).mean()
        

        return loss + self.lambda_g*non_mnfld_loss
        
        return total_loss
    def get_points(self, pc_input, local_sigma=None):

        sample_size, dim = pc_input.shape

        if local_sigma is not None:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma.unsqueeze(-1))
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        sample_global = (torch.rand(sample_size // 8, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=0)
        return sample
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
class GaussBonnetLoss(nn.Module):
    def __init__(self, delta=0.1, tau=1, lambda_g=0.1, regularizer_threshold=1, gauss_bonnet_weight=0.1):
        super(GaussBonnetLoss, self).__init__()
        self.delta = delta
        self.tau = tau
        self.lambda_g = lambda_g
        self.regularizer_threshold = regularizer_threshold
        self.gauss_bonnet_weight = gauss_bonnet_weight

    def __name__(self):
        return "GaussBonnetLoss"

    def forward(self, x_batch, y_batch, model, epoch, euler_characteristic):
        predicted_sdf = torch.clamp(model(x_batch), -self.delta, self.delta)
        target_sdf = torch.clamp(y_batch[:, 0], -self.delta, self.delta)

        loss = (predicted_sdf - target_sdf) ** 2
        regularizer_loss = torch.zeros_like(loss)

        surface_normal = compute_normal(model, x_batch)
        gradient_norm = surface_normal.norm(2, dim=-1)
        true_surface_normal = y_batch[:, 1:].view(-1, 3)
        surface_normal = surface_normal.view(-1, 3)
        
        # gaussian curvature for gauss-bonnet loss
        gaussian_curvature = compute_gaussian_curvature(model, x_batch)

        regularizer_loss = torch.where(
            target_sdf.abs() < self.regularizer_threshold,
            self.tau*(1 - torch.nn.functional.cosine_similarity(true_surface_normal, surface_normal))**2+
            self.lambda_g*(gradient_norm - 1) ** 2+
            self.gauss_bonnet_weight* (gaussian_curvature- 2 * torch.pi * euler_characteristic)**2,
            1e-8*torch.ones_like(loss)
        )
        




        # gradient_loss = torch.where(
        #     target_sdf.abs() < self.regularizer_threshold,
        #     (gradient_norm - 1) ** 2,
        #     1e-8 * torch.ones_like(loss)
        # )

        # compute the gauss bonnet loss just using the curvature

        # # find which of the loss among the above is nan
        # if torch.isnan(gradient_loss).any():
        #     print("Gradient loss is nan")
        #     exit(0)
        # if torch.isnan(regularizer_loss).any():
        #     print("Regularizer loss is nan")
        #     exit(0)
        # if torch.isnan(loss).any():
        #     print("Loss is nan")
        #     exit(0)
        total_loss = loss.mean() +  regularizer_loss.mean()
        # if(epoch>5):
        #     print("USING BONNET LOSS")
        #     gauss_bonnet_loss = self.compute_gauss_bonnet_loss(x_batch, model, euler_characteristic)
        #     total_loss += self.gauss_bonnet_weight * gauss_bonnet_loss.mean()

        return total_loss

    def compute_gauss_bonnet_loss(self, x_batch, model, euler_characteristic):
        gaussian_curvature = compute_gaussian_curvature(model, x_batch)


        sdf = model(x_batch)
        surface_mask = torch.abs(sdf) < self.regularizer_threshold

        gauss_bonnet_integral = torch.where(surface_mask, gaussian_curvature, torch.zeros_like(gaussian_curvature)).sum()
        boundary_integral = torch.where(surface_mask, geodesic_curvature, torch.zeros_like(geodesic_curvature)).sum()
        
        gauss_bonnet_term = gauss_bonnet_integral + boundary_integral - 2 * torch.pi * euler_characteristic
        return gauss_bonnet_term ** 2
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
def compute_hessian(model, points):
    points.requires_grad = True
    sdf_values = model(points)
    gradients = torch.autograd.grad(
        outputs=sdf_values,
        inputs=points,
        grad_outputs=torch.ones_like(sdf_values),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    hessian = []
    for i in range(gradients.shape[1]):
        grad_grad = torch.autograd.grad(
            outputs=gradients[:, i],
            inputs=points,
            grad_outputs=torch.ones_like(gradients[:, i]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        hessian.append(grad_grad)
    hessian = torch.stack(hessian, dim=-1)
    return hessian
def compute_gradient(model, points):
    points.requires_grad = True
    sdf_values = model(points)
    gradients = torch.autograd.grad(
        outputs=sdf_values,
        inputs=points,
        grad_outputs=torch.ones_like(sdf_values),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return gradients
def compute_gaussian_curvature(model, points):
    gradient = compute_gradient(model, points)
    hessian = compute_hessian(model, points)
    gradient_norm = torch.norm(gradient, dim=1, keepdim=True)
    hessian_determinant = torch.det(hessian)
    gaussian_curvature = hessian_determinant / (1 + gradient_norm**2)**2
    return gaussian_curvature
# def compute_tangent(normal):
#     if torch.allclose(normal[:, -1], torch.zeros_like(normal[:, -1])):
#         fixed_vector = torch.tensor([0, 1, 0], device=normal.device, dtype=normal.dtype)
#     else:
#         fixed_vector = torch.tensor([0, 0, 1], device=normal.device, dtype=normal.dtype)
#     tangent = torch.cross(normal, fixed_vector.unsqueeze(0).repeat(normal.size(0), 1))
#     tangent = tangent / tangent.norm(dim=1, keepdim=True)
#     return tangent
# def compute_geodesic_curvature(model, points):
#     normal = compute_gradient(model, points)
#     normal = normal.norm(2,dim=-1).view(-1,3) 
#     tangent = compute_tangent(normal)
#     # find which is nan here
#     if torch.isnan(normal).any():
#         print("Normal is nan")
#         exit(0)
#     if torch.isnan(tangent).any():
#         print("Tangent is nan")
#         exit(0)

#     # tangent.requires_grad = True
#     tangent_derivative = torch.autograd.grad(
#         outputs=tangent,
#         inputs=points,
#         grad_outputs=torch.ones_like(tangent),
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
#     geodesic_curvature = torch.norm(torch.cross(tangent_derivative, normal, dim=1), dim=1)
#     return geodesic_curvature