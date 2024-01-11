# loss Function to test 
import torch
import torch.nn as nn

# Every Loss functions def forward(self,x_batch,y_batch,model,epoch):
# forward arguments should be 
class CustomSDFLoss(nn.Module):
    def __init__(self, delta):
        super(CustomSDFLoss, self).__init__()
        self.delta =delta
    # x_batch, y_batch,model,epoch
    def forward(self,x_batch,y_batch,model,epoch):
        # Apply the clamp operation to predicted and target SDF values
        predicted_sdf = torch.clamp(model(x_batch), -self.delta, self.delta)
        target_sdf = torch.clamp(y_batch[:,0], -self.delta, self.delta)

        # Calculate the L2 loss between the clamped SDF values
        loss = (predicted_sdf - target_sdf)**2
        total_loss = loss.mean()
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
    
class WeightedSmoothL2Loss(nn.Module):
    def __init__(self, weight_factor=0.5,delta=0.1,alpha = 0.8):
        super(WeightedSmoothL2Loss, self).__init__()
        self.weight_factor = weight_factor
        self.delta = delta
        self.alpha = alpha
    def __name__(self):
        return "WeightedSmoothL2Loss"
    
    def forward(self, y_true, y_pred,model, pj,true_surface_normal,epoch,surface_normal):
        # Convert inputs to PyTorch tensors if they are not already
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)

        error = y_true - y_pred
        absolute_error = torch.abs(error)
        # calculate the l1 loss function as well here 
        loss = CustomSDFLoss(self.delta)(y_true,y_pred,model, pj,true_surface_normal,epoch,surface_normal)
        # Calculate the weight based on the proximity of y_true to zero
        weight = 1.0 + self.weight_factor * torch.exp(-torch.abs(y_true))

        # Implement mean squared error here using PyTorch operations and use alpha to combine 
        # the two loss functions
        l2_loss = self.alpha*torch.mean(weight * absolute_error**2) + (1-self.alpha)*loss
        return l2_loss
    
class CustomShapeLoss(nn.Module):
    def __init__(self, lambda_g=1e-4):
        super(CustomShapeLoss, self).__init__()
        # initialize the regularization weights
        self.lambda_g = lambda_g

    def forward(self, model_old, model_new, pj,distance_constraint=True):
        # Calculate the squared difference between previous and new model predictions
        with torch.no_grad():
            old_pred = model_old(pj)
        pj.requires_grad = True
        model_new_pred = model_new(pj)
        data_loss = torch.mean((old_pred - model_new_pred)**2)  # Use torch.mean to calculate the mean loss
        # print(model_new_pred.requires_grad)
        # let's use the normal constraint only if the predicted value of the sdf is less than 0.01
        # Calculate the gradient of G and its norm squared
        if distance_constraint == False:
            gradient = torch.where(old_pred.abs() < 1e-06, torch.autograd.grad(model_new_pred, pj, grad_outputs=torch.ones_like(model_new_pred), create_graph=True)[0] , torch.zeros_like(pj))
        else:
            gradient = torch.autograd.grad(model_new_pred, pj, grad_outputs=torch.ones_like(model_new_pred), create_graph=True)[0]
        # gradient = torch.ones_like(model_new_pred)
        gradient_norm = torch.norm(gradient, dim=1)
        gradient_loss = torch.mean((gradient_norm - 1)**2)  # Use torch.mean for mean aggregation
        hessian = []
        for g in gradient:
            h = torch.autograd.grad(g, x)
        hessian.append(h)

        # Convert the Hessian to a square matrix
        hessian_matrix = torch.stack(hessian)

        loss = data_loss + self.lambda_g * gradient_loss

        return loss
    
class IGRLOSS(nn.Module):
    def __init__(self,only_surface,delta=0.1, tau=0.1, lambda_g=1,lipsitch=False):
        super(IGRLOSS, self).__init__()
        self.delta = delta
        self.tau = tau
        self.lambda_g = lambda_g
        self.regularizer_threshold=0.1
        self.alpha=0.01
        self.lipsitch = lipsitch
        # y, y_predicted,model,inputs,true_surface_normal,epoch,surface_normal
    def forward(self,target_sdf, predicted_sdf,model,pj,true_surface_normal,epoch,surface_normal,ci=None):
        # Apply the clamp operation to predicted and target SDF values
        predicted_sdf = torch.clamp(predicted_sdf, -self.delta, self.delta)
        target_sdf = torch.clamp(target_sdf, -self.delta, self.delta)

        # Calculate the  loss between the clamped SDF values
        loss = (predicted_sdf - target_sdf)**2
        # print(f"THE SDF LOSS IS",{loss.mean()})
        regularizer_loss =  torch.zeros_like(loss)
        gradient_norm = surface_normal.norm(2,dim=-1)
        # regularizer_loss = torch.dot(true_surface_normal,surface_normal).abs().norm(2, dim=1)
        # print(f"The Regularizer loss",{regularizer_loss.mean()})
        # element_wise_product = torch.mul(true_surface_normal, surface_normal)
        true_surface_normal = true_surface_normal.view(-1, 3)
        surface_normal = surface_normal.view(-1, 3)
        # regularizer_loss = torch.where(target_sdf.abs() < self.regularizer_threshold, (true_surface_normal-surface_normal/surface_normal.norm())**2 , torch.zeros_like(loss))
        # # Sum along the last dimension to get the dot product for each element in the batch
        # dot_product = torch.sum(element_wise_product, dim=-1)
        # print(dot_product.shape)
        regularizer_loss = torch.where(
            target_sdf.abs() < self.regularizer_threshold,
            (1 - torch.nn.functional.cosine_similarity(true_surface_normal, surface_normal/surface_normal.norm(), dim=-1))**2,
            1e-8*torch.ones_like(loss)
        )
        # regularizer_loss = (1 - torch.nn.functional.cosine_similarity(true_surface_normal, surface_normal, dim=-1))**2


        gradient_loss = torch.where((target_sdf.abs() < self.regularizer_threshold) , ((gradient_norm-1)**2), 1e-8*torch.ones_like(loss))
        # gradient_loss = ((gradient_norm-1)**2)
        # # print(f"The gradient loss is ",{gradient_loss.mean()})
        # # gradient_loss = ((gradient_norm-1)**2)
        # # if epoch%10 == 0:
        # #     print(f"The gradient loss is ",{gradient_loss.mean()})
        # #     print(f"The Regularizer loss",{regularizer_loss.mean()})
        # #     print(f"The SDF loss is",{loss.mean()})
        # # if epoch>20:
        # if epoch>1:
        total_loss = loss.mean() +  self.tau*regularizer_loss.mean() + self.lambda_g*gradient_loss.mean()
        # total_loss = loss.mean() +  self.lambda_g*gradient_loss.mean()
        # else:
        # total_loss = loss.mean() 
        # total_loss = loss.mean()
        # if self.lipsitch == True:
        #     lipsitch_bound = torch.prod(ci)
        #     total_loss = total_loss + self.alpha*lipsitch_bound
        return total_loss
    #         loss{
    #     lambda = 0.1 grad_lambda
    #     normals_lambda = 1.0
    #     latent_lambda = 1e-3
    # }

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