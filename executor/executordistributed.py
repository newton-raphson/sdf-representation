from executor.executor import Executor
from model.losses import compute_normal
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import grad


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad

class DistributedExecutor(Executor):
    def __init__(self,config):
        super().__init__(config)
        self.global_sigma=0.1
        self.local_sigma=0.0001
        self.grad_lambda = self.loss.lambda_g

    def train(self):
        # read the surface points
        df = pd.read_csv(os.path.join(self.config.geometry, "surface.csv"),usecols=[0, 1, 2])
        # load it into dataloader
        df=df.apply(pd.to_numeric, errors='coerce')
        df=df.dropna()
        print(df.describe())
        X = torch.tensor(df.values, dtype=torch.float32)

        training_dataloader = torch.utils.data.DataLoader(X, batch_size=self.config.batchsize, shuffle=True, num_workers=30)
        self.model.to(self.device)

        # val_X = val_X.to(self.device)
        # val_Y = val_Y.to(self.device)

        # model = self.model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        if self.config.contd:
            # load the best model from the model_save_path
            self.model, optimizer, start_epoch, loss_per_epoch, best_val_loss,val_loss_per_epoch =\
                DistributedExecutor.load_model(self.model,optimizer,self.model_save_path,best=True)  
        else:
            start_epoch = 0
            loss_per_epoch = []
            val_loss_per_epoch = []
            best_val_loss = float('inf')
        # counter for early stopping
        # train the model
        self.model.train()
        for i in range(start_epoch, int(self.config.epochs)):
            loss=0
            train_loss = 0
            torch.cuda.empty_cache()
            for batch, x_batch in enumerate(training_dataloader):
                x_batch = x_batch.to(self.device)
                sampled_pts = self.get_points(x_batch)
                
                pred = self.model(x_batch)
                # print(x_batch)
                sampled_pts.requires_grad = True
                pred_sample = self.model(sampled_pts)

                # print(f"The prediction is {pred}")
                surface_loss = (pred.abs()).mean()
                # add the eikonal loss only to the points not in surface
                gradients = gradient(sampled_pts,pred_sample)
                grad_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
                # grad_loss = 0
                # print(f"\n The Grad Loss is {grad_loss} and surface loss is {surface_loss}\n")
                loss = surface_loss+ self.grad_lambda*grad_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                del x_batch
                del sampled_pts
            torch.cuda.empty_cache()
            train_loss = train_loss/len(training_dataloader)
            loss_per_epoch.append(train_loss)
            val_loss = 0
            # write this to a file 
            str_to_write = f"Epoch {i+1}/{self.config.epochs}: train loss {train_loss}\n"
            print(str_to_write)
            with open(os.path.join(self.train_path,"train_loss.txt"),"a") as f:
                f.write(str_to_write)
            if i%(1.5*self.config.checkpointing) == 0:
                DistributedExecutor.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=True)
            if i%self.config.checkpointing == 0:
                # save the model every 
                DistributedExecutor.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=False)
                fig, ax = plt.subplots()
                ax.plot(loss_per_epoch, label='train_loss')
                ax.set_title('Loss vs Epochs')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend() 
                fig.savefig(os.path.join(self.plot_save_path, f"loss{i}.png"))
                plt.close(fig)
    def get_points(self, pc_input, local_sigma=None):
        # print(pc_input.shape)
        sample_size, dim = pc_input.shape

        # just return 30% of the input points
        sample = pc_input[torch.randperm(sample_size)[:sample_size // 3]]


        sample_local = sample + (torch.randn_like(sample) * self.local_sigma)
        sample_local = sample_local.to(pc_input.device)
        # print(sample_local.shape)
        # sample_global = (torch.rand(sample_size // 8, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma
        # print(sample_global.shape)
        # sample = torch.cat([sample_local, sample_global], dim=0)

        return sample_local

    