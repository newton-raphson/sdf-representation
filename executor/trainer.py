import torch.optim.lr_scheduler as lr_scheduler

class CustomScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, initial_lr, last_epoch=-1):
        self.initial_lr = initial_lr
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Implement your custom learning rate scheduling logic here
        # Example: Decay the learning rate by a factor of 0.1 every 10 epochs
        return [self.initial_lr * (0.1 ** (epoch // 10)) for epoch in range(self.last_epoch + 1)]


