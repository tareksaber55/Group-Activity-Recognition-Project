import torch
import torch.optim as optim
import torch.nn as nn
from utils.logger import Logger
import os

def train(model,optimizer,criterion,dataloader,n_epoch,scheduler,device,output_path):
    os.makedirs(output_path,exist_ok=True)
    logger = Logger(output_path)
    model.to(device)
    for epoch in range(n_epoch):
        epoch_loss = 0
        for x_batch,y_batch in dataloader:
            x_batch,y_batch = x_batch.to(device),y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs,y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(dataloader)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':epoch_loss
            }
            checkpoint_dir = os.path.join(output_path,'checkpoints')
            os.makedirs(checkpoint_dir,exist_ok=True)
            torch.save(checkpoint,os.path.join(checkpoint_dir,f'checkpoint_{epoch}.pth'))
        if scheduler:
            if isinstance(scheduler , optim.lr_scheduler.ReduceLROnPlateau()):
                # scheduler.step(val_loss)
                pass
            else:
                scheduler.step()

        
        