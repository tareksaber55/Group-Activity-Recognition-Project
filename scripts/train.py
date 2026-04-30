import torch
import torch.optim as optim
import torch.nn as nn
from utils.logger import Logger


def train(model,optimizer,criterion,dataloader,n_epoch,sheduler,device,output_path):
    logger = Logger(output_path)
    model.to(device)
    for epoch in n_epoch:
        for x_batch,y_batch in dataloader:
            x_batch,y_batch = x_batch.to(device),y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs,y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            pass


        
        