import torch
import torch.optim as optim
import torch.nn as nn
from utils.logger import Logger
import os
from scripts.B4.b4_eval import evaluate
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

def train(model,optimizer,criterion,train_loader,val_loader,epochs,scheduler,device,output_path):
    logger = Logger(os.path.join(output_path,'logs.csv'))
    writer = SummaryWriter(log_dir=output_path)
    best_loss = float('inf')
    checkpoint_dir = os.path.join(output_path,'checkpoints')
    os.makedirs(checkpoint_dir,exist_ok=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_labels = 0
        total_labels = 0
        all_labels , all_preds = [] , []
        for x_batch,y_batch in train_loader:
            x_batch,y_batch = x_batch.to(device,non_blocking=True),y_batch.to(device,non_blocking=True)
            outputs = model(x_batch)
            loss = criterion(outputs,y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            total_labels += y_batch.size(0)
            correct_labels += (predicted == y_batch).sum().item()

            all_labels.extend(y_batch.detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())

        epoch_loss /= len(train_loader)
        train_accuracy = (correct_labels * 100) / total_labels
        train_f1score = f1_score(all_labels,all_preds,average='macro')
        val_loss , val_accuracy , val_f1score,_,_ = evaluate(model,val_loader,criterion,device)
        logger.write(
            epoch+1,
            epoch_loss,val_loss,
            train_accuracy,val_accuracy,
            train_f1score,val_f1score,
            optimizer.param_groups[0]['lr']
        )
        writer.add_scalar('Val Loss',val_loss,epoch)
        writer.add_scalar('Val Accuracy',val_accuracy,epoch)
        writer.add_scalar('Train Loss',epoch_loss,epoch)
        writer.add_scalar('Train Accuracy',train_accuracy,epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':val_loss
            }
            torch.save(checkpoint,os.path.join(checkpoint_dir,'checkpoint.pth'))
        if scheduler:
            writer.add_scalar('Learning Rate',optimizer.param_groups[0]['lr'],epoch)
            if isinstance(scheduler , optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        print(f"epoch {epoch} | train loss {epoch_loss:.4f} | val loss {val_loss:.4f} | val acc {val_accuracy:.2f} | val f1 {val_f1score:.4f}")
    writer.close()

        