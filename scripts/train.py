import torch
import torch.optim as optim
import torch.nn as nn
from utils.logger import Logger
import os
from eval import evaluate
from sklearn.metrics import f1_score

def train(model,optimizer,criterion,train_loader,val_loader,n_epoch,scheduler,device,output_path):
    os.makedirs(output_path,exist_ok=True)
    logger = Logger(output_path)
    model.to(device)
    best_loss = float('inf')
    for epoch in range(n_epoch):
        model.train()
        epoch_loss = 0
        correct_labels = 0
        total_labels = 0
        all_labels , all_preds = [] , []
        for x_batch,y_batch in train_loader:
            x_batch,y_batch = x_batch.to(device),y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs,y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _,predicted = torch.max(outputs.data,1)
            total_labels += y_batch.size(0)
            correct_labels += (predicted == y_batch).sum().item()

            all_labels.extend(y_batch.detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())

        epoch_loss /= len(train_loader)
        train_accuracy = (correct_labels * 100) / total_labels
        train_f1score = f1_score(all_labels,all_preds,average='macro')
        val_loss , val_accuracy , val_f1score = evaluate(model,val_loader)
        logger.write(
            epoch,
            epoch_loss,val_loss,
            train_accuracy,val_accuracy,
            train_f1score,val_f1score,
            optimizer.param_groups[0]['lr']
        )
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':val_loss
            }
            checkpoint_dir = os.path.join(output_path,'checkpoints')
            os.makedirs(checkpoint_dir,exist_ok=True)
            torch.save(checkpoint,os.path.join(checkpoint_dir,'checkpoint.pth'))
        if scheduler:
            if isinstance(scheduler , optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        print(f"epoch {epoch} | train loss {epoch_loss:.4f} | val loss {val_loss:.4f} | val acc {val_accuracy:.2f} | val f1 {val_f1score:.4f}")

        
        