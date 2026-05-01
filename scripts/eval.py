import torch
from sklearn.metrics import f1_score


def evaluate(model,val_loader,criterion,device):
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_preds = []
        total_loss = 0
        correct_labels = 0
        total_labels = 0
        for x_batch,y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs,y_batch)
            total_loss += loss.item()
            total_labels += y_batch.size(0)
            _,predicted = torch.max(outputs,1)
            correct_labels += (predicted == y_batch).sum().item()
            all_labels.extend(y_batch.detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())
        val_accuracy = (correct_labels*100)/total_labels
        val_f1score = f1_score(all_labels,all_preds,average='macro')
        val_loss = total_loss / len(val_loader)
        return val_loss,val_accuracy,val_f1score

