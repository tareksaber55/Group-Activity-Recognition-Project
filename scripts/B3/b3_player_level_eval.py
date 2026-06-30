import torch
from sklearn.metrics import f1_score


def evaluate(model, val_loader, criterion, device):
    model.eval()

    all_labels = []
    all_preds = []
    total_loss = 0
    correct_labels = 0
    total_labels = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            outputs = model(x_batch)   

            logits = outputs.reshape(-1, outputs.shape[-1])
            targets = y_batch.reshape(-1)

            loss = criterion(logits, targets)
            total_loss += loss.item()

            _,predicted = torch.max(outputs, dim=-1) 

            mask = (y_batch != -1)

            correct_labels += ((predicted == y_batch) & mask).sum().item()
            total_labels += mask.sum().item()

            valid_labels = y_batch[mask]
            valid_preds = predicted[mask]

            all_labels.extend(valid_labels.cpu().numpy())
            all_preds.extend(valid_preds.cpu().numpy())

    val_accuracy = 100 * correct_labels / total_labels
    val_f1score = f1_score(all_labels, all_preds, average='macro')
    val_loss = total_loss / len(val_loader)

    return val_loss, val_accuracy, val_f1score, all_labels, all_preds

