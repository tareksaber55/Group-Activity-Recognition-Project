import csv
import os

class Logger():
    def __init__(self,file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            with open(file_path,'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch',
                                 'train_loss','val_loss',
                                 'train_accuracy','val_accuracy',
                                 'LR'])
        
    def write(self,epoch,train_loss,val_loss,train_accuracy,val_accuracy,accuracy,lr):
        with open(self.file_path,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,
                             train_loss,val_loss,
                             train_accuracy,val_accuracy
                             ,lr])