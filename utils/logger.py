import csv
import os

class Logger():
    def __init__(self,file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            with open(file_path,'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch','loss','accuracy','LR'])
        
    def write(self,epoch,loss,accuracy,lr):
        with open(self.file_path,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,loss,accuracy,lr])