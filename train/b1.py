import yaml
from scripts.train import train 
from scripts.eval import evaluate
from scripts.test_report import report
from utils import dataset
from models import b1
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import shutil


# select config path
config_path = 'configs/b1_config1.yaml'

with open(config_path,'r') as f:
    config_dict = yaml.safe_load(f)


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model
model = b1.Baseline1().to(device)

# optimizer
optimizer_name = config_dict['train']['optimizer']['type']
lr = config_dict['train']['optimizer']['lr']
weight_decay = config_dict['train']['optimizer']['weight_decay']

if optimizer_name == 'adamw':
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)
else :
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          weight_decay=weight_decay,
                          momentum=0.9)

# epochs
epochs = config_dict['train']['epochs']


scheduler_name = config_dict['train']['scheduler']['type']
if scheduler_name == 'reduce_on_plateau':
    factor = config_dict['train']['scheduler']['factor']
    patience = config_dict['train']['scheduler']['patience']
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=patience,factor=factor)
elif scheduler_name == 'cosine_annealing':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
else:
    scheduler = None

# criterion
criterion = nn.CrossEntropyLoss()




# dataset
input_root = config_dict['dataset']['root']
annot_file = config_dict['dataset']['annot_file']
train_ids = config_dict['dataset']['splits']['train']
val_ids = config_dict['dataset']['splits']['val']
test_ids = config_dict['dataset']['splits']['test']
categories_dict = config_dict['dataset']['classes']
# image has a lot of space around objects. Let's crop around
preprocessor = transforms.Compose([ 
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = dataset.ImageLevelDataset(input_root,annot_file,categories_dict,train_ids,preprocessor)
val_dataset = dataset.ImageLevelDataset(input_root,annot_file,categories_dict,val_ids,preprocessor)
test_dataset = dataset.ImageLevelDataset(input_root,annot_file,categories_dict,test_ids,preprocessor)

train_loader = DataLoader(train_dataset,batch_size=config_dict['train']['batch_size']['train'],shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=config_dict['train']['batch_size']['val'],shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=config_dict['train']['batch_size']['val'],shuffle=False)

# output_path
output_path = os.path.join(config_dict['experiment']['output_dir'],
                           config_dict['experiment']['name'])
os.makedirs(output_path, exist_ok=True)

train(model,optimizer,criterion,train_loader,val_loader,epochs,scheduler,device,output_path)
shutil.copy(config_path, os.path.join(output_path, "config.yaml"))

checkpoint_dict = torch.load(os.path.join(output_path,'checkpoints','checkpoint.pth'),map_location=device)
model.load_state_dict(checkpoint_dict['model_state_dict'])

_,_,_,all_labels,all_preds =  evaluate(model,test_loader,criterion,device)
final_report = report(all_labels,all_preds,team_level=True)
final_report.make_report(output_path)

