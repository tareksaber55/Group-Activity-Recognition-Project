import yaml
from scripts.B3.b3_group_level_train import train 
from scripts.B3.b3_group_level_eval import evaluate
from scripts.test_report import report
from utils.dataset import PlayerGroupDataset
from models.b3 import B3GroupClassifier,B3PlayerClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import shutil


# select config path
config_path = 'configs/b3_group_classifier_config1.yaml'

with open(config_path,'r') as f:
    config_dict = yaml.safe_load(f)


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# backbone
with open(config_dict['train']['backbone'],'rb') as f:
    backbone_dict =  torch.load(f,map_location=device)

backbone = B3PlayerClassifier().to(device)
backbone.load_state_dict(state_dict=backbone_dict['model_state_dict'])

# model
model = B3GroupClassifier(backbone=backbone,num_classes=len(config_dict['dataset']['group_classes'])).to(device)

# optimizer
optimizer_name = config_dict['train']['optimizer']['type']
lr = config_dict['train']['optimizer']['lr']
weight_decay = config_dict['train']['optimizer']['weight_decay']

if optimizer_name == 'adamw':
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=float(weight_decay)
    )
else :
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=float(weight_decay),
        momentum=0.9
    )

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
criterion = nn.CrossEntropyLoss(ignore_index=-1)




# dataset
input_root = config_dict['dataset']['root']
annot_file = config_dict['dataset']['annot_file']
train_ids = config_dict['dataset']['splits']['train']
val_ids = config_dict['dataset']['splits']['val']
test_ids = config_dict['dataset']['splits']['test']
player_dict = config_dict['dataset']['player_classes']
group_dict = config_dict['dataset']['group_classes']

# image has a lot of space around objects. Let's crop around
# Do NOT use RandomHorizontalFlip , classes contain direction : l-pass , r-pass

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(
        (224, 224),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1)
    ),

    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = PlayerGroupDataset(input_root,annot_file,player_dict,group_dict,train_ids,train_transform,one_frame=True)
val_dataset = PlayerGroupDataset(input_root,annot_file,player_dict,group_dict,val_ids,val_transform,one_frame=True)
test_dataset = PlayerGroupDataset(input_root,annot_file,player_dict,group_dict,test_ids,val_transform,one_frame=True)

num_workers = config_dict['train']['num_workers']

train_loader = DataLoader(train_dataset,batch_size=config_dict['train']['batch_size']['train'],shuffle=True,num_workers=num_workers,pin_memory=True)
val_loader = DataLoader(val_dataset,batch_size=config_dict['train']['batch_size']['val'],shuffle=False,num_workers=num_workers,pin_memory=True)
test_loader = DataLoader(test_dataset,batch_size=config_dict['train']['batch_size']['val'],shuffle=False,num_workers=num_workers,pin_memory=True)

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

