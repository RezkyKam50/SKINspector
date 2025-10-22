import torch, torch.nn as nn, torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from cuml.metrics import accuracy_score

import matplotlib.pyplot as plt

from model import setup_model

from preprocess import (
    prepare_data
)

from dataloader import (
    get_transforms,
    SkinDataset
)


import os, pandas as pd
from datetime import datetime
from collections import Counter
 

def compute_class_weights(dataset):
    if hasattr(dataset, "df"):
 
        if not isinstance(dataset.df, (pd.DataFrame,)):
            df = dataset.df.to_pandas()
        else:
            df = dataset.df
        labels = df["disease"].tolist()
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise AttributeError("Could not find 'disease' labels in dataset.")
    
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}

    class_weights = torch.tensor(
        [weights[cls] for cls in dataset.label_to_idx.keys()],
        dtype=torch.float32
    )
    return class_weights


# for BF16, we dont need gradscaler since it has better dynamic range than FP16
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda', log_dir=None):
    model = model.to(device)

    if log_dir is None:
        log_dir = os.path.join('./utils/classification_utils/logs/', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_acc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.bfloat16)   
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):   
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
         
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
 
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device, dtype=torch.bfloat16)   
            labels = labels.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):   
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc.cpu().numpy())
        
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        
        scheduler.step()
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model.state_dict().copy()
            writer.add_scalar('Best_accuracy', best_acc, epoch)
 
    model.load_state_dict(best_model_wts)
 
    writer.close()
 
    plot(train_losses, val_losses, val_accuracies)

    return model, train_losses, val_losses, val_accuracies

def plot(train_losses, val_losses, val_accuracies):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Validation Accuracy over epochs')

    plt.tight_layout()
    plt.show()


def prepare_dataset(dataset=None):
    train_df, val_df = prepare_data()
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = SkinDataset(train_df, transform=train_transform, is_train=True)
    val_dataset = SkinDataset(val_df, transform=val_transform, is_train=False, 
                            label_mapping=train_dataset.label_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    num_classes = len(train_dataset.labels)

    return train_loader, val_loader, train_dataset, val_dataset, num_classes


def train(epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, train_dataset, val_dataset, num_classes = prepare_dataset()
    
    model = setup_model(num_classes)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)   
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3,   
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,  
        num_epochs=epochs, device=device
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_mapping': train_dataset.label_to_idx,
        'class_names': train_dataset.labels,
        'num_classes': num_classes
    }, './models/EF-NET_DERMA.pth')
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, dtype=torch.bfloat16)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
 
if __name__ == "__main__":
    train()