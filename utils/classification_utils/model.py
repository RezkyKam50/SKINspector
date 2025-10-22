import torch.nn as nn
from torchvision import models


def setup_model(num_classes):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.45, inplace=True),  
        nn.Linear(in_features=1280, out_features=num_classes)
    )
    
    return model