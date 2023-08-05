import torch.nn as nn
from torchvision import models

def build_model(pretrained=True, fine_tune=True, num_classes=10):
    model = models.resnet34(pretrained=pretrained)
    
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for param in model.parameters():
            param.requires_grad = False
    
    # Change the final classification head.
    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    
    return model

