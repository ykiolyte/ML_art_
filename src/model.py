# src/model.py

import torch
import torch.nn as nn
from torchvision import models
from timm import create_model

class PS2Net(nn.Module):
    def __init__(self, model_name='efficientnet_v2_s'):
        super(PS2Net, self).__init__()
        self.model_name = model_name
        if model_name == 'efficientnet_v2_s':
            self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, 1)
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)
        elif model_name == 'convnext_tiny':
            self.model = create_model('convnext_tiny', pretrained=True)
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        # Размораживаем все слои
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x  # Без активации

class StackingMetaModel(nn.Module):
    def __init__(self, num_models):
        super(StackingMetaModel, self).__init__()
        self.meta_classifier = nn.Linear(num_models, 1)

    def forward(self, x):
        x = self.meta_classifier(x)
        return x  # Без активации
