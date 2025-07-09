import torch
import torch.nn as nn
import torchvision.models as models


def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    OUTPUTS = 8
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS)

    return model


