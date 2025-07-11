import torch.nn as nn
import torchvision.models as models


def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    OUTPUTS = 8
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS)

    return model


