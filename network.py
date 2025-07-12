import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


def create_model():
    model = models.resnext50_32x4d(weights = ResNeXt50_32X4D_Weights.DEFAULT)
    OUTPUTS = 8
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS)

    return model


