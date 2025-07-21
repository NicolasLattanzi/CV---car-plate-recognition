import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet():
    model = models.resnet18(weights = ResNet18_Weights.DEFAULT)
    OUTPUTS = 8
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS)

    return model


class LicensePlateResNet(nn.Module):
    def __init__(self, num_classes):
        super(LicensePlateResNet, self).__init__()

        resnet = models.resnet18(weights=None)  #carico resnet
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #modifico primo layer per accettare immagini rgb
        
        self.features = nn.Sequential(*list(resnet.children())[:-2]) # Remove avgpool and FC
        
        # Sequence modeling: Convert spatial features to sequence
        self.sequence_model = nn.Conv1d(
            in_channels=1024, 
            out_channels=256, 
            kernel_size=3, 
            padding=1)
        
        # Output layer: predict a character for each sequence position
        self.classifier = nn.Conv1d(
            in_channels=256, 
            out_channels=num_classes, 
            kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, height, width)
        feats = self.features(x)  # (batch, 512, H, W)
        batch, channels, height, width = feats.size()
        # Unisce channels e height
        feats = feats.reshape(batch, channels * height, width)  # (batch, 512*H, W)
        seq_feats = self.sequence_model(feats)  # (batch, hidden, width)
        logits = self.classifier(seq_feats)     # (batch, num_classes, width)
        logits = logits.permute(2, 0, 1)        # (seq_len, batch, num_classes)
        return logits


