import torch
import torch.nn as nn
from torch.nn.functional import relu

# CNN
class create_model(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        #  self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(32*4*4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool( relu(x) )
        x = self.conv2(x)
        x = self.pool( relu(x) )

        x = self.adaptive_pool(x) # fixed dimension
        #print(x.shape)
        x = torch.flatten(x, 1)

        #x = x.view(-1, 12*5*5) # flattening
        x = relu( self.fc1(x) )
        x = self.dropout(x)
        x = relu( self.fc2(x) )
        x = self.fc3(x)
        return x
    

#m = create_model()
