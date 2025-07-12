import torch

from network import LPRNet
from network import create_model_Detection

#carico modello detection 
modelDet = torch.load("modelDetection.pth", map_location="cpu")


#carico modello riconoscimento e i suoi pesi

num_classes=68 #numero di caratteri supportati
dropout_rate=0.5

modelReco=LPRNet(class_num=num_classes, dropout_rate=dropout_rate)

state_dict=torch.load("Final_LPRNet_model.pth", map_location="cpu")
modelReco.load_state_dict(state_dict)

