import torch
import random
from network import LPRNet
from network import create_model_Detection
from train import test_dataset
import utils
import matplotlib.pyplot as plt

#carico modello detection 
modelDet = torch.load("modelDetection.pth", map_location="cpu")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
modelDet = modelDet.to(device)

#picking random image
test_size = len(test_dataset)
int=random.randint(0,test_size-1)
image, label=test_dataset[int]

#compute detection on random image
output1=modelDet(image) #output1=vertici della targa nell'immagine originale

input2=utils.LP_photo(image, output1) #immagine 94x24 ottenuta dall'immagine originale

#carico modello riconoscimento e i suoi pesi

num_classes=68 #numero di caratteri supportati
dropout_rate=0.5

modelReco=LPRNet(class_num=num_classes, dropout_rate=dropout_rate)

state_dict=torch.load("Final_LPRNet_model.pth", map_location="cpu")
modelReco.load_state_dict(state_dict)


#passo il tensore ottenuto al modello
output2=modelReco(input2) #stringa con i caratteri della targa
print(output2)
plt.imshow(utils.BgrToRgb(image))  # stampo l'immaigne originale convertita in rgb per confronto.
plt.title("Immagine originale CCPD")
plt.axis('off')
plt.show()



