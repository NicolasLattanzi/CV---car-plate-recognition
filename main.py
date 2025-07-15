import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import network
import data
import utils


# data
dataset = data.CarPlateDataset("../CCPD2019")
_, test_dataset = data.train_test_split(dataset)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# detection model 
resnet = torch.load("models/detection_model.pth", map_location="cpu", weights_only=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet = resnet.to(device)
resnet.eval()

# recognition model
num_classes = 68 # numero di caratteri supportati
dropout_rate = 0.5

LPRnet = network.build_lprnet(class_num=num_classes, dropout_rate=dropout_rate)
state_dict=torch.load("models/Final_LPRNet_model.pth", map_location="cpu", weights_only=False)
LPRnet.load_state_dict(state_dict)
LPRnet.eval()


##### test ######

for image, _, plate in test_loader:
    # compute detection on random image
    img = image[0]
    lp = plate
    vertices = resnet(image)
    break

img_permuted = img.permute(1, 2, 0)
img_np = img_permuted.detach().cpu().numpy()

# Visualizza immagine originale
plt.imshow(img_np)
plt.axis('off')
plt.show()

resized_img = utils.crop_photo(img, vertices[0]) # resizing 94x24
print(resized_img.shape)

# Converti in numpy
img_permuted = resized_img.permute(1, 2, 0)
img_np = img_permuted.detach().cpu().numpy()

# Visualizza targa
plt.imshow(img_np)
plt.axis('off')
plt.show()



resized_img = resized_img.float().unsqueeze(0)

out = LPRnet( resized_img ) # license plate string

# img_permuted = out.permute(1, 2, 0)
# img_np = img_permuted.detach().cpu().numpy()
# plt.imshow(img_np)
# plt.axis('off')
# plt.show()

print('output:  ', out)
print('real plate: ', lp)



