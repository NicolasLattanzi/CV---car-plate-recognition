import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import network
import data
import utils

#############################################################################
#############################################################################
#
# Note: this file was created for the sole purpose of testing th correct
# operation of the functions and models inside this project. to execute the
# actual models, please use the train.py and evaluate.py files.
#
#############################################################################
#############################################################################


# data
dataset = data.CarPlateDataset("CCPD2019")
_, test_dataset = data.train_test_split(dataset)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# detection model 
resnet = torch.load("CV---car-plate-recognition-main/models/detection_model.pth", map_location="cpu", weights_only=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet = resnet.to(device)
resnet.eval()

# recognition model
num_classes = 68 # numero di caratteri supportati
dropout_rate = 0.5

LPRnet = network.build_lprnet(class_num=num_classes, dropout_rate=dropout_rate)
state_dict=torch.load("CV---car-plate-recognition-main/models/Final_LPRNet_model.pth", map_location="cpu", weights_only=False)
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

output2 = LPRnet( resized_img ) # license plate string
output3=utils.tensorToString(output2)


# img_permuted = out.permute(1, 2, 0)
# img_np = img_permuted.detach().cpu().numpy()
# plt.imshow(img_np)
# plt.axis('off')
# plt.show()

print('output:  ', output3)
print('real plate: ', utils.lpDecoder(lp))



