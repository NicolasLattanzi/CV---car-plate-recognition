import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import network
import data
import utils
from paddleocr import PaddleOCR
import numpy as np
import cv2

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
dataset = data.CarPlateDataset("../../CCPD2019")
_, test_dataset = data.train_test_split(dataset)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# detection model 
resnet = torch.load("CV---car-plate-recognition-main/models/detection_model.pth", map_location="cpu", weights_only=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet = resnet.to(device)
resnet.eval()

# recognition model

ocr=PaddleOCR(use_angle_cls=True, lang='en')


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
test2=np.transpose(resized_img,(1,2,0))
plt.imshow(test2)
plt.axis('off')
plt.show()
test=cv2.imread("CV---car-plate-recognition-main/京PL3N67.jpg")



# Converti in numpy
#img_permuted = resized_img.permute(1, 2, 0)
#img_np = resized_img.detach().cpu().numpy()

# Visualizza targa
#img_np=np.transpose(img_np, (1,2,0))
#test1=np.transpose(test, (1,2,0))
plt.imshow(test)
plt.axis('off')
plt.show()

if img_np.max()<=1.0:
    img_np=(img_np*255).astype(np.uint8)
else:
    img_np=img_np.astype(np.uint8)





resized_img = resized_img.float().unsqueeze(0)
print(resized_img)
results=[]
result=ocr.predict(test)
for line in result[0]:
    results.append(line[1][0])

print(results)



# img_permuted = out.permute(1, 2, 0)
# img_np = img_permuted.detach().cpu().numpy()
# plt.imshow(img_np)
# plt.axis('off')
# plt.show()

#print('output:  ', output3)
print('real plate: ', utils.lpDecoder(lp))



