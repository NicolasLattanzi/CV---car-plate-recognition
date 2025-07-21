import torch
from torch.utils.data import DataLoader

import network
import data
import utils

###### hyper parameters ########

batch_size = 32
num_epochs = 1
dropout_rate = 0.5
num_classes = 68 # supported characters

###############################

dataset = data.CarPlateDataset("../CCPD2019", training=False)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

data_size = len(data_loader)

# detection model

resnet = torch.load("models/latest_resnet.pth", map_location="cpu")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resnet = resnet.to(device)

# recognition model

LPRnet = network.build_lprnet(class_num=num_classes, dropout_rate=dropout_rate)

state_dict = torch.load("models/Final_LPRNet_model.pth", map_location="cpu")
LPRnet.load_state_dict(state_dict)


print('//  starting evaluation  //')

loss_function = torch.nn.MSELoss()
resnet.eval()
LPRnet.eval()

for epoch in range(num_epochs):
    train_loss = 0.0
    print(f'###\t\t  starting epoch n.{epoch+1}  \t\t###\n')
    for i, (images, vertices, license_plates) in enumerate(data_loader):
        images = images.to(device)
        vertices = vertices.to(device)

        resnet_outputs = resnet(images)

        resized_images = []
        for img in images:
            resized_images.append( utils.crop_photo(img, resnet_outputs) )

        final_outputs = LPRnet(resized_images)

        loss = loss_function(final_outputs, license_plates)
        train_loss += loss.item()

        # printing error every X batch
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{data_size}], Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / data_size
    print(f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {avg_train_loss:.4f}")




