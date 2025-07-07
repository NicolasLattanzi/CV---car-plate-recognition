import torch
from torch.utils.data import DataLoader

import network
import data

###### Model variables ########

batch_size = 32
num_classes = 8
num_epochs = 16

###############################

dataset = data.CarPlateDataset("../CCPD2019")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
image, label = dataset[5]
print(label)

model = network.create_model(num_classes)
# checking if gpu is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


print('//  starting training  //')

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    pass


