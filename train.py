import torch
from torch.utils.data import DataLoader

import network
import data
import utils

###### hyper parameters ########

batch_size = 32
num_epochs = 4
learning_rate = 0.001

###############################

dataset = data.create_dataset("../CCPD2019")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = network.create_model()
# checking if gpu is available, otherwise cpu is used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


print('//  starting training  //')

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train(True)
    train_loss = 0.0
    print(f'###\t\t  starting epoch n.{epoch}  \t\t###\n')
    for i, (images, labels) in enumerate(dataloader):
        plate_positions = []
        for x in labels:
            plate_positions.append( utils.vertices_from_image_path(x) )

        images = images.to(device)
        #labels = labels.to(device)
        #labels = labels.float().to(device)

        # forward step
        outputs = model(images)
        labels_tensor = torch.tensor(plate_positions, dtype=torch.float32) # MSE accepts only float32
        #print("outputs shape:", outputs.shape)
        #print("labels shape:", labels_tensor.shape)
        loss = loss_function(outputs, labels_tensor)

        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # printing error every X batch
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = train_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")

model.train(False)

