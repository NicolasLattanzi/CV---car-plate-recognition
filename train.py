import torch
from torch.utils.data import DataLoader

import network
import data

###### hyper parameters ########

batch_size = 32
num_epochs = 4
learning_rate = 0.001

###############################

dataset = data.CarPlateDataset("../CCPD2019")
train_dataset, test_dataset = data.train_test_split(dataset)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_size = len(trainloader)
test_size = len(test_dataset)

model = network.create_model()
# checking if gpu is available, otherwise cpu is used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


print('//  starting training  //')

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    print(f'###\t\t  starting epoch n.{epoch+1}  \t\t###\n')
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        # forward step
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # printing error every X batch
        if (i + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{train_size}], Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / train_size
    print(f"Epoch [{epoch+1}/{num_epochs}] training completed. Average Loss: {avg_train_loss:.4f}")


    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for (images, labels) in testLoader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / test_size
    print(f"Epoch [{epoch+1}/{num_epochs}] test completed. Average Loss: {avg_test_loss:.4f}\n")



