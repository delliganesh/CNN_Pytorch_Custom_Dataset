import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from dataloder import custom_dataset
from model import Net
import torch.optim as optim
import matplotlib.pyplot as plt
from parameters import train_data_path, Batch_Size, Epoch,LEARNING_RATE,MODEL_NAME
print(train_data_path)

train_data = custom_dataset(train_data_path)
print("Loaded of Train data",len(train_data))

train_loader = DataLoader(dataset=train_data, batch_size=30, shuffle=True)

net = Net()
print(net)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE)

loss_history = []

def train(epoch):
    running_loss =0.0
    n_batches = len(train_data)//Batch_Size
    for i , data in enumerate(train_loader,0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        la =torch.from_numpy(np.array(labels))
        loss = criterion(outputs,la)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i % n_batches == 0 and i != 0:
            loss = running_loss/n_batches
            loss_history.append(loss)
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/n_batches))
            running_loss=0.0

for epoch in range(Epoch):
    train(epoch)

torch.save(net, 'models/{}.pt'.format(MODEL_NAME))
print("Saved model...")

# Plotting loss vs number of epochs
#print(np.array(range(Epoch),len(loss_history)))
#plt.plot(np.array(range(Epoch), loss_history))
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.show()

