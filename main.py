# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:22:44 2025

@author: caniv
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([28,28])])


device = "cpu"

batch_size = 4




train_dataset = datasets.ImageFolder("./MeryemProcessed", transform=transformation)
test_dataset = datasets.ImageFolder("./KylianProcessed", transform=transformation)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    
#%%

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64,10),
        )

    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.softmax(x, dim= 1)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

MLP = MLP().to(device)
CNN = CNN(10).to(device)

cout = nn.CrossEntropyLoss()
optimizer_MLP = torch.optim.Adam(MLP.parameters())
optimizer_CNN = torch.optim.Adam(CNN.parameters())

def train(data_loader, model, cout, optimizer) :
    model.train()
    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)
    total_loss = 0
    total_correct = 0
    for data, label in data_loader:
        #En cas d’utilisation d’un GPU
        data = data.to(device)
        label = label.to(device)
        # Feed−forward
        output = model(data)
        # Calcul du cout
        loss = cout(output, label)
        total_loss += loss
        # Nombre de bonnes classifications
        predictions = output.argmax(1)
        correct = (predictions == label ).type(torch.float)
        total_correct += correct.sum().item()
        # Retro−propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad ()
    train_loss = total_loss/num_batches
    acc = total_correct/num_items
    print (f"Average loss: {train_loss:7f}, precision: {acc:.2%}")

epochs = 10

for epoch in range(epochs):
    print("Training epoch: " + str(epoch+1))
    print("MLP:")
    train(train_loader, MLP, cout, optimizer_MLP)
    print("CNN:")
    train(train_loader, CNN, cout, optimizer_CNN)

#%%
def test(test_loader, model, cout):
    print("Test")
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, label in test_loader:
            # En cas d'utilisation d'un GPU
            data = data.to(device)
            label = label.to(device)
        
            # Feed-forward
            output = model(data)
        
            # Calcul du cout
            loss = cout(output, label)
            test_loss += loss.item()
        
            # Nombre de bonnes classifications
            predictions = output.argmax(1)
            correct = (predictions == label).type(torch.float)
            total_correct += correct.sum().item()

    test_loss = test_loss/num_batches
    acc = total_correct/num_items

    print(f"Precision: {100*acc:>0.1f}%, average loss: {test_loss:>7f}")

print("MLP")
test(test_loader, MLP, cout)
print("CNN")
test(test_loader, CNN, cout)

#%%

import json

def export_weights(model, filename):
    weights = {}
    for name, param in model.state_dict().items():
        weights[name] = param.tolist()
    
    with open(filename, 'w') as f:
        json.dump(weights, f)
        
export_weights(MLP, 'MLP_weights.json')
export_weights(CNN, 'CNN_weights.json')




