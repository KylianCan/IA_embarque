import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T

# --- Définition du modèle ---
class MLP(nn.Module):
    def __init__(self, H, input_size):
        super(MLP, self).__init__()
        
        self.C = 10  # Nombre de classes
        self.D = input_size  # Taille d'entrée
        self.H = H  # Taille de la couche cachée
        
        # Définitions des couches
        self.fc1 = nn.Linear(self.D, self.H)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.H, self.C)
        
    def forward(self, X):
        X1 = self.fc1(X)  # Couche linéaire
        X2 = self.relu(X1)  # Activation ReLU
        O = self.fc2(X2)  # Sortie finale
        return O


# --- Chargement des données ---
def load_data(data_dir, input_size):
    data = []
    labels = []

    for digit in range(10):
        for i in range(10):  
            image_path = os.path.join(data_dir, f"{digit}_{i}.bmp")
            if os.path.exists(image_path):  # Vérifier que le fichier existe
                image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
                image = image.resize((28, 28))  # Redimensionner à 28x28
                image = np.array(image).flatten() / 255.0  # Normalisation
                data.append(image)
                labels.append(digit)
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return data, labels


# --- Paramètres ---
data_dir = "/home/docker/Work/customized_data"
input_size = 28 * 28  # Taille des images aplaties
H = 30  # Taille de la couche cachée
lr = 1e-2  # Taux d'apprentissage
beta = 0.9  # Paramètre de momentum
n_epoch = 20  # Nombre d'époques

# Charger les données
data, labels = load_data(data_dir, input_size)
# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

plt.figure(1)
for i in range(4):
    image = X_train[i]
    label = y_train[i]
    plt.subplot(1,4,i+1)
    plt.imshow(T.ToPILImage()(image))
    plt.title('True label {}'.format(label))
    
plt.pause(1.)




# Convertir en tenseurs PyTorch
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# --- Instanciation du modèle ---
model = MLP(H, input_size)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=beta)
criterion = nn.CrossEntropyLoss()

# --- Entraînement ---
for epoch in range(n_epoch):
    model.train()
    optimizer.zero_grad()
    
    # Forward
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{n_epoch}], Loss: {loss.item():.4f}")

# --- Évaluation ---
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Précision sur les données de test : {accuracy:.4f}")
