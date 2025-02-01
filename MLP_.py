import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd


data_dir = "/home/docker/Work/dataProcessed"
# data_dir = "/home/docker/Work/customized_data"

data = []
labels = []
imagess = []

for digit in range(10):
    print(f"Classe : {digit}")
    class_images = []
    for i in range(10):  
        image_path = os.path.join(data_dir, f"{digit}_{i}.bmp")
        image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
        image = image.resize((28, 28))  # Redimensionner à 28x28
        image2 = np.array(image) / 255.0  # Normalisation
        class_images.append(image2)
        imagess.append(image)
    
    # Ajouter les données et les labels
    data.extend(class_images[:8])  # Les 8 premières images pour l'entraînement
    labels.extend([digit] * 8)
    data.extend(class_images[8:])  # Les 2 dernières pour le test
    labels.extend([digit] * 2)

print("Labels :", labels)
print('Shape des images :', data[0].shape)

# Convertir les listes en tableaux NumPy
data = np.array(data)
labels = np.array(labels)

# Diviser les données manuellement en train/test
X_train, X_test = [], []
y_train, y_test = [], []

for digit in range(10):
    digit_indices = np.where(labels == digit)[0]
    X_train.extend(data[digit_indices[:8]])
    X_test.extend(data[digit_indices[8:]])
    y_train.extend(labels[digit_indices[:8]])
    y_test.extend(labels[digit_indices[8:]])

# Convertir en tableaux NumPy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Afficher les shapes des ensembles
print(f"Ensemble d'entraînement : {X_train.shape}, {y_train.shape}")
print(f"Ensemble de test : {X_test.shape}, {y_test.shape}")

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation='relu'),  # Réduire le nombre de neurones
    #layers.Dropout(0.5),                   # Ajouter un dropout
    layers.Dense(10, activation='softmax')
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Précision sur les données de test :', test_acc)

prediction=model.predict(data)

print(prediction[0])


weights = model.get_weights()
W1=weights[0]
print(np.shape(W1))
B1=weights[1]
print(np.shape(B1))
W2=weights[2]
print(np.shape(W2))
B2=weights[3]
print(np.shape(B2))


output_dir = "poids_MLP_Processeddata"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Sauvegarde des poids
with open(os.path.join(output_dir, "W1.txt"), "w") as file:    
    for w in W1:
        np.savetxt(file, w, fmt='%0.8f', delimiter='\n ')

with open(os.path.join(output_dir, "B1.txt"), "w") as file:    
    np.savetxt(file, B1, fmt='%0.8f', delimiter='\n ')

with open(os.path.join(output_dir, "W2.txt"), "w") as file:    
    for w in W2:
        np.savetxt(file, w, fmt='%0.8f', delimiter='\n ')

with open(os.path.join(output_dir, "B2.txt"), "w") as file:    
    np.savetxt(file, B2, fmt='%0.8f', delimiter='\n ')







