import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Aqui carregaremos o arquivo CSV que contém os clusters / Here we will load the CSV file that contains the clusters.
df = pd.read_csv("cluster.csv")
df = df.dropna()

# Separaremos os atributos e alvo / We will separate the attributes and target.
X = df.drop(columns=["cluster"]).values
y = df["cluster"].values

# Normalizamos oa dados / We will normalize the data.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividimos os dados em treino e teste / We will split the data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convertemos para tensores PyTorch / We will convert to PyTorch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Criamos o DataLoader / We will create the DataLoader.
# O DataLoader é uma classe PyTorch que nos ajuda a iterar no conjunto de dados em lotes / The DataLoader is a PyTorch class that helps us iterate over the dataset in batches.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Definimos o modelo / We will define the model.
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# vamos criar a instância do modelo / We will create the model instance.
num_classes = len(np.unique(y))
model = NeuralNet(input_size=X_train.shape[1], num_classes=num_classes)

# Configurar loss e optimizer / We will set up loss and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento / Training.
num_epochs = 50
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Época [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# E o teste / And the test.
# Vamos fazer a previsão no conjunto de teste e calcular a acurácia / We will make the prediction on the test set and calculate the accuracy.
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred = torch.max(outputs, 1)
    acc = accuracy_score(y_test, y_pred.numpy())

print(f"Acurácia: {acc:.4f}")


# Salvar os pesos do modelo treinado
torch.save(model.state_dict(), "modelo_credito.pth")
# Salvar o scaler normalmente
joblib.dump(scaler, "scaler.pkl")