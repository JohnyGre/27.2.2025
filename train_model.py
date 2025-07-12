import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# --- Konfigurácia ---
DATA_PATH = "gesture_data"
MODEL_PATH = "gesture_model.pth"
gestures = np.array(['pest', 'otvorena_dlan', 'palec_hore', 'ukazovak'])
num_sequences = 30
sequence_length = 30
# Parametre pre neurónovú sieť a trénovanie
input_size = 21 * 3  # 21 landmarkov * 3 súradnice (x, y, z)
hidden_size = 128
num_classes = len(gestures)
learning_rate = 0.001
batch_size = 16
num_epochs = 100

# --- 1. Načítanie a príprava dát ---
print("Načítavam dáta...")
sequences, labels = [], []
for i, gesture in enumerate(gestures):
    for sequence in range(num_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, gesture, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(i)

X = np.array(sequences)
y = np.array(labels)

print(f"Dáta načítané. Tvar X: {X.shape}, Tvar y: {y.shape}")

# Rozdelenie dát na trénovacie a testovacie
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Prevod na PyTorch tenzory
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Vytvorenie DataLoaderov
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# --- 2. Definícia modelu (Neurónovej siete) ---
# Použijeme LSTM (Long Short-Term Memory) sieť, ktorá je vhodná na spracovanie sekvencií
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Inicializácia skrytých stavov
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM vrstva
        out, _ = self.lstm(x, (h0, c0))
        
        # Zoberieme výstup z posledného časového kroku a pošleme ho do plne prepojenej vrstvy
        out = self.fc(out[:, -1, :])
        return out

# --- 3. Trénovanie modelu ---
print("\nZačínam trénovanie modelu...")
# Overenie dostupnosti CUDA a nastavenie zariadenia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Trénovanie bude prebiehať na zariadení: {device}")

model = GestureLSTM(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    for i, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Vypísanie stavu po každej 10. epoche
    if (epoch + 1) % 10 == 0:
        print(f'Epocha [{epoch+1}/{num_epochs}], Strata (Loss): {loss.item():.4f}')

end_time = time.time()
print(f"\nTrénovanie dokončené za {end_time - start_time:.2f} sekúnd.")

# --- 4. Testovanie modelu ---
print("Testujem presnosť modelu na testovacích dátach...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Presnosť modelu na testovacích dátach: {accuracy:.2f} %')

# --- 5. Uloženie modelu ---
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel úspešne uložený do súboru: {MODEL_PATH}")

# Informácia o ďalšom kroku
print("\nĎalší krok: Integrácia modelu do `gesture_recognition.py` na rozpoznávanie v reálnom čase.")
