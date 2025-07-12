#!/usr/bin/env python3
"""
Rozšírený skript na trénovanie modelu rozpoznávania giest
Fáza 3 - Úloha 1: Vytvorenie skriptu na trénovanie
"""

import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GestureTrainer:
    """Trieda na trénovanie modelu rozpoznávania giest"""
    
    def __init__(self, data_path="gesture_data", model_path="gesture_model.pth"):
        self.data_path = Path(data_path)
        self.model_path = model_path
        
        # Konfigurácia
        self.sequence_length = 30
        self.input_size = 21 * 3  # 21 landmarks * 3 súradnice
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.3
        self.learning_rate = 0.001
        self.batch_size = 16
        self.num_epochs = 100
        self.patience = 15  # Pre early stopping
        
        # Zariadenie (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Používam zariadenie: {self.device}")
        
        # Gestá a dáta
        self.gestures = []
        self.X = None
        self.y = None
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def load_data(self):
        """Načíta dáta z gesture_data adresára"""
        logger.info("Načítavam dáta...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Adresár {self.data_path} neexistuje")
        
        # Načítanie metadát
        metadata_file = self.data_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.gestures = list(metadata['gestures'].keys())
            logger.info(f"Načítané gestá z metadát: {self.gestures}")
        else:
            # Fallback - načítanie z adresárov
            self.gestures = [d.name for d in self.data_path.iterdir() if d.is_dir()]
            logger.warning("Metadata súbor neexistuje, používam adresáre")
        
        if not self.gestures:
            raise ValueError("Žiadne gestá neboli nájdené")
        
        # Načítanie sekvencií
        sequences = []
        labels = []
        
        for gesture_idx, gesture in enumerate(self.gestures):
            gesture_dir = self.data_path / gesture
            if not gesture_dir.exists():
                logger.warning(f"Adresár pre gesto '{gesture}' neexistuje")
                continue
            
            sequence_dirs = [d for d in gesture_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            logger.info(f"Gesto '{gesture}': {len(sequence_dirs)} sekvencií")
            
            for seq_dir in sequence_dirs:
                try:
                    sequence = self.load_sequence(seq_dir)
                    if sequence is not None:
                        sequences.append(sequence)
                        labels.append(gesture_idx)
                except Exception as e:
                    logger.warning(f"Chyba pri načítaní sekvencie {seq_dir}: {e}")
        
        if not sequences:
            raise ValueError("Žiadne platné sekvencie neboli načítané")
        
        self.X = np.array(sequences)
        self.y = np.array(labels)
        
        logger.info(f"Dáta načítané: {self.X.shape} sekvencií, {len(self.gestures)} giest")
        logger.info(f"Rozdelenie giest: {dict(zip(self.gestures, np.bincount(self.y)))}")
        
    def load_sequence(self, sequence_dir):
        """Načíta jednu sekvenciu z adresára"""
        frames = []
        
        for frame_idx in range(self.sequence_length):
            frame_file = sequence_dir / f"{frame_idx}.npy"
            if frame_file.exists():
                frame_data = np.load(frame_file)
                if len(frame_data) == self.input_size:
                    frames.append(frame_data)
                else:
                    logger.warning(f"Nesprávna veľkosť frame {frame_file}: {len(frame_data)}")
                    return None
            else:
                logger.warning(f"Chýba frame {frame_file}")
                return None
        
        return np.array(frames)
    
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """Pripraví dáta pre trénovanie"""
        if self.X is None or self.y is None:
            raise ValueError("Dáta nie sú načítané. Spustite load_data() najprv.")
        
        # Rozdelenie na train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Rozdelenie train na train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        # Konverzia na PyTorch tensory
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        
        # Vytvorenie DataLoaderov
        self.train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size, shuffle=False
        )
        
        logger.info(f"Dáta pripravené - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")

    def train_model(self):
        """Trénuje model s pokročilými funkciami"""
        logger.info("Začínam trénovanie modelu...")

        # Inicializácia modelu
        self.model = GestureLSTM(
            self.input_size, self.hidden_size, len(self.gestures),
            self.num_layers, self.dropout
        ).to(self.device)

        # Loss a optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Trénovanie
            train_loss, train_acc = self._train_epoch(criterion, optimizer)

            # Validácia
            val_loss, val_acc = self._validate_epoch(criterion)

            # Scheduler krok
            scheduler.step(val_loss)

            # Uloženie metrík
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Uloženie najlepšieho modelu
                torch.save(self.model.state_dict(), f"best_{self.model_path}")
            else:
                patience_counter += 1

            # Výpis pokroku
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epocha [{epoch+1}/{self.num_epochs}]')
                logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping po {epoch+1} epochách")
                break

        end_time = time.time()
        logger.info(f"Trénovanie dokončené za {end_time - start_time:.2f} sekúnd")

        # Načítanie najlepšieho modelu
        self.model.load_state_dict(torch.load(f"best_{self.model_path}", weights_only=True))

    def _train_epoch(self, criterion, optimizer):
        """Jeden trénovací epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for sequences, labels in self.train_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, criterion):
        """Jeden validačný epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def test_model(self):
        """Testuje model na testovacích dátach"""
        logger.info("Testovanie modelu...")

        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        logger.info(f'Presnosť na testovacích dátach: {accuracy:.2f}%')

        # Detailný report
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(
            all_labels, all_predictions,
            target_names=self.gestures, zero_division=0
        )
        logger.info(f"Classification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm)

        return accuracy

    def save_model(self):
        """Uloží model a metadata"""
        # Uloženie modelu
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model uložený do: {self.model_path}")

        # Uloženie metadát modelu
        model_info = {
            'gestures': self.gestures,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': len(self.gestures),
            'sequence_length': self.sequence_length,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0
        }

        with open(self.model_path.replace('.pth', '_info.json'), 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        logger.info("Metadata modelu uložené")

    def plot_training_history(self):
        """Zobrazí grafy trénovania"""
        if not self.train_losses:
            logger.warning("Žiadne trénovacie dáta na zobrazenie")
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss graf
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy graf
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Graf trénovania uložený ako training_history.png")

    def plot_confusion_matrix(self, cm):
        """Zobrazí confusion matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.gestures, yticklabels=self.gestures)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Confusion matrix uložená ako confusion_matrix.png")

class GestureLSTM(nn.Module):
    """Pokročilá LSTM architektúra pre rozpoznávanie giest"""
    
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.3):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM vrstvy
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout a batch normalization
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Plne prepojené vrstvy
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Inicializácia skrytých stavov
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Posledný výstup
        out = out[:, -1, :]
        
        # Batch normalization a dropout
        out = self.batch_norm(out)
        out = self.dropout(out)
        
        # Plne prepojené vrstvy
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def main():
    """Hlavná funkcia trénovania"""
    print("TRÉNOVANIE MODELU ROZPOZNÁVANIA GIEST")
    print("Fáza 3 - Úloha 1")

    try:
        # Inicializácia trainera
        trainer = GestureTrainer()

        # Načítanie dát
        trainer.load_data()

        # Príprava dát
        trainer.prepare_data()

        # Trénovanie
        trainer.train_model()

        # Testovanie
        trainer.test_model()

        # Uloženie modelu
        trainer.save_model()

        # Vizualizácia výsledkov
        trainer.plot_training_history()

        logger.info("Trénovanie úspešne dokončené!")

    except Exception as e:
        logger.error(f"Chyba počas trénovania: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
