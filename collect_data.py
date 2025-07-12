#!/usr/bin/env python3
"""
Skript na zber dát pre trénovanie modelu rozpoznávania giest
Fáza 2 - Úloha 1: Vytvorenie skriptu na zber dát
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
from pathlib import Path
import logging

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GestureDataCollector:
    """Trieda na zber dát pre trénovanie giest"""
    
    def __init__(self, data_path="gesture_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # Inicializácia MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Konfigurácia zberu dát
        self.sequence_length = 30  # Počet snímkov v sekvencii
        self.num_landmarks = 21   # Počet kľúčových bodov ruky
        self.coords_per_landmark = 3  # x, y, z súradnice
        
        # Predvolené gestá (kompatibilné s train_model.py)
        self.available_gestures = [
            'pest',           # Zatvorená päsť
            'otvorena_dlan',  # Otvorená dlaň
            'palec_hore',     # Thumbs up
            'ukazovak',       # Ukazovák
        ]
        
        # Načítanie existujúcich dát
        self.load_metadata()
        
    def load_metadata(self):
        """Načíta metadata o existujúcich dátach"""
        metadata_file = self.data_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'gestures': {},
                'total_sequences': 0,
                'last_updated': None
            }
    
    def save_metadata(self):
        """Uloží metadata"""
        self.metadata['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        metadata_file = self.data_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def extract_landmarks(self, results):
        """Extraktuje kľúčové body z MediaPipe výsledkov"""
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        else:
            # Ak nie sú detekované ruky, vráti nuly
            return np.zeros(self.num_landmarks * self.coords_per_landmark)
    
    def create_gesture_directory(self, gesture_name):
        """Vytvorí adresár pre gesto"""
        gesture_dir = self.data_path / gesture_name
        gesture_dir.mkdir(exist_ok=True)
        return gesture_dir
    
    def get_next_sequence_number(self, gesture_name):
        """Získa číslo ďalšej sekvencie pre dané gesto"""
        gesture_dir = self.data_path / gesture_name
        if not gesture_dir.exists():
            return 0
        
        existing_sequences = [int(d.name) for d in gesture_dir.iterdir() 
                            if d.is_dir() and d.name.isdigit()]
        return max(existing_sequences, default=-1) + 1
    
    def collect_gesture_sequence(self, gesture_name, sequence_number):
        """Zbiera jednu sekvenciu pre dané gesto"""
        print(f"\n=== ZBER SEKVENCIE {sequence_number} PRE GESTO '{gesture_name.upper()}' ===")
        
        # Vytvorenie adresára pre sekvenciu
        sequence_dir = self.data_path / gesture_name / str(sequence_number)
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializácia kamery
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Nemožno otvoriť kameru")
            return False
        
        collected_frames = []
        frame_count = 0
        recording = False
        
        print("Inštrukcie:")
        print("- Stlačte SPACE pre začatie nahrávania")
        print("- Udržujte gesto počas nahrávania")
        print("- Stlačte 'q' pre ukončenie")
        print("- Stlačte 'r' pre reštart sekvencie")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Zrkadlenie obrazu
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detekcia rúk
            results = self.hands.process(rgb_frame)
            
            # Kreslenie kľúčových bodov
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Zobrazenie stavu
            status_color = (0, 255, 0) if recording else (0, 0, 255)
            status_text = f"NAHRAVAM {frame_count}/{self.sequence_length}" if recording else "STLAC SPACE PRE START"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.putText(frame, f"Gesto: {gesture_name.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Sekvencia: {sequence_number}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Nahrávanie
            if recording:
                landmarks = self.extract_landmarks(results)
                collected_frames.append(landmarks)
                frame_count += 1
                
                if frame_count >= self.sequence_length:
                    # Uloženie sekvencie
                    for i, landmarks in enumerate(collected_frames):
                        np.save(sequence_dir / f"{i}.npy", landmarks)
                    
                    print(f"✅ Sekvencia {sequence_number} uložená!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
            
            cv2.imshow('Zber dát pre gestá', frame)
            
            # Spracovanie klávesov
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording:
                recording = True
                collected_frames = []
                frame_count = 0
                print("🔴 Začínam nahrávanie...")
            elif key == ord('r'):
                recording = False
                collected_frames = []
                frame_count = 0
                print("🔄 Reštart sekvencie")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def collect_gesture_data(self, gesture_name, num_sequences=30):
        """Zbiera dáta pre konkrétne gesto"""
        if gesture_name not in self.available_gestures:
            print(f"⚠️ Gesto '{gesture_name}' nie je v zozname dostupných giest.")
            print(f"Dostupné gestá: {', '.join(self.available_gestures)}")
            return False
        
        print(f"\n🎯 ZBER DÁT PRE GESTO: {gesture_name.upper()}")
        print(f"Cieľ: {num_sequences} sekvencií")
        
        # Vytvorenie adresára
        gesture_dir = self.create_gesture_directory(gesture_name)
        
        # Získanie počtu existujúcich sekvencií
        start_sequence = self.get_next_sequence_number(gesture_name)
        existing_count = start_sequence
        
        if existing_count > 0:
            print(f"📁 Nájdených {existing_count} existujúcich sekvencií")
            response = input(f"Pokračovať od sekvencie {start_sequence}? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Zber sekvencií
        successful_sequences = 0
        for i in range(start_sequence, start_sequence + num_sequences):
            print(f"\n--- Sekvencia {i+1}/{start_sequence + num_sequences} ---")
            
            if self.collect_gesture_sequence(gesture_name, i):
                successful_sequences += 1
            else:
                print("❌ Sekvencia nebola uložená")
                break
        
        # Aktualizácia metadát
        if gesture_name not in self.metadata['gestures']:
            self.metadata['gestures'][gesture_name] = 0
        
        self.metadata['gestures'][gesture_name] += successful_sequences
        self.metadata['total_sequences'] += successful_sequences
        self.save_metadata()
        
        print(f"\n✅ Úspešne zozbierané {successful_sequences} sekvencií pre gesto '{gesture_name}'")
        return True
    
    def show_statistics(self):
        """Zobrazí statistiky zozbieraných dát"""
        print("\n" + "="*60)
        print("ŠTATISTIKY ZOZBIERANÝCH DÁT")
        print("="*60)
        
        if not self.metadata['gestures']:
            print("❌ Žiadne dáta nie sú zozbierané")
            return
        
        total_sequences = 0
        for gesture, count in self.metadata['gestures'].items():
            print(f"{gesture:15} {count:3d} sekvencií")
            total_sequences += count
        
        print("-" * 30)
        print(f"{'CELKOM':15} {total_sequences:3d} sekvencií")
        
        if self.metadata['last_updated']:
            print(f"\nPosledná aktualizácia: {self.metadata['last_updated']}")
    
    def interactive_menu(self):
        """Interaktívne menu pre zber dát"""
        while True:
            print("\n" + "="*60)
            print("ZBER DÁT PRE TRÉNOVANIE GIEST")
            print("="*60)
            print("1. Zobraziť dostupné gestá")
            print("2. Zobraziť štatistiky")
            print("3. Zbierať dáta pre gesto")
            print("4. Zbierať dáta pre všetky gestá")
            print("5. Ukončiť")
            
            choice = input("\nVyberte možnosť (1-5): ").strip()
            
            if choice == '1':
                print("\nDostupné gestá:")
                for i, gesture in enumerate(self.available_gestures, 1):
                    print(f"{i:2d}. {gesture}")
            
            elif choice == '2':
                self.show_statistics()
            
            elif choice == '3':
                print("\nDostupné gestá:")
                for i, gesture in enumerate(self.available_gestures, 1):
                    print(f"{i:2d}. {gesture}")
                
                try:
                    gesture_idx = int(input("\nVyberte gesto (číslo): ")) - 1
                    if 0 <= gesture_idx < len(self.available_gestures):
                        gesture_name = self.available_gestures[gesture_idx]
                        num_sequences = int(input(f"Počet sekvencií pre '{gesture_name}' (predvolené 30): ") or "30")
                        self.collect_gesture_data(gesture_name, num_sequences)
                    else:
                        print("❌ Neplatné číslo gesta")
                except ValueError:
                    print("❌ Neplatný vstup")
            
            elif choice == '4':
                num_sequences = int(input("Počet sekvencií pre každé gesto (predvolené 30): ") or "30")
                for gesture in self.available_gestures:
                    print(f"\n🎯 Zbieranie dát pre gesto: {gesture}")
                    if not self.collect_gesture_data(gesture, num_sequences):
                        break
            
            elif choice == '5':
                print("👋 Ukončujem zber dát")
                break
            
            else:
                print("❌ Neplatná voľba")

def main():
    """Hlavná funkcia"""
    print("ZBER DÁT PRE TRÉNOVANIE MODELU GIEST")
    print("Fáza 2 - Úloha 1")
    
    collector = GestureDataCollector()
    collector.interactive_menu()

if __name__ == "__main__":
    main()
