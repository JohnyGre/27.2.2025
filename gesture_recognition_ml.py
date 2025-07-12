#!/usr/bin/env python3
"""
Rozšírené rozpoznávanie giest s integráciou natrénovaného ML modelu
Fáza 4 - Úloha 1: Úprava gesture_recognition.py
"""

import mediapipe as mp
import cv2
import pyautogui
from queue import Queue
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from collections import deque
from multiprocessing import Process

logger = logging.getLogger(__name__)

class GestureLSTM(nn.Module):
    """LSTM model pre rozpoznávanie giest (kópia z train_model_enhanced.py)"""
    
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

class MLGestureRecognizer:
    """Rozšírený rozpoznávač giest s ML modelom"""
    
    def __init__(self, output_queue: Queue, model_path="gesture_model.pth", use_ml=True):
        self.output_queue = output_queue
        self.use_ml = use_ml
        
        # MediaPipe inicializácia
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        
        # Kamera inicializácia
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Nemôžem otvoriť kameru")
        
        # Stav
        self.running = True
        self.last_gesture_time = time.time()
        
        # ML model inicializácia
        self.model = None
        self.gestures = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 30
        self.input_size = 21 * 3  # 21 landmarks * 3 súradnice
        
        # Buffer pre sekvencie
        self.landmarks_buffer = deque(maxlen=self.sequence_length)
        
        # Confidence threshold
        self.confidence_threshold = 0.7
        
        # Načítanie ML modelu
        if self.use_ml:
            self.load_ml_model(model_path)
        
        logger.info(f"MLGestureRecognizer inicializovaný (ML: {self.use_ml}, Device: {self.device})")
    
    def load_ml_model(self, model_path):
        """Načíta natrénovaný ML model"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.warning(f"Model {model_path} neexistuje, používam rule-based rozpoznávanie")
                self.use_ml = False
                return
            
            # Načítanie metadát modelu
            info_path = model_path.with_suffix('').with_suffix('_info.json')
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                self.gestures = model_info['gestures']
                hidden_size = model_info.get('hidden_size', 128)
                num_layers = model_info.get('num_layers', 2)
                num_classes = len(self.gestures)
                
                logger.info(f"Načítané gestá: {self.gestures}")
            else:
                # Fallback hodnoty
                self.gestures = ['pest', 'otvorena_dlan', 'palec_hore', 'ukazovak']
                hidden_size = 128
                num_layers = 2
                num_classes = 4
                logger.warning("Metadata modelu nenájdené, používam predvolené hodnoty")
            
            # Vytvorenie a načítanie modelu
            self.model = GestureLSTM(
                self.input_size, hidden_size, num_classes, num_layers
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            
            logger.info(f"ML model úspešne načítaný z {model_path}")
            
        except Exception as e:
            logger.error(f"Chyba pri načítaní ML modelu: {e}")
            self.use_ml = False
    
    def extract_landmarks(self, hand_landmarks):
        """Extraktuje landmarks z MediaPipe výsledkov"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def predict_gesture_ml(self):
        """Predikcia gesta pomocou ML modelu"""
        if not self.use_ml or self.model is None or len(self.landmarks_buffer) < self.sequence_length:
            return None, 0.0
        
        try:
            # Príprava sekvencie
            sequence = np.array(list(self.landmarks_buffer))
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Predikcia
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
                
                if confidence >= self.confidence_threshold:
                    gesture = self.gestures[predicted_idx]
                    return gesture, confidence
                else:
                    return None, confidence
                    
        except Exception as e:
            logger.error(f"Chyba pri ML predikcii: {e}")
            return None, 0.0
    
    def recognize_gesture_rule_based(self, hand_landmarks):
        """Rule-based rozpoznávanie giest (pôvodná logika)"""
        # Kópia pôvodnej logiky z gesture_recognition.py
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # THUMBS UP
        if thumb_tip.y < index_finger_tip.y and \
           index_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
           middle_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
           ring_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y and \
           pinky_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y:
            return "palec_hore"
        
        # FIST
        if self.is_fist(hand_landmarks):
            return "pest"
        
        # PEACE
        if self.is_peace_gesture(hand_landmarks):
            return "peace"
        
        # POINTING
        if self.is_pointing_gesture(hand_landmarks):
            return "ukazovak"
        
        # OPEN PALM (default ak nie sú prsty ohnuté)
        if not self.is_fist(hand_landmarks):
            return "otvorena_dlan"
        
        return None
    
    def is_fist(self, hand_landmarks):
        """Detekcia päste"""
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        return index_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
               middle_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
               ring_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y and \
               pinky_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y
    
    def is_peace_gesture(self, hand_landmarks):
        """Detekcia peace gesta"""
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        return index_finger_tip.y < index_finger_mcp.y and \
               middle_finger_tip.y < middle_finger_mcp.y and \
               ring_finger_tip.y > ring_finger_mcp.y and \
               pinky_finger_tip.y > pinky_finger_mcp.y
    
    def is_pointing_gesture(self, hand_landmarks):
        """Detekcia pointing gesta"""
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        return index_finger_tip.y < index_finger_mcp.y and \
               middle_finger_tip.y > middle_finger_mcp.y and \
               ring_finger_tip.y > ring_finger_mcp.y and \
               pinky_finger_tip.y > pinky_finger_mcp.y
    
    def is_gesture_rate_limited(self):
        """Rate limiting pre gestá"""
        return (time.time() - self.last_gesture_time) >= 0.5
    
    def run(self):
        """Hlavná slučka rozpoznávania"""
        try:
            while self.running:
                success, image = self.cap.read()
                if not success:
                    logger.error("Prázdny snímok z kamery")
                    continue
                
                # Spracovanie obrazu
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                gesture = None
                confidence = 0.0
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extrakcia landmarks
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.landmarks_buffer.append(landmarks)
                        
                        # ML predikcia ak je dostupná
                        if self.use_ml:
                            gesture, confidence = self.predict_gesture_ml()
                        
                        # Fallback na rule-based ak ML nefunguje
                        if gesture is None:
                            gesture = self.recognize_gesture_rule_based(hand_landmarks)
                            confidence = 1.0  # Rule-based má vždy plnú dôveru
                        
                        # Odoslanie gesta ak je rozpoznané a nie je rate limited
                        if gesture and self.is_gesture_rate_limited():
                            gesture_info = {
                                "gesture": gesture,
                                "confidence": confidence,
                                "method": "ML" if self.use_ml and confidence < 1.0 else "Rule-based"
                            }
                            self.output_queue.put(("gesture", gesture_info))
                            logger.info(f"Rozpoznané gesto: {gesture} (confidence: {confidence:.2f}, method: {gesture_info['method']})")
                            self.last_gesture_time = time.time()
                
                # Zobrazenie pre debugging (voliteľné)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Zobrazenie informácií na obraze
                cv2.putText(image, f"ML: {'ON' if self.use_ml else 'OFF'}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.use_ml else (0, 0, 255), 2)
                
                if gesture:
                    cv2.putText(image, f"Gesture: {gesture} ({confidence:.2f})", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('ML Gesture Recognition', image)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Chyba v hlavnej slučke: {e}", exc_info=True)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("MLGestureRecognizer zastavený")
    
    def stop(self):
        """Zastavenie rozpoznávania"""
        self.running = False

def ml_gesture_recognition_process(output_queue: Queue, model_path="gesture_model.pth", use_ml=True):
    """Thread funkcia pre ML rozpoznávanie giest"""
    try:
        recognizer = MLGestureRecognizer(output_queue, model_path, use_ml)
        recognizer.run()
    except IOError as e:
        logger.error(f"Chyba pri inicializácii MLGestureRecognizer: {e}")
    except Exception as e:
        logger.error(f"Neočakávaná chyba v ml_gesture_recognition_process: {e}", exc_info=True)

def main():
    """Hlavná funkcia pre testovanie"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("ML GESTURE RECOGNITION TEST")
    print("Stlačte 'q' v OpenCV okne pre ukončenie")
    
    q = Queue()
    
    # Test s ML modelom
    use_ml = Path("gesture_model.pth").exists()
    print(f"ML Model: {'Dostupný' if use_ml else 'Nedostupný - používam rule-based'}")
    
    p = Process(target=ml_gesture_recognition_process, args=(q, "gesture_model.pth", use_ml))
    p.start()
    
    try:
        while True:
            if not q.empty():
                gesture_type, gesture_info = q.get()
                if gesture_type == "gesture":
                    print(f"Rozpoznané: {gesture_info}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Ukončujem...")
    finally:
        p.terminate()
        p.join()
        print("Test ukončený")

if __name__ == "__main__":
    main()
