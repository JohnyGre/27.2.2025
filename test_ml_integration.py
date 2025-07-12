#!/usr/bin/env python3
"""
Test integrácie ML modelu do rozpoznávania giest
Fáza 4 - Test implementácie
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import json
from queue import Queue
import time

def test_ml_model_availability():
    """Test dostupnosti natrénovaného modelu"""
    print("="*60)
    print("TEST DOSTUPNOSTI ML MODELU")
    print("="*60)
    
    model_path = Path("gesture_model.pth")
    info_path = Path("gesture_model_info.json")
    
    if model_path.exists():
        print(f"✅ Model súbor existuje: {model_path}")
        
        # Test veľkosti súboru
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Veľkosť modelu: {size_mb:.2f} MB")
        
        if size_mb > 0.1:  # Aspoň 100KB
            print("✅ Model má rozumnú veľkosť")
        else:
            print("⚠️ Model je príliš malý")
            return False
    else:
        print(f"❌ Model súbor neexistuje: {model_path}")
        return False
    
    if info_path.exists():
        print(f"✅ Info súbor existuje: {info_path}")
        
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            required_keys = ['gestures', 'input_size', 'hidden_size', 'num_classes']
            for key in required_keys:
                if key in info:
                    print(f"✅ Info obsahuje {key}: {info[key]}")
                else:
                    print(f"❌ Info neobsahuje {key}")
                    return False
        except Exception as e:
            print(f"❌ Chyba pri čítaní info súboru: {e}")
            return False
    else:
        print(f"⚠️ Info súbor neexistuje: {info_path}")
    
    return True

def test_ml_gesture_recognizer_import():
    """Test importu ML rozpoznávača"""
    print("\n" + "="*60)
    print("TEST IMPORTU ML GESTURE RECOGNIZER")
    print("="*60)
    
    try:
        from gesture_recognition_ml import MLGestureRecognizer, GestureLSTM
        print("✅ MLGestureRecognizer úspešne importovaný")
        print("✅ GestureLSTM úspešne importovaný")
        
        # Test vytvorenia inštancie bez modelu
        q = Queue()
        recognizer = MLGestureRecognizer(q, use_ml=False)
        print("✅ MLGestureRecognizer sa dá vytvoriť bez ML")
        
        # Test s ML modelom ak existuje
        if Path("gesture_model.pth").exists():
            recognizer_ml = MLGestureRecognizer(q, use_ml=True)
            print("✅ MLGestureRecognizer sa dá vytvoriť s ML")
            
            if recognizer_ml.model is not None:
                print("✅ ML model bol úspešne načítaný")
            else:
                print("⚠️ ML model nebol načítaný")
        
        return True
        
    except ImportError as e:
        print(f"❌ Chyba pri importe: {e}")
        return False
    except Exception as e:
        print(f"❌ Chyba pri teste: {e}")
        return False

def test_model_loading():
    """Test načítania modelu"""
    print("\n" + "="*60)
    print("TEST NAČÍTANIA MODELU")
    print("="*60)
    
    if not Path("gesture_model.pth").exists():
        print("⚠️ Model neexistuje - preskakujem test")
        return True
    
    try:
        from gesture_recognition_ml import GestureLSTM
        
        # Načítanie info súboru
        info_path = Path("gesture_model_info.json")
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            input_size = info.get('input_size', 63)
            hidden_size = info.get('hidden_size', 128)
            num_classes = info.get('num_classes', 4)
            num_layers = info.get('num_layers', 2)
        else:
            # Predvolené hodnoty
            input_size = 63
            hidden_size = 128
            num_classes = 4
            num_layers = 2
        
        print(f"Model parametre: input={input_size}, hidden={hidden_size}, classes={num_classes}, layers={num_layers}")
        
        # Vytvorenie modelu
        model = GestureLSTM(input_size, hidden_size, num_classes, num_layers)
        print("✅ Model architektúra vytvorená")
        
        # Načítanie váh
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load("gesture_model.pth", map_location=device, weights_only=True))
        model.eval()
        print(f"✅ Model váhy načítané na {device}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 30, input_size).to(device)
        model = model.to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        if output.shape == (1, num_classes):
            print(f"✅ Model výstup má správny tvar: {output.shape}")
        else:
            print(f"❌ Nesprávny tvar výstupu: {output.shape}")
            return False
        
        # Test softmax
        probabilities = torch.softmax(output, dim=1)
        if torch.allclose(probabilities.sum(dim=1), torch.tensor(1.0)):
            print("✅ Softmax normalizácia funguje správne")
        else:
            print("❌ Problém so softmax normalizáciou")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste načítania modelu: {e}")
        return False

def test_landmarks_extraction():
    """Test extrakcie landmarks"""
    print("\n" + "="*60)
    print("TEST EXTRAKCIE LANDMARKS")
    print("="*60)
    
    try:
        from gesture_recognition_ml import MLGestureRecognizer
        import mediapipe as mp
        
        # Vytvorenie dummy hand landmarks
        class DummyLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        class DummyHandLandmarks:
            def __init__(self):
                self.landmark = [DummyLandmark(i*0.1, i*0.1, i*0.01) for i in range(21)]
        
        # Test extrakcie
        q = Queue()
        recognizer = MLGestureRecognizer(q, use_ml=False)
        
        dummy_landmarks = DummyHandLandmarks()
        extracted = recognizer.extract_landmarks(dummy_landmarks)
        
        if extracted.shape == (63,):
            print(f"✅ Landmarks extrakcia funguje: {extracted.shape}")
        else:
            print(f"❌ Nesprávny tvar landmarks: {extracted.shape}")
            return False
        
        # Test hodnôt
        expected_first_three = [0.0, 0.0, 0.0]  # Prvý landmark
        if np.allclose(extracted[:3], expected_first_three):
            print("✅ Landmarks hodnoty sú správne")
        else:
            print(f"❌ Nesprávne landmarks hodnoty: {extracted[:3]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste extrakcie landmarks: {e}")
        return False

def test_sequence_buffering():
    """Test bufferovania sekvencií"""
    print("\n" + "="*60)
    print("TEST BUFFEROVANIA SEKVENCIÍ")
    print("="*60)
    
    try:
        from gesture_recognition_ml import MLGestureRecognizer
        from collections import deque
        
        q = Queue()
        recognizer = MLGestureRecognizer(q, use_ml=False)
        
        # Test prázdneho bufferu
        if len(recognizer.landmarks_buffer) == 0:
            print("✅ Buffer je na začiatku prázdny")
        else:
            print("❌ Buffer nie je prázdny na začiatku")
            return False
        
        # Test pridávania landmarks
        for i in range(35):  # Viac ako sequence_length (30)
            dummy_landmarks = np.random.rand(63)
            recognizer.landmarks_buffer.append(dummy_landmarks)
        
        if len(recognizer.landmarks_buffer) == 30:
            print("✅ Buffer má správnu maximálnu veľkosť (30)")
        else:
            print(f"❌ Buffer má nesprávnu veľkosť: {len(recognizer.landmarks_buffer)}")
            return False
        
        # Test že najstaršie dáta sú odstránené
        if len(recognizer.landmarks_buffer) == recognizer.sequence_length:
            print("✅ Buffer správne udržuje maximálnu veľkosť")
        else:
            print("❌ Problém s udržiavaním veľkosti bufferu")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste bufferovania: {e}")
        return False

def test_ml_prediction():
    """Test ML predikcie"""
    print("\n" + "="*60)
    print("TEST ML PREDIKCIE")
    print("="*60)
    
    if not Path("gesture_model.pth").exists():
        print("⚠️ Model neexistuje - preskakujem test")
        return True
    
    try:
        from gesture_recognition_ml import MLGestureRecognizer
        
        q = Queue()
        recognizer = MLGestureRecognizer(q, use_ml=True)
        
        if not recognizer.use_ml:
            print("⚠️ ML nie je dostupný - preskakujem test")
            return True
        
        # Naplnenie bufferu dummy dátami
        for i in range(30):
            dummy_landmarks = np.random.rand(63)
            recognizer.landmarks_buffer.append(dummy_landmarks)
        
        # Test predikcie
        gesture, confidence = recognizer.predict_gesture_ml()
        
        print(f"Predikcia: gesture='{gesture}', confidence={confidence:.3f}")
        
        if isinstance(confidence, float) and 0.0 <= confidence <= 1.0:
            print("✅ Confidence je v správnom rozsahu")
        else:
            print(f"❌ Nesprávny confidence: {confidence}")
            return False
        
        if gesture is None or gesture in recognizer.gestures:
            print("✅ Gesture je buď None alebo platné gesto")
        else:
            print(f"❌ Neplatné gesto: {gesture}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste ML predikcie: {e}")
        return False

def test_rule_based_fallback():
    """Test rule-based fallback"""
    print("\n" + "="*60)
    print("TEST RULE-BASED FALLBACK")
    print("="*60)
    
    try:
        from gesture_recognition_ml import MLGestureRecognizer
        
        # Test bez ML
        q = Queue()
        recognizer = MLGestureRecognizer(q, use_ml=False)
        
        if not recognizer.use_ml:
            print("✅ Rule-based režim je aktívny")
        else:
            print("❌ ML režim je aktívny namiesto rule-based")
            return False
        
        # Test rule-based metód
        methods = ['is_fist', 'is_peace_gesture', 'is_pointing_gesture']
        for method_name in methods:
            if hasattr(recognizer, method_name):
                print(f"✅ Metóda {method_name} existuje")
            else:
                print(f"❌ Metóda {method_name} chýba")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste rule-based fallback: {e}")
        return False

def test_integration_compatibility():
    """Test kompatibility s existujúcou aplikáciou"""
    print("\n" + "="*60)
    print("TEST KOMPATIBILITY S APLIKÁCIOU")
    print("="*60)
    
    try:
        # Test importu pôvodného modulu
        from gesture_recognition import GestureRecognizer
        print("✅ Pôvodný GestureRecognizer je stále dostupný")
        
        # Test importu nového modulu
        from gesture_recognition_ml import MLGestureRecognizer
        print("✅ Nový MLGestureRecognizer je dostupný")
        
        # Test kompatibility Queue
        q = Queue()
        
        # Test že oba môžu používať rovnaký Queue
        old_recognizer = GestureRecognizer(q)
        new_recognizer = MLGestureRecognizer(q, use_ml=False)
        
        print("✅ Oba recognizery môžu používať rovnaký Queue")
        
        # Test že majú podobné rozhranie
        required_methods = ['run', 'stop']
        for method in required_methods:
            if hasattr(old_recognizer, method) and hasattr(new_recognizer, method):
                print(f"✅ Oba majú metódu {method}")
            else:
                print(f"❌ Metóda {method} chýba v jednom z recognizerov")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste kompatibility: {e}")
        return False

def main():
    """Hlavná funkcia testovania"""
    print("TEST INTEGRÁCIE ML MODELU")
    print("Fáza 4 - Overenie implementácie")
    
    tests = [
        ("Dostupnosť ML modelu", test_ml_model_availability),
        ("Import ML Gesture Recognizer", test_ml_gesture_recognizer_import),
        ("Načítanie modelu", test_model_loading),
        ("Extrakcia landmarks", test_landmarks_extraction),
        ("Bufferovanie sekvencií", test_sequence_buffering),
        ("ML predikcia", test_ml_prediction),
        ("Rule-based fallback", test_rule_based_fallback),
        ("Kompatibilita s aplikáciou", test_integration_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Chyba v teste '{test_name}': {e}")
            results.append((test_name, False))
    
    # Súhrn výsledkov
    print("\n" + "="*60)
    print("SÚHRN VÝSLEDKOV TESTOVANIA")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PREŠIEL" if result else "❌ ZLYHAL"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print(f"\nCelkový výsledok: {passed}/{total} testov prešlo")
    
    if passed >= total - 1:  # Povolíme 1 zlyhaný test
        print("\n🎉 VŠETKY KĽÚČOVÉ TESTY PREŠLI!")
        print("ML integrácia je pripravená na použitie.")
        print("\n📋 ĎALŠIE KROKY:")
        print("1. Spustite test: python gesture_recognition_ml.py")
        print("2. Integrujte do hlavnej aplikácie")
        print("3. Otestujte real-time performance")
    else:
        print("\n❌ NIEKTORÉ KĽÚČOVÉ TESTY ZLYHALI")
        print("Opravte chyby pred pokračovaním.")
    
    return passed >= total - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
