#!/usr/bin/env python3
"""
Test skript pre overenie funkcionality zberu dát
Fáza 2 - Test implementácie
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json

def test_mediapipe_functionality():
    """Test základnej funkcionality MediaPipe"""
    print("="*60)
    print("TEST MEDIAPIPE FUNKCIONALITY")
    print("="*60)
    
    try:
        # Inicializácia MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils
        
        print("✅ MediaPipe úspešne inicializovaný")
        
        # Test kamery
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Kamera je dostupná")
            
            # Test jedného snímku
            ret, frame = cap.read()
            if ret:
                print("✅ Snímanie z kamery funguje")
                
                # Test detekcie rúk
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    print("✅ Ruky boli detekované v teste")
                    
                    # Test extrakcie landmarks
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    landmarks_array = np.array(landmarks)
                    print(f"✅ Extraktované {len(landmarks_array)} hodnôt landmarks")
                    
                    if len(landmarks_array) == 63:  # 21 landmarks * 3 súradnice
                        print("✅ Správny počet landmarks (21 * 3 = 63)")
                    else:
                        print(f"⚠️ Neočakávaný počet landmarks: {len(landmarks_array)}")
                else:
                    print("⚠️ Žiadne ruky neboli detekované (normálne ak nie sú ruky pred kamerou)")
            else:
                print("❌ Nemožno čítať z kamery")
                cap.release()
                return False
            
            cap.release()
        else:
            print("❌ Kamera nie je dostupná")
            return False
        
        hands.close()
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste MediaPipe: {e}")
        return False

def test_collect_data_import():
    """Test importu collect_data modulu"""
    print("\n" + "="*60)
    print("TEST IMPORTU COLLECT_DATA MODULU")
    print("="*60)
    
    try:
        from collect_data import GestureDataCollector
        print("✅ GestureDataCollector úspešne importovaný")
        
        # Test inicializácie
        collector = GestureDataCollector("test_gesture_data")
        print("✅ GestureDataCollector úspešne inicializovaný")
        
        # Test dostupných giest
        gestures = collector.available_gestures
        print(f"✅ Dostupné gestá: {', '.join(gestures)}")
        
        if len(gestures) >= 4:
            print("✅ Dostatok giest pre trénovanie")
        else:
            print("⚠️ Málo giest pre efektívne trénovanie")
        
        # Test metadát
        collector.load_metadata()
        print("✅ Metadata úspešne načítané")
        
        return True
        
    except ImportError as e:
        print(f"❌ Chyba pri importe: {e}")
        return False
    except Exception as e:
        print(f"❌ Chyba pri teste: {e}")
        return False

def test_data_structure():
    """Test štruktúry dát"""
    print("\n" + "="*60)
    print("TEST ŠTRUKTÚRY DÁT")
    print("="*60)
    
    try:
        from collect_data import GestureDataCollector
        
        # Vytvorenie test adresára
        test_data_path = Path("test_gesture_data")
        collector = GestureDataCollector(test_data_path)
        
        # Test vytvorenia adresára pre gesto
        test_gesture = "test_gesture"
        gesture_dir = collector.create_gesture_directory(test_gesture)
        
        if gesture_dir.exists():
            print("✅ Adresár pre gesto úspešne vytvorený")
        else:
            print("❌ Adresár pre gesto nebol vytvorený")
            return False
        
        # Test číslovania sekvencií
        next_seq = collector.get_next_sequence_number(test_gesture)
        if next_seq == 0:
            print("✅ Správne číslovanie sekvencií (začína od 0)")
        else:
            print(f"⚠️ Neočakávané číslo sekvencie: {next_seq}")
        
        # Test uloženia dummy dát
        sequence_dir = test_data_path / test_gesture / "0"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # Vytvorenie dummy landmarks
        dummy_landmarks = np.random.rand(63)  # 21 landmarks * 3 súradnice
        np.save(sequence_dir / "0.npy", dummy_landmarks)
        
        # Test načítania
        loaded_landmarks = np.load(sequence_dir / "0.npy")
        if np.array_equal(dummy_landmarks, loaded_landmarks):
            print("✅ Uloženie a načítanie landmarks funguje")
        else:
            print("❌ Problém s uložením/načítaním landmarks")
            return False
        
        # Vyčistenie test dát
        import shutil
        if test_data_path.exists():
            shutil.rmtree(test_data_path)
            print("✅ Test dáta vyčistené")
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste štruktúry dát: {e}")
        return False

def test_compatibility_with_train_model():
    """Test kompatibility s train_model.py"""
    print("\n" + "="*60)
    print("TEST KOMPATIBILITY S TRAIN_MODEL.PY")
    print("="*60)
    
    try:
        from collect_data import GestureDataCollector
        
        # Načítanie konfigurácie z train_model.py
        if os.path.exists("train_model.py"):
            with open("train_model.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extrakcia giest z train_model.py
            if "gestures = np.array([" in content:
                start = content.find("gestures = np.array([") + len("gestures = np.array([")
                end = content.find("])", start)
                gestures_str = content[start:end]
                
                # Parsovanie giest
                train_gestures = [g.strip().strip("'\"") for g in gestures_str.split(",")]
                print(f"Gestá v train_model.py: {train_gestures}")
                
                # Porovnanie s collect_data.py
                collector = GestureDataCollector()
                collect_gestures = collector.available_gestures
                print(f"Gestá v collect_data.py: {collect_gestures}")
                
                # Kontrola kompatibility
                compatible = all(gesture in collect_gestures for gesture in train_gestures)
                if compatible:
                    print("✅ Gestá sú kompatibilné medzi collect_data.py a train_model.py")
                else:
                    missing = [g for g in train_gestures if g not in collect_gestures]
                    print(f"⚠️ Chýbajúce gestá v collect_data.py: {missing}")
                
                return compatible
            else:
                print("⚠️ Nemožno extrahovať gestá z train_model.py")
                return False
        else:
            print("⚠️ train_model.py súbor neexistuje")
            return False
            
    except Exception as e:
        print(f"❌ Chyba pri teste kompatibility: {e}")
        return False

def test_data_format():
    """Test formátu dát"""
    print("\n" + "="*60)
    print("TEST FORMÁTU DÁT")
    print("="*60)
    
    try:
        # Simulácia MediaPipe výstupu
        mp_hands = mp.solutions.hands
        
        # Vytvorenie dummy landmarks
        class DummyLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        class DummyHandLandmarks:
            def __init__(self):
                self.landmark = [DummyLandmark(i*0.1, i*0.1, i*0.01) for i in range(21)]
        
        class DummyResults:
            def __init__(self):
                self.multi_hand_landmarks = [DummyHandLandmarks()]
        
        # Test extrakcie
        from collect_data import GestureDataCollector
        collector = GestureDataCollector()
        
        dummy_results = DummyResults()
        landmarks = collector.extract_landmarks(dummy_results)
        
        print(f"✅ Extraktované landmarks: shape {landmarks.shape}")
        
        if landmarks.shape == (63,):
            print("✅ Správny formát landmarks (63 hodnôt)")
        else:
            print(f"❌ Nesprávny formát landmarks: {landmarks.shape}")
            return False
        
        # Test prázdnych výsledkov
        class EmptyResults:
            def __init__(self):
                self.multi_hand_landmarks = None
        
        empty_results = EmptyResults()
        empty_landmarks = collector.extract_landmarks(empty_results)
        
        if empty_landmarks.shape == (63,) and np.all(empty_landmarks == 0):
            print("✅ Správne spracovanie prázdnych výsledkov")
        else:
            print("❌ Problém so spracovaním prázdnych výsledkov")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste formátu dát: {e}")
        return False

def main():
    """Hlavná funkcia testovania"""
    print("TEST FUNKCIONALITY ZBERU DÁT")
    print("Fáza 2 - Overenie implementácie")
    
    tests = [
        ("MediaPipe funkcionalita", test_mediapipe_functionality),
        ("Import collect_data modulu", test_collect_data_import),
        ("Štruktúra dát", test_data_structure),
        ("Kompatibilita s train_model", test_compatibility_with_train_model),
        ("Formát dát", test_data_format)
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
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nCelkový výsledok: {passed}/{total} testov prešlo")
    
    if passed == total:
        print("\n🎉 VŠETKY TESTY PREŠLI!")
        print("Zber dát je pripravený na použitie.")
        print("\n📋 ĎALŠIE KROKY:")
        print("1. Spustite: python collect_data.py")
        print("2. Zozbierajte dáta pre všetky gestá (odporúčané: 30+ sekvencií na gesto)")
        print("3. Po zbere dát spustite: python train_model.py")
    else:
        print("\n❌ NIEKTORÉ TESTY ZLYHALI")
        print("Opravte chyby pred pokračovaním.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
