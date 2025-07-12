#!/usr/bin/env python3
"""
Jednoduchý test ML gesture recognition bez multiprocessing
"""

import sys
import time
import threading
from queue import Queue
from pathlib import Path
import pyautogui

# Vypnutie PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

def test_ml_gesture_recognition():
    """Test ML gesture recognition s threading"""
    print("🤲 TEST ML GESTURE RECOGNITION")
    print("=" * 50)
    
    try:
        from gesture_recognition_ml import MLGestureRecognizer
        
        # Vytvorenie queue
        q = Queue()
        
        # Test bez ML modelu (fallback)
        print("Testovanie bez ML modelu (rule-based fallback)...")
        recognizer = MLGestureRecognizer(q, use_ml=False)
        print(f"✅ MLGestureRecognizer vytvorený (ML: {recognizer.use_ml})")
        
        # Test s ML modelom ak existuje
        model_exists = Path("gesture_model.pth").exists()
        print(f"ML model: {'✅ Dostupný' if model_exists else '❌ Nedostupný'}")
        
        if model_exists:
            recognizer_ml = MLGestureRecognizer(q, use_ml=True)
            print(f"✅ MLGestureRecognizer s ML vytvorený (ML: {recognizer_ml.use_ml})")
        
        # Test spustenia v thread
        print("\nSpúšťam gesture recognition v thread...")
        
        def run_recognition():
            try:
                recognizer.run()
            except Exception as e:
                print(f"Chyba v recognition thread: {e}")
        
        # Spustenie v thread
        recognition_thread = threading.Thread(target=run_recognition, daemon=True)
        recognition_thread.start()
        
        print("✅ Thread spustený")
        print("Sledovanie giest na 10 sekúnd...")
        print("Ukážte gestá pred kamerou!")
        
        # Sledovanie výstupu
        start_time = time.time()
        gesture_count = 0
        
        while time.time() - start_time < 10:
            if not q.empty():
                try:
                    gesture_type, gesture_info = q.get_nowait()
                    if gesture_type == "gesture":
                        gesture_count += 1
                        print(f"🎯 Rozpoznané gesto #{gesture_count}: {gesture_info}")
                except:
                    pass
            time.sleep(0.1)
        
        print(f"\n📊 Výsledky:")
        print(f"   - Celkový čas: 10 sekúnd")
        print(f"   - Rozpoznané gestá: {gesture_count}")
        print(f"   - Thread stav: {'✅ Aktívny' if recognition_thread.is_alive() else '❌ Ukončený'}")
        
        # Zastavenie
        recognizer.stop()
        print("✅ Test dokončený")
        
        return True
        
    except ImportError as e:
        print(f"❌ Chyba pri importe: {e}")
        return False
    except Exception as e:
        print(f"❌ Chyba pri teste: {e}")
        return False

def test_main_ml_import():
    """Test importu main_ml.py"""
    print("\n🏠 TEST MAIN_ML IMPORT")
    print("=" * 50)
    
    try:
        # Test importu bez spustenia
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("main_ml", "main_ml.py")
        if spec is None:
            print("❌ main_ml.py súbor neexistuje")
            return False
        
        main_ml = importlib.util.module_from_spec(spec)
        
        # Test načítania modulu
        spec.loader.exec_module(main_ml)
        print("✅ main_ml.py sa dá načítať")
        
        # Test existencie kľúčových tried
        if hasattr(main_ml, 'ModernGUI'):
            print("✅ ModernGUI trieda existuje")
        else:
            print("❌ ModernGUI trieda chýba")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste main_ml: {e}")
        return False

def main():
    """Hlavná funkcia testovania"""
    print("🧪 JEDNODUCHÝ TEST ML GESTURE RECOGNITION")
    print("Stlačte Ctrl+C pre ukončenie")
    
    tests = [
        ("ML Gesture Recognition", test_ml_gesture_recognition),
        ("Main ML Import", test_main_ml_import)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n⏹️ Test prerušený používateľom")
            break
        except Exception as e:
            print(f"❌ Chyba v teste '{test_name}': {e}")
            results.append((test_name, False))
    
    # Súhrn
    print("\n" + "=" * 50)
    print("📊 SÚHRN TESTOV")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PREŠIEL" if result else "❌ ZLYHAL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nVýsledok: {passed}/{total} testov prešlo")
    
    if passed == total:
        print("\n🎉 VŠETKY TESTY PREŠLI!")
        print("\n📋 ĎALŠIE KROKY:")
        print("1. Spustite: python main_ml.py")
        print("2. Kliknite na '🤲 ML Gestá OFF' pre zapnutie")
        print("3. Ukážte gestá pred kamerou")
    else:
        print("\n⚠️ NIEKTORÉ TESTY ZLYHALI")
        print("Skontrolujte chyby vyššie")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test ukončený")
        sys.exit(0)
