#!/usr/bin/env python3
"""
Test skript pre overenie funkcionality trénovania modelu
Fáza 3 - Test implementácie
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import shutil

def test_pytorch_cuda_availability():
    """Test dostupnosti PyTorch a CUDA"""
    print("="*60)
    print("TEST PYTORCH A CUDA DOSTUPNOSTI")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch verzia: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA je dostupná - verzia: {torch.version.cuda}")
            print(f"✅ GPU zariadení: {torch.cuda.device_count()}")
            
            # Test základnej CUDA operácie
            device = torch.device('cuda')
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            print("✅ CUDA operácie fungujú")
            
            return True
        else:
            print("⚠️ CUDA nie je dostupná, použije sa CPU")
            return True
            
    except Exception as e:
        print(f"❌ Chyba pri teste PyTorch/CUDA: {e}")
        return False

def test_train_model_import():
    """Test importu train_model modulov"""
    print("\n" + "="*60)
    print("TEST IMPORTU TRAIN_MODEL MODULOV")
    print("="*60)
    
    # Test pôvodného train_model.py
    try:
        # Kontrola existencie súboru
        if os.path.exists("train_model.py"):
            print("✅ train_model.py súbor existuje")
        else:
            print("❌ train_model.py súbor neexistuje")
            return False
        
        # Test importu (bez spustenia)
        with open("train_model.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Kontrola kľúčových komponentov
        required_components = [
            "import torch",
            "import torch.nn as nn",
            "class GestureLSTM",
            "torch.save",
            "CUDA"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✅ Obsahuje: {component}")
            else:
                print(f"⚠️ Chýba: {component}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste train_model.py: {e}")
        return False

def test_enhanced_train_model():
    """Test rozšíreného train_model_enhanced.py"""
    print("\n" + "="*60)
    print("TEST ROZŠÍRENÉHO TRAIN_MODEL_ENHANCED.PY")
    print("="*60)
    
    try:
        from train_model_enhanced import GestureTrainer, GestureLSTM
        print("✅ GestureTrainer a GestureLSTM úspešne importované")
        
        # Test inicializácie
        trainer = GestureTrainer()
        print("✅ GestureTrainer úspešne inicializovaný")
        
        # Test modelu
        model = GestureLSTM(input_size=63, hidden_size=128, num_classes=4)
        print("✅ GestureLSTM model úspešne vytvorený")
        
        # Test forward pass
        dummy_input = torch.randn(2, 30, 63)  # batch_size=2, seq_len=30, features=63
        output = model(dummy_input)
        
        if output.shape == (2, 4):  # batch_size=2, num_classes=4
            print("✅ Model forward pass funguje správne")
        else:
            print(f"❌ Nesprávny výstup modelu: {output.shape}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Chyba pri importe: {e}")
        return False
    except Exception as e:
        print(f"❌ Chyba pri teste enhanced modelu: {e}")
        return False

def test_data_loading():
    """Test načítania dát"""
    print("\n" + "="*60)
    print("TEST NAČÍTANIA DÁT")
    print("="*60)
    
    try:
        from train_model_enhanced import GestureTrainer
        
        # Kontrola existencie gesture_data adresára
        data_path = Path("gesture_data")
        if not data_path.exists():
            print("⚠️ gesture_data adresár neexistuje - vytváram dummy dáta")
            return create_dummy_data_and_test()
        
        # Test s existujúcimi dátami
        trainer = GestureTrainer()
        
        try:
            trainer.load_data()
            print(f"✅ Dáta úspešne načítané: {trainer.X.shape if trainer.X is not None else 'None'}")
            
            if trainer.X is not None and len(trainer.X) > 0:
                print(f"✅ Počet sekvencií: {len(trainer.X)}")
                print(f"✅ Gestá: {trainer.gestures}")
                return True
            else:
                print("⚠️ Žiadne dáta neboli načítané")
                return False
                
        except Exception as e:
            print(f"⚠️ Chyba pri načítaní existujúcich dát: {e}")
            print("Vytváram dummy dáta pre test...")
            return create_dummy_data_and_test()
        
    except Exception as e:
        print(f"❌ Chyba pri teste načítania dát: {e}")
        return False

def create_dummy_data_and_test():
    """Vytvorí dummy dáta a otestuje načítanie"""
    try:
        from train_model_enhanced import GestureTrainer
        
        # Vytvorenie dočasného adresára
        temp_dir = Path("test_gesture_data")
        temp_dir.mkdir(exist_ok=True)
        
        gestures = ['pest', 'otvorena_dlan', 'palec_hore', 'ukazovak']
        
        # Vytvorenie dummy dát
        for gesture in gestures:
            gesture_dir = temp_dir / gesture
            gesture_dir.mkdir(exist_ok=True)
            
            # Vytvorenie 5 sekvencií pre každé gesto
            for seq_idx in range(5):
                seq_dir = gesture_dir / str(seq_idx)
                seq_dir.mkdir(exist_ok=True)
                
                # Vytvorenie 30 snímkov pre každú sekvenciu
                for frame_idx in range(30):
                    dummy_landmarks = np.random.rand(63)  # 21 landmarks * 3 coords
                    np.save(seq_dir / f"{frame_idx}.npy", dummy_landmarks)
        
        # Vytvorenie metadát
        metadata = {
            'gestures': {gesture: 5 for gesture in gestures},
            'total_sequences': 20,
            'last_updated': '2025-07-12 23:00:00'
        }
        
        with open(temp_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Test načítania dummy dát
        trainer = GestureTrainer(data_path=temp_dir)
        trainer.load_data()
        
        if trainer.X is not None and len(trainer.X) == 20:
            print("✅ Dummy dáta úspešne vytvorené a načítané")
            
            # Test prípravy dát
            trainer.prepare_data()
            print("✅ Dáta úspešne pripravené pre trénovanie")
            
            # Vyčistenie
            shutil.rmtree(temp_dir)
            return True
        else:
            print("❌ Problém s dummy dátami")
            shutil.rmtree(temp_dir)
            return False
            
    except Exception as e:
        print(f"❌ Chyba pri vytváraní dummy dát: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False

def test_model_architecture():
    """Test architektúry modelu"""
    print("\n" + "="*60)
    print("TEST ARCHITEKTÚRY MODELU")
    print("="*60)
    
    try:
        from train_model_enhanced import GestureLSTM
        
        # Test rôznych konfigurácií
        configs = [
            {'input_size': 63, 'hidden_size': 64, 'num_classes': 4, 'num_layers': 1},
            {'input_size': 63, 'hidden_size': 128, 'num_classes': 4, 'num_layers': 2},
            {'input_size': 63, 'hidden_size': 256, 'num_classes': 8, 'num_layers': 3},
        ]
        
        for i, config in enumerate(configs):
            model = GestureLSTM(**config)
            
            # Test forward pass
            batch_size = 4
            seq_len = 30
            dummy_input = torch.randn(batch_size, seq_len, config['input_size'])
            
            output = model(dummy_input)
            expected_shape = (batch_size, config['num_classes'])
            
            if output.shape == expected_shape:
                print(f"✅ Konfigurácia {i+1}: {config} - OK")
            else:
                print(f"❌ Konfigurácia {i+1}: očakávaný {expected_shape}, dostal {output.shape}")
                return False
        
        # Test počtu parametrov
        model = GestureLSTM(63, 128, 4, 2)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Celkový počet parametrov: {total_params:,}")
        print(f"✅ Trénovateľné parametre: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste architektúry: {e}")
        return False

def test_training_components():
    """Test komponentov trénovania"""
    print("\n" + "="*60)
    print("TEST KOMPONENTOV TRÉNOVANIA")
    print("="*60)
    
    try:
        import torch.optim as optim
        from train_model_enhanced import GestureLSTM
        
        # Test modelu
        model = GestureLSTM(63, 128, 4)
        print("✅ Model vytvorený")
        
        # Test loss funkcie
        criterion = nn.CrossEntropyLoss()
        print("✅ Loss funkcia vytvorená")
        
        # Test optimizera
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("✅ Optimizer vytvorený")
        
        # Test schedulera
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        print("✅ Scheduler vytvorený")
        
        # Test trénovacieho kroku
        dummy_input = torch.randn(4, 30, 63)
        dummy_labels = torch.randint(0, 4, (4,))
        
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
        
        print(f"✅ Trénovací krok úspešný - Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste trénovacích komponentov: {e}")
        return False

def test_model_saving_loading():
    """Test ukladania a načítania modelu"""
    print("\n" + "="*60)
    print("TEST UKLADANIA A NAČÍTANIA MODELU")
    print("="*60)
    
    try:
        from train_model_enhanced import GestureLSTM
        
        # Vytvorenie modelu
        model1 = GestureLSTM(63, 128, 4)
        
        # Uloženie modelu
        test_model_path = "test_model.pth"
        torch.save(model1.state_dict(), test_model_path)
        print("✅ Model úspešne uložený")
        
        # Načítanie modelu
        model2 = GestureLSTM(63, 128, 4)
        model2.load_state_dict(torch.load(test_model_path))
        print("✅ Model úspešne načítaný")
        
        # Test, že modely sú identické
        dummy_input = torch.randn(2, 30, 63)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(dummy_input)
            output2 = model2(dummy_input)
        
        if torch.allclose(output1, output2, atol=1e-6):
            print("✅ Modely sú identické po uložení/načítaní")
        else:
            print("❌ Modely sa líšia po uložení/načítaní")
            return False
        
        # Vyčistenie
        os.remove(test_model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri teste ukladania/načítania: {e}")
        return False

def main():
    """Hlavná funkcia testovania"""
    print("TEST FUNKCIONALITY TRÉNOVANIA MODELU")
    print("Fáza 3 - Overenie implementácie")
    
    tests = [
        ("PyTorch a CUDA dostupnosť", test_pytorch_cuda_availability),
        ("Import train_model modulov", test_train_model_import),
        ("Rozšírený train_model", test_enhanced_train_model),
        ("Načítanie dát", test_data_loading),
        ("Architektúra modelu", test_model_architecture),
        ("Komponenty trénovania", test_training_components),
        ("Ukladanie/načítanie modelu", test_model_saving_loading)
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
        print("Trénovanie modelu je pripravené na použitie.")
        print("\n📋 ĎALŠIE KROKY:")
        print("1. Zozbierajte dáta: python collect_data.py")
        print("2. Spustite trénovanie: python train_model.py alebo python train_model_enhanced.py")
        print("3. Pokračujte s Fázou 4: Integrácia do aplikácie")
    else:
        print("\n❌ NIEKTORÉ TESTY ZLYHALI")
        print("Opravte chyby pred pokračovaním s tréningom.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
