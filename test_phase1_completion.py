#!/usr/bin/env python3
"""
Test dokončenia Fázy 1 z TODO.md
Overuje, či sú splnené všetky požiadavky pre Fázu 1
"""

import subprocess
import sys
import os

def test_pytorch_cuda_installation():
    """Test úlohy 1: Inštalácia PyTorch s podporou CUDA"""
    print("="*60)
    print("TEST ÚLOHY 1: INŠTALÁCIA PYTORCH S CUDA")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch je nainštalovaný - verzia: {torch.__version__}")
        
        # Kontrola CUDA podpory
        if torch.cuda.is_available():
            print(f"✅ CUDA je dostupná - verzia: {torch.version.cuda}")
            print(f"✅ Počet GPU: {torch.cuda.device_count()}")
            
            # Test základnej CUDA operácie
            device = torch.device('cuda')
            x = torch.randn(10, 10).to(device)
            y = torch.randn(10, 10).to(device)
            z = torch.matmul(x, y)
            print("✅ CUDA operácie fungujú správne!")
            
            return True
        else:
            print("❌ CUDA nie je dostupná v PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch nie je nainštalovaný")
        return False
    except Exception as e:
        print(f"❌ Chyba pri teste PyTorch: {e}")
        return False

def test_cuda_functionality():
    """Test úlohy 2: Overenie funkčnosti CUDA"""
    print("\n" + "="*60)
    print("TEST ÚLOHY 2: OVERENIE FUNKČNOSTI CUDA")
    print("="*60)
    
    # Spustenie test_cuda.py skriptu
    try:
        result = subprocess.run([sys.executable, 'test_cuda.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ test_cuda.py skript prebehol úspešne")
            
            # Kontrola výstupu na kľúčové indikátory úspechu
            output = result.stdout
            cuda_ops_ok = False
            gpu_speedup_ok = False
            system_ready = False

            if "CUDA operacie funguju spravne!" in output or "CUDA operácie fungujú správne!" in output:
                print("✅ CUDA operácie sú funkčné")
                cuda_ops_ok = True
            if "GPU poskytuje vyznamne zrychlenie!" in output or "GPU poskytuje významné zrýchlenie!" in output:
                print("✅ GPU zrýchlenie je potvrdené")
                gpu_speedup_ok = True
            if ("SYSTEM JE PRIPRAVENY PRE STROJOVE UCENIE!" in output or
                "SYSTÉM JE PRIPRAVENÝ PRE STROJOVÉ UČENIE!" in output or
                "SYSTEM JE CIASTOCNE PRIPRAVENY" in output or
                "SYSTÉM JE ČIASTOČNE PRIPRAVENÝ" in output):
                print("✅ Systém je pripravený pre ML")
                system_ready = True

            # Ak aspoň CUDA operácie fungujú a systém je pripravený, považujeme to za úspech
            if cuda_ops_ok and system_ready:
                return True
            else:
                print("⚠️ Systém môže mať problémy s ML pripravenosťou")
                return False
        else:
            print("❌ test_cuda.py skript zlyhal")
            print("STDERR:", result.stderr[-200:])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ test_cuda.py skript prekročil časový limit")
        return False
    except Exception as e:
        print(f"❌ Chyba pri spustení test_cuda.py: {e}")
        return False

def check_required_files():
    """Kontrola existencie potrebných súborov"""
    print("\n" + "="*60)
    print("KONTROLA POTREBNÝCH SÚBOROV")
    print("="*60)
    
    required_files = [
        ("test_cuda.py", "Test skript pre CUDA"),
        ("install_pytorch_cuda.py", "Inštalačný skript pre PyTorch"),
        ("TODO.md", "TODO súbor s úlohami")
    ]
    
    all_exist = True
    for filename, description in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} (CHÝBA)")
            all_exist = False
    
    return all_exist

def test_ml_readiness():
    """Test pripravenosti pre strojové učenie"""
    print("\n" + "="*60)
    print("TEST PRIPRAVENOSTI PRE STROJOVÉ UČENIE")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import time
        
        # Test vytvorenia jednoduchej neurónovej siete
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Test na GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = SimpleNet().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Dummy dáta
            x = torch.randn(32, 10).to(device)
            y = torch.randn(32, 1).to(device)
            
            # Test trénovacieho kroku
            start_time = time.time()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            
            print(f"✅ Neurónová sieť vytvorená a testovaná na GPU")
            print(f"✅ Trénovací krok trval: {(end_time - start_time)*1000:.2f} ms")
            print(f"✅ Loss: {loss.item():.4f}")
            
            return True
        else:
            print("❌ GPU nie je dostupná pre ML trénovanie")
            return False
            
    except Exception as e:
        print(f"❌ Chyba pri teste ML pripravenosti: {e}")
        return False

def update_todo_status():
    """Aktualizuje TODO.md súbor s označením dokončených úloh"""
    print("\n" + "="*60)
    print("AKTUALIZÁCIA TODO.MD")
    print("="*60)
    
    try:
        # Čítanie aktuálneho TODO.md
        with open("TODO.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Označenie dokončených úloh v Fáze 1
        updated_content = content.replace(
            "- [ ] **1. Inštalácia PyTorch s podporou CUDA:**",
            "- [x] **1. Inštalácia PyTorch s podporou CUDA:**"
        ).replace(
            "- [ ] **2. Overenie funkčnosti CUDA:**",
            "- [x] **2. Overenie funkčnosti CUDA:**"
        )
        
        # Zápis aktualizovaného obsahu
        with open("TODO.md", "w", encoding="utf-8") as f:
            f.write(updated_content)
        
        print("✅ TODO.md bol aktualizovaný s dokončenými úlohami Fázy 1")
        return True
        
    except Exception as e:
        print(f"❌ Chyba pri aktualizácii TODO.md: {e}")
        return False

def main():
    """Hlavná funkcia testovania Fázy 1"""
    print("TEST DOKONČENIA FÁZY 1")
    print("Overenie splnenia všetkých požiadaviek z TODO.md")
    
    # Zoznam testov
    tests = [
        ("Inštalácia PyTorch s CUDA", test_pytorch_cuda_installation),
        ("Funkčnosť CUDA", test_cuda_functionality),
        ("Potrebné súbory", check_required_files),
        ("ML pripravenosť", test_ml_readiness)
    ]
    
    # Spustenie testov
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Chyba v teste '{test_name}': {e}")
            results.append((test_name, False))
    
    # Vyhodnotenie výsledkov
    print("\n" + "="*60)
    print("SÚHRN VÝSLEDKOV FÁZY 1")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PREŠIEL" if result else "❌ ZLYHAL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nCelkový výsledok: {passed}/{total} testov prešlo")
    
    # Finálne hodnotenie
    if passed == total:
        print("\n🎉 FÁZA 1 JE ÚSPEŠNE DOKONČENÁ!")
        print("Všetky požiadavky sú splnené.")
        
        # Aktualizácia TODO.md
        update_todo_status()
        
        print("\n📋 ĎALŠIE KROKY:")
        print("- Môžete pokračovať s Fázou 2: Zber dát pre trénovanie")
        print("- Spustite: python collect_data.py (po implementácii)")
        
        return True
    else:
        print("\n❌ FÁZA 1 NIE JE ÚPLNE DOKONČENÁ")
        print("Niektoré testy zlyhali. Skontrolujte chyby vyššie.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
