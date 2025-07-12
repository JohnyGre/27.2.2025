#!/usr/bin/env python3
"""
Skript na inštaláciu PyTorch s podporou CUDA
Fáza 1 - Úloha 1: Inštalácia PyTorch s podporou CUDA
"""

import subprocess
import sys
import platform
import importlib.util

def check_current_pytorch():
    """Skontroluje aktuálnu inštaláciu PyTorch"""
    print("="*60)
    print("KONTROLA AKTUÁLNEJ PYTORCH INŠTALÁCIE")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch je už nainštalovaný - verzia: {torch.__version__}")
        
        # Kontrola CUDA podpory
        if torch.cuda.is_available():
            print(f"✅ CUDA je dostupná - verzia: {torch.version.cuda}")
            print(f"✅ Počet GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True, True
        else:
            print("❌ CUDA nie je dostupná v aktuálnej inštalácii")
            return True, False
    except ImportError:
        print("❌ PyTorch nie je nainštalovaný")
        return False, False

def detect_system_info():
    """Detekuje informácie o systéme pre správnu inštaláciu"""
    print("\n" + "="*60)
    print("DETEKCIA SYSTÉMU")
    print("="*60)
    
    os_name = platform.system()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    print(f"Operačný systém: {os_name}")
    print(f"Python verzia: {python_version}")
    
    # Detekcia CUDA verzie z nvidia-smi
    cuda_version = None
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    break
            print(f"CUDA verzia (z nvidia-smi): {cuda_version}")
        else:
            print("⚠️ nvidia-smi nie je dostupné")
    except:
        print("⚠️ Nemožno detekovať CUDA verziu")
    
    return os_name, python_version, cuda_version

def get_pytorch_install_command(os_name, python_version, cuda_version):
    """Vráti správny príkaz na inštaláciu PyTorch"""
    print("\n" + "="*60)
    print("GENEROVANIE INŠTALAČNÉHO PRÍKAZU")
    print("="*60)
    
    # Základný príkaz
    base_cmd = [sys.executable, "-m", "pip", "install"]
    
    # Určenie správnej CUDA verzie pre PyTorch
    if cuda_version:
        if cuda_version.startswith("12."):
            torch_cuda = "cu121"  # PyTorch podporuje CUDA 12.1
        elif cuda_version.startswith("11.8"):
            torch_cuda = "cu118"
        elif cuda_version.startswith("11."):
            torch_cuda = "cu118"  # Fallback na 11.8
        else:
            torch_cuda = "cu121"  # Default na najnovšiu
    else:
        torch_cuda = "cu121"  # Default ak nevieme detekovať
    
    print(f"Vybraná PyTorch CUDA verzia: {torch_cuda}")
    
    # Balíčky na inštaláciu
    packages = ["torch", "torchvision", "torchaudio"]
    
    # Index URL pre CUDA verziu
    index_url = f"https://download.pytorch.org/whl/{torch_cuda}"
    
    # Kompletný príkaz
    install_cmd = base_cmd + packages + ["--index-url", index_url]
    
    print(f"Inštalačný príkaz: {' '.join(install_cmd)}")
    return install_cmd

def install_pytorch_cuda(install_cmd):
    """Nainštaluje PyTorch s CUDA podporou"""
    print("\n" + "="*60)
    print("INŠTALÁCIA PYTORCH S CUDA")
    print("="*60)
    
    try:
        print("Spúšťam inštaláciu...")
        print("Toto môže trvať niekoľko minút...")
        
        # Spustenie inštalácie
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Inštalácia úspešná!")
            return True
        else:
            print("❌ Inštalácia zlyhala!")
            print("STDOUT:", result.stdout[-500:])  # Posledných 500 znakov
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Inštalácia prekročila časový limit (10 minút)")
        return False
    except Exception as e:
        print(f"❌ Chyba pri inštalácii: {e}")
        return False

def verify_installation():
    """Overí úspešnosť inštalácie"""
    print("\n" + "="*60)
    print("OVERENIE INŠTALÁCIE")
    print("="*60)
    
    try:
        # Reload modulu ak už bol importovaný
        if 'torch' in sys.modules:
            importlib.reload(sys.modules['torch'])
        
        import torch
        print(f"✅ PyTorch verzia: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA je dostupná - verzia: {torch.version.cuda}")
            print(f"✅ Počet GPU: {torch.cuda.device_count()}")
            
            # Test základnej operácie
            device = torch.device('cuda')
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            print("✅ CUDA operácie fungujú!")
            
            return True
        else:
            print("❌ CUDA nie je dostupná po inštalácii")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch sa nepodarilo importovať: {e}")
        return False
    except Exception as e:
        print(f"❌ Chyba pri overovaní: {e}")
        return False

def main():
    """Hlavná funkcia inštalácie"""
    print("INŠTALÁCIA PYTORCH S CUDA PODPOROU")
    print("Fáza 1 - Úloha 1")
    
    # Kontrola aktuálnej inštalácie
    pytorch_installed, cuda_available = check_current_pytorch()
    
    if pytorch_installed and cuda_available:
        print("\n🎉 PyTorch s CUDA je už správne nainštalovaný!")
        print("Môžete pokračovať s ďalšími úlohami.")
        return True
    
    # Detekcia systému
    os_name, python_version, cuda_version = detect_system_info()
    
    # Generovanie inštalačného príkazu
    install_cmd = get_pytorch_install_command(os_name, python_version, cuda_version)
    
    # Potvrdenie od používateľa
    print(f"\n⚠️ Chystáte sa nainštalovať/aktualizovať PyTorch s CUDA podporou.")
    print("Toto môže prepísať existujúcu inštaláciu PyTorch.")
    
    # Pre automatické spustenie bez interakcie
    response = "y"  # Automaticky pokračujeme
    
    if response.lower() in ['y', 'yes', 'ano', 'a']:
        # Inštalácia
        success = install_pytorch_cuda(install_cmd)
        
        if success:
            # Overenie
            if verify_installation():
                print("\n🎉 ÚSPECH! PyTorch s CUDA je správne nainštalovaný!")
                print("Fáza 1 - Úloha 1 je dokončená.")
                return True
            else:
                print("\n❌ Inštalácia prebehla, ale overenie zlyhalo.")
                return False
        else:
            print("\n❌ Inštalácia zlyhala.")
            return False
    else:
        print("Inštalácia zrušená používateľom.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
