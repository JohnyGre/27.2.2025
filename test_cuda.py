#!/usr/bin/env python3
"""
Test skript na overenie CUDA funkcionality pre PyTorch
Fáza 1 - Úloha 2: Overenie funkčnosti CUDA
"""

import sys
import subprocess
import platform
import time

def check_system_info():
    """Zobrazí základné informácie o systéme"""
    print("="*60)
    print("SYSTÉMOVÉ INFORMÁCIE")
    print("="*60)
    print(f"Operačný systém: {platform.system()} {platform.release()}")
    print(f"Architektúra: {platform.machine()}")
    print(f"Python verzia: {sys.version}")
    print()

def check_nvidia_gpu():
    """Overí prítomnosť NVIDIA GPU"""
    print("="*60)
    print("KONTROLA NVIDIA GPU")
    print("="*60)

    try:
        # Pokus o spustenie nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] NVIDIA GPU je dostupna!")
            print("NVIDIA-SMI vystup:")
            print("-" * 40)
            # Zobrazíme len prvých pár riadkov
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(line)
            print("-" * 40)
            return True
        else:
            print("[CHYBA] NVIDIA GPU nie je dostupna alebo nvidia-smi nefunguje")
            return False
    except FileNotFoundError:
        print("[CHYBA] nvidia-smi nie je nainstalovane")
        return False
    except subprocess.TimeoutExpired:
        print("[VAROVANIE] nvidia-smi timeout")
        return False
    except Exception as e:
        print(f"[CHYBA] Chyba pri kontrole GPU: {e}")
        return False

def check_pytorch_installation():
    """Overí inštaláciu PyTorch"""
    print("\n" + "="*60)
    print("KONTROLA PYTORCH INŠTALÁCIE")
    print("="*60)

    try:
        import torch
        print(f"[OK] PyTorch je nainstalovany - verzia: {torch.__version__}")

        # Kontrola CUDA podpory v PyTorch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA dostupnost v PyTorch: {'[OK] ANO' if cuda_available else '[CHYBA] NIE'}")

        if cuda_available:
            print(f"CUDA verzia: {torch.version.cuda}")
            print(f"Pocet GPU zariadeni: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")

            # Test základnej CUDA operácie
            try:
                device = torch.device('cuda')
                x = torch.randn(3, 3).to(device)
                y = torch.randn(3, 3).to(device)
                z = torch.matmul(x, y)
                print("[OK] CUDA operacie funguju spravne!")
                return True, True
            except Exception as e:
                print(f"[CHYBA] Chyba pri CUDA operaciach: {e}")
                return True, False
        else:
            print("[VAROVANIE] PyTorch nema CUDA podporu")
            return True, False

    except ImportError:
        print("[CHYBA] PyTorch nie je nainstalovany")
        return False, False
    except Exception as e:
        print(f"[CHYBA] Chyba pri kontrole PyTorch: {e}")
        return False, False

def check_cuda_toolkit():
    """Overí inštaláciu CUDA Toolkit"""
    print("\n" + "="*60)
    print("KONTROLA CUDA TOOLKIT")
    print("="*60)

    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] CUDA Toolkit je nainstalovany!")
            # Extraktujeme verziu
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"Verzia: {line.strip()}")
            return True
        else:
            print("[CHYBA] CUDA Toolkit nie je nainstalovany alebo nvcc nefunguje")
            return False
    except FileNotFoundError:
        print("[CHYBA] nvcc nie je dostupne (CUDA Toolkit nie je nainstalovany)")
        return False
    except Exception as e:
        print(f"[CHYBA] Chyba pri kontrole CUDA Toolkit: {e}")
        return False

def test_basic_ml_operations():
    """Test základných ML operácií"""
    print("\n" + "="*60)
    print("TEST ZÁKLADNÝCH ML OPERÁCIÍ")
    print("="*60)

    try:
        import torch
        import torch.nn as nn

        # Test na CPU
        print("Testovanie na CPU...")
        device_cpu = torch.device('cpu')
        x_cpu = torch.randn(1000, 1000)
        y_cpu = torch.randn(1000, 1000)

        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU čas: {cpu_time:.4f} sekúnd")

        # Test na GPU (ak je dostupná)
        if torch.cuda.is_available():
            print("Testovanie na GPU...")
            device_gpu = torch.device('cuda')
            x_gpu = torch.randn(1000, 1000).to(device_gpu)
            y_gpu = torch.randn(1000, 1000).to(device_gpu)

            # Warm-up
            torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()

            start_time = time.time()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            print(f"GPU čas: {gpu_time:.4f} sekúnd")

            speedup = cpu_time / gpu_time
            print(f"Zrýchlenie GPU vs CPU: {speedup:.2f}x")

            if speedup > 1.5:
                print("[OK] GPU poskytuje vyznamne zrychlenie!")
            else:
                print("[VAROVANIE] GPU zrychlenie je minimalne")

        return True
    except Exception as e:
        print(f"[CHYBA] Chyba pri testovani ML operacii: {e}")
        return False

def recommend_installation():
    """Poskytne odporúčania pre inštaláciu"""
    print("\n" + "="*60)
    print("ODPORÚČANIA PRE INŠTALÁCIU")
    print("="*60)

    print("Pre správnu funkcionalitu strojového učenia odporúčame:")
    print()
    print("1. NVIDIA GPU Driver:")
    print("   - Stiahnite najnovší driver z https://www.nvidia.com/drivers")
    print()
    print("2. CUDA Toolkit:")
    print("   - Stiahnite z https://developer.nvidia.com/cuda-downloads")
    print("   - Odporúčaná verzia: CUDA 11.8 alebo 12.x")
    print()
    print("3. PyTorch s CUDA podporou:")
    print("   - Navštívte https://pytorch.org/get-started/locally/")
    print("   - Vyberte správnu kombináciu OS/CUDA/Python")
    print("   - Príklad pre Windows + CUDA 11.8:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("4. Overenie inštalácie:")
    print("   - Spustite tento skript znova: python test_cuda.py")

def main():
    """Hlavná funkcia testovania"""
    print("CUDA A PYTORCH DIAGNOSTIKA")
    print("Testovanie pripravenosti systému pre strojové učenie")

    # Základné informácie o systéme
    check_system_info()

    # Kontrola NVIDIA GPU
    gpu_available = check_nvidia_gpu()

    # Kontrola CUDA Toolkit
    cuda_toolkit = check_cuda_toolkit()

    # Kontrola PyTorch
    pytorch_installed, pytorch_cuda = check_pytorch_installation()

    # Test ML operácií
    if pytorch_installed:
        ml_test = test_basic_ml_operations()
    else:
        ml_test = False

    # Súhrn
    print("\n" + "="*60)
    print("SÚHRN DIAGNOSTIKY")
    print("="*60)

    status_items = [
        ("NVIDIA GPU", "[OK]" if gpu_available else "[CHYBA]"),
        ("CUDA Toolkit", "[OK]" if cuda_toolkit else "[CHYBA]"),
        ("PyTorch", "[OK]" if pytorch_installed else "[CHYBA]"),
        ("PyTorch CUDA", "[OK]" if pytorch_cuda else "[CHYBA]"),
        ("ML operacie", "[OK]" if ml_test else "[CHYBA]")
    ]

    for item, status in status_items:
        print(f"{item:15} {status}")

    # Celkové hodnotenie
    all_good = all([gpu_available, cuda_toolkit, pytorch_installed, pytorch_cuda, ml_test])

    print("\n" + "="*60)
    if all_good:
        print("[USPECH] SYSTEM JE PRIPRAVENY PRE STROJOVE UCENIE!")
        print("Mozete pokracovat s Fazou 2 - zber dat.")
        return True
    elif pytorch_installed and pytorch_cuda:
        print("[VAROVANIE] SYSTEM JE CIASTOCNE PRIPRAVENY")
        print("PyTorch s CUDA funguje, ale mozu chybat niektore komponenty.")
        return True
    else:
        print("[CHYBA] SYSTEM NIE JE PRIPRAVENY")
        print("Potrebujete nainstalovat chybajuce komponenty.")
        recommend_installation()
        return False

    print("="*60)

if __name__ == "__main__":
    main()
