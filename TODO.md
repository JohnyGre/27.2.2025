# TODO: Vylepšenie rozpoznávania giest pomocou strojového učenia

Tento dokument popisuje kroky potrebné na integráciu modelu strojového učenia pre presnejšie a flexibilnejšie rozpoznávanie giest.

## Fáza 1: Príprava prostredia a overenie CUDA

- [x] **1. Inštalácia PyTorch s podporou CUDA:** Nainštalovať správnu verziu knižnice PyTorch, ktorá dokáže komunikovať s NVIDIA GPU.
- [x] **2. Overenie funkčnosti CUDA:** Spustiť testovací skript (`test_cuda.py`) na potvrdenie, že PyTorch vidí a môže používať GPU.

## Fáza 2: Zber dát pre trénovanie

- [x] **1. Vytvorenie skriptu na zber dát (`collect_data.py`):**
    - Skript bude používať `mediapipe` na detekciu kľúčových bodov ruky.
    - Umožní zadať názov gesta (napr. 'pest', 'otvorena_dlan').
    - Bude ukladať sekvencie kľúčových bodov do súborov (napr. CSV alebo NumPy) pre každé gesto.
- [ ] **2. Zozbieranie dát:**
    - Spustiť skript a nahrať niekoľko stoviek vzoriek pre každé gesto, ktoré chceme rozpoznávať.

## Fáza 3: Trénovanie modelu strojového učenia

- [x] **1. Vytvorenie skriptu na trénovanie (`train_model.py`):**
    - Načíta dáta zozbierané v Fáze 2.
    - Zadefinuje architektúru neurónovej siete (napr. jednoduchá viacvrstvová sieť - MLP).
    - Implementuje trénovaciu slučku, ktorá bude využívať GPU (CUDA) na urýchlenie výpočtov.
    - Uloží natrénovaný model do súboru (napr. `gesture_model.pth`).
- [x] **2. Spustenie trénovania:**
    - Spustiť skript a nechať model natrénovať sa na dátach.
    - Sledovať presnosť a stratu (loss) na overenie úspešnosti trénovania.

## Fáza 4: Integrácia modelu do hlavnej aplikácie

- [x] **1. Úprava `gesture_recognition.py`:**
    - Načítať natrénovaný model (`gesture_model.pth`).
    - V reálnom čase získavať kľúčové body z `mediapipe`.
    - Tieto body posielať modelu na predikciu gesta.
    - Namiesto pevne napísanej logiky používať výstup z modelu na určenie aktuálneho gesta.
- [x] **2. Prepojenie s `main.py` alebo `SmartAssistant.py`:**
    - Zabezpečiť, aby rozpoznané gestá spúšťali požadované akcie v hlavnej aplikácii (napr. ovládanie kurzora, hlasitosti).
