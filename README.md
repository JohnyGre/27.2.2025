
# AI Asistent ovládaný hlasom a gestami

Tento projekt je desktopový asistent, ktorý kombinuje rozpoznávanie hlasu, spracovanie prirodzeného jazyka a počítačové videnie na ovládanie kurzora a vykonávanie príkazov.

## Funkcie

- **Ovládanie kurzora gestami:** Pohybujte kurzorom myši pomocou prsta snímaného kamerou.
- **Hlasové príkazy:** Ovládajte rôzne funkcie asistenta pomocou hlasu.
- **Text-to-Speech (TTS):** Asistent vám odpovedá hlasom.
- **Integrácia s Gemini:** Využíva generatívny model Gemini na odpovede na otázky.

## Inštalácia

1.  **Klonujte repozitár:**
    ```bash
    git clone https://github.com/JohnyGre/27.2.2025.git
    cd 27.2.2025
    ```

2.  **Vytvorte a aktivujte virtuálne prostredie:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Na Linuxe/macOS
    .venv\Scripts\activate  # Na Windows
    ```

3.  **Nainštalujte závislosti:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Nastavte API kľúč:**
    - Premenujte súbor `config.example.json` na `config.json`.
    - Vložte svoj Gemini API kľúč do súboru `config.json`.

## Použitie

1.  **Spustite aplikáciu:**
    ```bash
    python main.py
    ```

2.  **Kalibrácia:**
    - Po spustení sa otvorí okno s obrazom z kamery.
    - Nasmerujte prst na štyri rohy oblasti, ktorú chcete používať na ovládanie kurzora.
    - Aplikácia automaticky zdetekuje rohy a nakreslí okolo nich obdĺžnik.
    - Ak automatická kalibrácia zlyhá, stlačte klávesu `a`.
    - Na resetovanie kalibrácie stlačte klávesu `r`.

3.  **Ovládanie:**
    - Po úspešnej kalibrácii môžete pohybovať kurzorom pomocou prsta.
    - Hlasové príkazy sú aktívne počas behu aplikácie.

## Hlasové príkazy

- **"ahoj"**: Asistent odpovie pozdravom.
- **"kurzor"**: Aktivuje alebo deaktivuje ovládanie kurzora.
- **"klik"**: Vykoná kliknutie myšou.
- **"dvojklik"**: Vykoná dvojité kliknutie myšou.
- **"prečítaj text"**: Prečíta text z obrazovky.
- **"napíš <text>"**: Napíše zadaný text.
- **"otvor <aplikácia>"**: Otvorí zadanú aplikáciu (napr. "otvor poznámkový blok").
- **"zatvor"**: Zatvorí aktívne okno.
- **"koniec"**: Ukončí aplikáciu.
- **"opýtaj sa <otázka>"**: Položí otázku modelu Gemini a odpoveď prehrá ako reč.
