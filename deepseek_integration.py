import os
import re
import json
import logging
from typing import Dict, Optional
import google.generativeai as genai

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GeminiAPI:
    """API klient pre Gemini model."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable GEMINI_API_KEY is not set.")
        genai.configure(api_key=self.api_key)
        # Použijeme novší a schopnejší model, ak je dostupný
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("GeminiAPI initialized with model gemini-1.5-flash.")

    def process_command(self, command: str) -> Dict[str, any]:
        """Spracuje príkaz a vráti štruktúrovanú odpoveď."""
        prompt = self._build_prompt(command)
        logger.debug(f"Sending prompt to Gemini: {prompt}")

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            logger.debug(f"Raw Gemini response: {response_text}")

            # Pokus o extrakciu JSON bloku z odpovede
            json_match = re.search(r"```json\s*({.*?})\s*```", response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r"({.*?})", response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                try:
                    json_data = json.loads(json_str)
                    # Validácia, či má JSON očakávanú štruktúru
                    if "type" in json_data and ("action" in json_data or "answer" in json_data):
                        return {"status": "success", "message": json_data}
                    else:
                        raise ValueError("Invalid JSON structure from model.")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON from model: {e}\nResponse was: {json_str}")
                    return {"status": "error", "message": "Model vrátil neplatný JSON."}
            else:
                # Ak model vráti len text, považujeme to za odpoveď na otázku
                logger.warning("No JSON block found in response, treating as a simple answer.")
                return {"status": "success", "message": {"type": "answer", "answer": response_text}}

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"status": "error", "message": f"Chyba pri komunikácii s modelom: {e}"}

    def _build_prompt(self, command: str) -> str:
        """
        Vytvorí jasný a štruktúrovaný prompt pre Gemini API.
        """
        # Systémový prompt, ktorý dáva modelu kontext a inštrukcie
        system_prompt = """
            Tvojou úlohou je byť AI asistent v desktopovej aplikácii. Analyzuj požiadavku používateľa a VŽDY odpovedz v JSON formáte.
            JSON musí obsahovať kľúč "type", ktorý môže mať hodnotu "action" alebo "answer".

            1.  Ak je požiadavka príkaz na vykonanie akcie (napr. "otvor youtube", "klikni", "zvýš hlasitosť"), JSON bude vyzerať takto:
                {
                    "type": "action",
                    "action": {
                        "name": "názov_funkcie_v_kóde",
                        "parameters": { "názov_parametra": "hodnota" }
                    },
                    "response": "Stručná odpoveď pre používateľa v slovenčine."
                }
                - "name" musí byť platný názov funkcie: 'open_webpage', 'click', 'double_click', 'scroll_up', 'scroll_down', 'volume_up', 'volume_down', 'volume_set', 'open_application'.
                - Pre 'open_application' použi parameter "app_name".
                - Ak akcia nemá parametre, "parameters" je prázdny objekt {}.
                - Príklad pre "otvor youtube": {"type": "action", "action": {"name": "open_webpage", "parameters": {"url": "https://www.youtube.com"}}, "response": "Otváram YouTube."}
                - Príklad pre "klikni": {"type": "action", "action": {"name": "click", "parameters": {}}, "response": "Klikol som."}

            2.  Ak je požiadavka otázka (napr. "čo je hlavné mesto slovenska", "kto napísal hamlet"), JSON bude vyzerať takto:
                {
                    "type": "answer",
                    "answer": "Stručná a vecná odpoveď na otázku v slovenčine."
                }
                - Príklad pre "čo je python": {"type": "answer", "answer": "Python je interpretovaný programovací jazyk na vysokej úrovni pre všeobecné programovanie."}

            Analyzuj nasledujúcu požiadavku a vygeneruj príslušný JSON.
        """

        # Spojenie systémového promptu a používateľskej požiadavky
        full_prompt = f'''{system_prompt}

Požiadavka používateľa: "{command}"'''
        return full_prompt

if __name__ == "__main__":
    # Tento blok sa spustí, len ak sa súbor spúšťa priamo
    # Vyžaduje nastavenie API kľúča v premenných prostredia
    try:
        gemini = GeminiAPI()
        
        # Test 1: Otázka
        print("\n--- Test 1: Otázka ---")
        question_result = gemini.process_command("Čo je hlavné mesto Slovenska?")
        print(json.dumps(question_result, indent=2, ensure_ascii=False))

        # Test 2: Akcia bez parametrov
        print("\n--- Test 2: Akcia bez parametrov ---")
        action_result = gemini.process_command("dvojklik")
        print(json.dumps(action_result, indent=2, ensure_ascii=False))

        # Test 3: Akcia s parametrami
        print("\n--- Test 3: Akcia s parametrami ---")
        action_param_result = gemini.process_command("otvor google")
        print(json.dumps(action_param_result, indent=2, ensure_ascii=False))

    except ValueError as e:
        print(f"Chyba: {e}")
    except Exception as e:
        print(f"Vyskytla sa neočakávaná chyba: {e}")
