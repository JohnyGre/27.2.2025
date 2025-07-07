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
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("GeminiAPI initialized.")

    def process_command(self, command: str) -> Dict[str, str]:
        """Spracuje príkaz a vráti odpoveď."""
        is_question = bool(re.search(r"^(ako|čo|prečo|kedy|kto|kde|aký|aká|aké|akí|koľko|môžem|má)\b", command.lower()))
        prompt = self._build_prompt(command, is_question)

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            logger.debug(f"Raw Gemini response: {response_text}")

            if is_question:
                return {"message": response_text[:200], "status": "success"}  # Limit na 2 vety približne
            else:
                json_block = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_block:
                    json_data = json.loads(json_block.group(0))
                    if "intent" in json_data and "action_command" in json_data:
                        return {"message": json_data, "status": "success"}
                    raise ValueError("Invalid JSON response: missing required keys")
                raise ValueError("No JSON block found in response")

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"message": "Nepodarilo sa spracovať požiadavku.", "status": "error", "error": str(e)}

    def _build_prompt(self, command: str, is_question: bool) -> str:
        """Vytvorí prompt pre Gemini API."""
        base_instructions = """
            [INŠTRUKCIE PRE ASISTENTA]
            1. Rozumieš hlasovým príkazom používateľa v slovenčine.
            2. **Ak používateľ položí otázku (začínajúcu sa napr. "ako", "čo", "prečo", atď.) alebo ak používateľský príkaz patrí do kategórie "ask" (pozri konfiguračný súbor pre zoznam kľúčových slov pre "ask" príkazy), odpovedz stručne a fakticky v slovenčine v maximálne dvoch vetách.  Zameraj sa na poskytnutie priamej odpovede a *neotváraj webovú stránku* pre tieto typy príkazov.**
            3. Pre *ostatné príkazy*, urči *zámer* používateľa a *akciu*, ktorú má systém vykonať. Odpovedaj v JSON formáte s kľúčmi "intent" a "action_command".
            4. "intent" je stručný popis zámeru v slovenčine.
            5. "action_command" je JSON objekt s kľúčmi "name" a "parameters". "name" je názov akcie (metóda triedy `Cursor`), "parameters" sú parametre akcie v JSON formate. Ak akcia nepotrebuje parametre, "parameters" musí byť prázdny objekt {{}}.
            6. **Pre webové vyhľadávanie (ak je *explicitne* požadované iným typom príkazu, napr. "search web for ..."), použi "intent": "vyhľadávanie na webe" a "action_command": {{ "name": "open_webpage", "parameters": {{ "url": "URL_VYHĽADÁVANIA" }} }}. Pre otázky NEPOUŽÍVAJ vyhľadávanie na webe.**
            7. **Ak používateľ povie "search web for [hľadaný výraz]" alebo niečo podobné, intent bude "vyhľadávanie na webe" a action_command bude {{ "name": "open_webpage", "parameters": {{ "url": "https://www.google.com/search?q=https://www.urlencoder.org/enc/hyphen%27s/" }} }}.**
            8. Ak nerozumieš, odpovedz s intentom "unknown" a `action_command`: {{ "name": null, "parameters": {{}} }}.
            9. Odpovedaj v slovenčine pre "intent", ale JSON pre "action_command" musí byť v angličtine.
            10. **Pre všetky príkazy, ktoré nie sú otázkou, odpovedaj v JSON formate.**
        """

        return base_instructions.format(command=command) if is_question else base_instructions.format(command=command) + "\nVráť odpoveď vo formáte JSON."

if __name__ == "__main__":
    os.environ["GEMINI_API_KEY"] = "test_key"  # Pre testovanie
    gemini = GeminiAPI()
    print(gemini.process_command("Čo je Python?"))
