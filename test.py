import os

import logger

from deepseek_integration import genai

api_key = os.environ.get("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("Environmentálna premenná GEMINI_API_KEY nie je nastavená.")
logger.debug(f"API kľúč načten: {api_key[:10]}...")  # Logovanie pre kontrolu
genai.configure(api_key=api_key)