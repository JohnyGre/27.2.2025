import pytesseract
from PIL import ImageGrab, Image
import pyautogui
import speech_recognition as sr
import webbrowser
import os
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import cv2

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Konštanta pre rozpoznávanie webových prvkov
WEB_ELEMENTS = {
    "link": ["odkaz", "link", "prepojenie", "klikni na odkaz", "otvor odkaz"],
    "button": ["tlačidlo", "button", "klikni na tlačidlo", "odoslať", "hľadať"],
    "search_field": ["vyhľadávanie", "hľadať", "search", "vstup", "pole", "vyhľadaj na"],
}


@lru_cache(maxsize=1024)
def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity between two strings with caching."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def read_text_from_screen(region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[dict, List[Tuple[str, int]], int]:
    """
    Read text from a specified region of the screen with optimized image processing.
    If no region is specified, it captures the entire screen.
    """
    try:
        # Určenie oblasti snímania
        bbox = region or (0, 0, pyautogui.size().width, pyautogui.size().height)
        screen = ImageGrab.grab(bbox=bbox)

        # Konverzia na numpy pole pre OpenCV
        screen_np = np.array(screen)

        # Prevod na odtiene sivej
        gray_screen = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

        # Aplikácia prahovania (binarizácia) na zlepšenie kontrastu
        # THRESH_OTSU automaticky nájde optimálnu hodnotu prahu
        _, thresh_screen = cv2.threshold(gray_screen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Konverzia späť na PIL Image pre Tesseract
        processed_image = Image.fromarray(thresh_screen)

        # Zmena veľkosti pre lepšiu presnosť (ak je to potrebné)
        resize_factor = 1 # V tomto prípade už nepotrebujeme meniť veľkosť, lebo predspracovanie to rieši
        
        primary_lang = 'slk'
        data = pytesseract.image_to_data(
            processed_image,
            output_type=pytesseract.Output.DICT,
            lang=primary_lang,
            config='--psm 6'  # Predpokladáme jeden blok textu
        )

        valid_words = [(word, i) for i, word in enumerate(data['text']) if word and word.strip() and len(word) > 1]
        logger.debug(f"Extracted {len(valid_words)} valid words from screen.")
        return data, valid_words, resize_factor
    except Exception as e:
        logger.error(f"Error reading text from screen: {e}")
        return {}, [], 1


def find_text_position(data: dict, spoken_text: str, valid_words: List[Tuple[str, int]]) -> Tuple[
    Optional[Tuple[int, int, int, int]], Optional[str]]:
    """Find text position with optimized matching algorithm."""
    best_match = None
    best_ratio = 0.8
    spoken_text = spoken_text.strip().lower()

    for word, idx in valid_words:
        if word.lower() == spoken_text:
            best_match = idx
            break

    if best_match is None:
        for word, idx in valid_words:
            ratio = similarity_ratio(spoken_text, word)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = idx

    if best_match is not None:
        position = (
        data['left'][best_match], data['top'][best_match], data['width'][best_match], data['height'][best_match])
        logger.info(f"Found match: '{data['text'][best_match]}' at position {position}")
        return position, data['text'][best_match]
    logger.warning(f"No match found for '{spoken_text}'")
    return None, None


def move_cursor_to_text_position(position: Tuple[int, int, int, int], matched_text: str, resize_factor: int = 1,
                                 command: str = "", region_offset: Tuple[int, int] = (0, 0)) -> bool:
    """Move cursor to text position and interact with elements, considering region offset."""
    try:
        if not position:
            return False

        x, y, w, h = position
        # Prirátanie offsetu regiónu k pozícii
        center_x = region_offset[0] + x + w // 2
        center_y = region_offset[1] + y + h // 2

        # Ošetrenie hraníc obrazovky
        screen_width, screen_height = pyautogui.size()
        center_x = max(1, min(center_x, screen_width - 1))
        center_y = max(1, min(center_y, screen_height - 1))

        pyautogui.moveTo(center_x, center_y, duration=0.3)
        logger.debug(f"Kurzor presunutý na: ({center_x}, {center_y})")

        text_lower = matched_text.lower()
        command_lower = command.lower()

        # Detekcia URL
        url_pattern = re.compile(r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})')
        if url_pattern.match(text_lower) or any(ext in text_lower for ext in ['.com', '.sk', '.org']):
            url = matched_text if text_lower.startswith(('http://', 'https://')) else f'http://{matched_text}'
            webbrowser.open(url)
            logger.info(f"Opened URL: {url}")
            return True

        # Detekcia súborov
        file_extensions = ['.txt', '.doc', '.docx', '.pdf', '.xlsx', '.exe', '.jpg', '.png']
        if any(text_lower.endswith(ext) for ext in file_extensions) or os.path.exists(matched_text):
            try:
                os.startfile(matched_text)
                logger.info(f"Opened file: {matched_text}")
                return True
            except (FileNotFoundError, OSError) as e:
                logger.error(f"Error opening file '{matched_text}': {e}")
                return False

        # Interakcia s webovými prvkami cez OCR
        if any(keyword in command_lower for keyword in WEB_ELEMENTS["link"]):
            pyautogui.click()
            logger.info(f"Clicked on link: '{matched_text}'")
            return True
        elif any(keyword in command_lower for keyword in WEB_ELEMENTS["button"]):
            pyautogui.click()
            logger.info(f"Clicked on button: '{matched_text}'")
            return True
        elif any(keyword in command_lower for keyword in WEB_ELEMENTS["search_field"]):
            pyautogui.click()
            import time
            time.sleep(0.5)  # Pauza na zaistenie focusu poľa
            search_text = " ".join(command_lower.split()[1:]) if len(command_lower.split()) > 1 else ""
            if "vyhľadaj na" in command_lower or "hľadať na" in command_lower:
                parts = command_lower.split()
                search_text = " ".join(parts[parts.index("na") + 1:]).strip() if "na" in parts else search_text
            if search_text:
                pyautogui.typewrite(search_text)
                pyautogui.press("enter")
                logger.info(f"Entered '{search_text}' into search field: '{matched_text}'")
            else:
                logger.warning("No search text provided for search field")
            return True

        # Default: dvojklik
        pyautogui.doubleClick()
        logger.info(f"Performed double-click on '{matched_text}'")
        return True
    except Exception as e:
        logger.error(f"Error moving cursor to position: {e}")
        return False


def process_voice_command(command: str, region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, str]:
    """Process voice command using OCR within a specified region."""
    if not command or len(command.strip()) < 2:
        logger.warning("Command too short or empty")
        return False, "Príkaz je príliš krátky alebo prázdny"

    command_lower = command.lower()
    data, valid_words, resize_factor = read_text_from_screen(region=region)
    
    region_offset = (region[0], region[1]) if region else (0, 0)

    # Pokus o rozpoznanie celého príkazu
    target_text = command_lower.strip()
    position, matched_text = find_text_position(data, target_text, valid_words)

    # Ak celý príkaz zlyhá, pokus o čiastočnú zhodu s kľúčovými slovami
    if not position:
        # Extrakcia cieľového textu pre vyhľadávanie
        if "vyhľadaj na" in command_lower or "hľadať na" in command_lower:
            parts = command_lower.split()
            if "na" in parts:
                target_text = " ".join(parts[parts.index("na") + 1:]).strip()
            else:
                target_text = " ".join(parts[1:]).strip()
        else:
            # Pokus o hľadanie jednotlivých slov
            for word in command_lower.split():
                position, matched_text = find_text_position(data, word, valid_words)
                if position:
                    break
            else:
                target_text = matched_text = None

    if position and matched_text:
        success = move_cursor_to_text_position(position, matched_text, resize_factor, command, region_offset)
        return success, f"Našiel som a interagoval som s '{matched_text}'"
    return False, "Nepodarilo sa nájsť zodpovedajúci text na obrazovke"


def listen_for_command() -> Optional[str]:
    """Listens for a single voice command."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        logger.info("Počúvam...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            command = r.recognize_google(audio, language="sk-SK")
            logger.info(f"Rozpoznaný príkaz: {command}")
            return command
        except sr.UnknownValueError:
            logger.warning("Nerozumel som príkazu")
            return None
        except sr.RequestError as e:
            logger.error(f"Chyba pri komunikácii s Google Speech Recognition: {e}")
            return None
        except Exception as e:
            logger.error(f"Vyskytla sa neočakávaná chyba pri počúvaní: {e}")
            return None


def main():
    """Main function with improved error handling and threading."""
    max_retries = 3
    for attempt in range(max_retries):
        command = listen_for_command()
        if command:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(process_voice_command, command)
                try:
                    success, message = future.result(timeout=10)
                    logger.info(message)
                    if success:
                        break
                except TimeoutError:
                    logger.error("Processing command took too long")
                    continue
        elif attempt < max_retries - 1:
            logger.info(f"Trying again... (attempt {attempt + 2}/{max_retries})")
        else:
            logger.error("Maximum number of attempts reached")


if __name__ == "__main__":
    main()
