import json
import time
import os
import re
from typing import Dict, Optional, Callable
import speech_recognition as sr
import pyautogui
import webbrowser
from gtts import gTTS
from playsound3 import playsound
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from queue import Queue
from deepseek_integration import GeminiAPI
from screen_reader import process_voice_command
import logging
import pygetwindow as gw

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Cursor:
    """Trieda na ovládanie kurzora a spracovanie hlasových príkazov."""

    def __init__(self, air_cursor, config_path: str = "config.json"):
        self.running = True
        self.mic_enabled = True
        self.headphones_enabled = True
        self.air_cursor = air_cursor
        self.config = self._load_config(config_path)
        self.commands = self.config["voice"]["commands"]

        # Inicializácia Gemini API
        try:
            self.gemini = GeminiAPI()
            logger.info("GeminiAPI inicializované.")
        except ValueError as e:
            logger.warning(f"Gemini API kľúč nie je nakonfigurovaný: {e}. Gemini funkcie budú deaktivované.")
            self.gemini = None

        # Inicializácia ovládania hlasitosti
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

        # Mapovanie príkazov na funkcie
        self.command_map: Dict[str, tuple[Callable, str]] = {
            "toggle_cursor": (self.toggle_air_cursor, "Kurzor prepnutý."),
            "exit": (self._exit, "Aplikácia sa ukončuje."),
            "click": (self.click, "Kurzor klikol."),
            "right_click": (self.right_click, "Kurzor klikol pravým tlačidlom."),
            "scroll_up": (self.scroll_up, "Posunutie hore."),
            "scroll_down": (self.scroll_down, "Posunutie dole."),
            "double_click": (self.double_click, "Dvojklik."),
            "zoom_in": (self.zoom_in, "Okno zväčšené."),
            "zoom_out": (self.zoom_out, "Okno zmenšené."),
            "save_file": (self.save_file, "Súbor uložený."),
            "undo": (self.undo, "Vrátené späť."),
            "copy": (self.copy, "Text skopírovaný."),
            "paste": (self.paste, "Text vložený."),
            "switch_application": (self.switch_application, "Aplikácia prepnutá."),
            "close_window": (self.close_window, "Okno zatvorené."),
            "refresh_page": (self.refresh_page, "Stránka obnovená."),
            "volume_up": (self.volume_up, "Hlasitosť zvýšená."),
            "volume_down": (self.volume_down, "Hlasitosť znížená."),
            "open_youtube": (lambda: self.open_webpage("https://www.youtube.com"), "YouTube bol otvorený."),
            "open_google": (lambda: self.open_webpage("https://www.google.com"), "Google bol otvorený."),
            "open_facebook": (lambda: self.open_webpage("https://www.facebook.com"), "Facebook bol otvorený."),
            "open_twitter": (lambda: self.open_webpage("https://www.twitter.com"), "Twitter bol otvorený."),
            "open_notepad": (lambda: os.system("notepad"), "Notepad bol otvorený."),
            "open_calculator": (lambda: os.system("calc"), "Kalkulačka bola otvorená."),
            "web_back": (self.web_back, "Návrat na predchádzajúcu stránku."),
        }

    def toggle_air_cursor(self):
        """Prepne stav sledovania kurzora."""
        is_enabled = self.air_cursor.toggle_tracking()
        status = "zapnuté" if is_enabled else "vypnuté"
        logger.info(f"Sledovanie kurzora bolo {status}.")
        return is_enabled

    def _load_config(self, config_path: str) -> Dict:
        """Načíta konfiguráciu zo súboru."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Chyba pri načítaní konfigurácie: {e}")
            raise

    def _exit(self):
        """Ukončí beh aplikácie."""
        self.running = False

    def click(self):
        """Vykoná kliknutie ľavým tlačidlom myši."""
        time.sleep(0.5)
        pyautogui.click()

    def right_click(self):
        """Vykoná kliknutie pravým tlačidlom myši."""
        time.sleep(0.5)
        pyautogui.rightClick()

    def scroll_up(self):
        """Posunie obsah nahor."""
        pyautogui.scroll(self.config["mouse"]["scroll_amount_up"])

    def scroll_down(self):
        """Posunie obsah nadol."""
        pyautogui.scroll(self.config["mouse"]["scroll_amount_down"])

    def double_click(self):
        """Vykoná dvojklik."""
        pyautogui.doubleClick()

    def zoom_in(self):
        """Maximalizuje okno."""
        pyautogui.hotkey("win", "up")

    def zoom_out(self):
        """Minimalizuje okno."""
        pyautogui.hotkey("win", "down")

    def web_back(self):
        """Vráti sa na predchádzajúcu webovú stránku."""
        pyautogui.hotkey("alt", "left")

    def save_file(self):
        """Uloží súbor."""
        pyautogui.hotkey("ctrl", "s")

    def undo(self):
        """Vráti poslednú akciu späť."""
        pyautogui.hotkey("ctrl", "z")

    def copy(self):
        """Skopíruje vybraný text."""
        pyautogui.hotkey("ctrl", "c")

    def paste(self):
        """Vloží skopírovaný text."""
        pyautogui.hotkey("ctrl", "v")

    def switch_application(self):
        """Prepnutie medzi aplikáciami."""
        pyautogui.hotkey("alt", "tab")

    def close_window(self):
        """Zatvorí aktuálne okno."""
        pyautogui.hotkey("alt", "f4")

    def refresh_page(self):
        """Obnoví stránku."""
        pyautogui.hotkey("f5")

    def volume_up(self):
        """Zvýši hlasitosť."""
        current = self.volume.GetMasterVolumeLevelScalar()
        self.volume.SetMasterVolumeLevelScalar(min(1.0, current + 0.1), None)

    def volume_down(self):
        """Zníži hlasitosť."""
        current = self.volume.GetMasterVolumeLevelScalar()
        self.volume.SetMasterVolumeLevelScalar(max(0.0, current - 0.1), None)

    def volume_set(self, value: int):
        """Nastaví hlasitosť na konkrétnu hodnotu."""
        volume_level = max(0.0, min(1.0, value / 100.0))
        self.volume.SetMasterVolumeLevelScalar(volume_level, None)

    def open_webpage(self, url: str) -> bool:
        """Otvorí webovú stránku."""
        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.error(f"Chyba pri otváraní stránky {url}: {e}")
            return False

    def open_application(self, app_name: str) -> bool:
        """Otvorí aplikáciu alebo web podľa názvu."""
        app_map = {
            "notepad": "notepad.exe",
            "kalkulačka": "calc.exe",
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "facebook": "https://www.facebook.com",
            "twitter": "https://www.twitter.com",
        }
        app = app_map.get(app_name.lower())
        if not app:
            return False
        if app.startswith("http"):
            return self.open_webpage(app)
        try:
            os.startfile(app)
            return True
        except Exception as e:
            logger.error(f"Chyba pri otváraní aplikácie {app_name}: {e}")
            return False


import pygetwindow as gw

def voice_command_listener(cursor: Cursor, queue: Queue):
    """Spracováva hlasové príkazy s jasne oddelenou logikou."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while cursor.running:
        if not cursor.mic_enabled:
            time.sleep(0.1)
            continue

        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio, language="sk-SK").lower().strip()
            queue.put(("stt", command))
            logger.info(f"Rozpoznaný príkaz: {command}")

            command_handled = False
            # 1. Lokálne príkazy s priamou akciou (najvyššia priorita)
            for cmd_key, (action, response) in cursor.command_map.items():
                if any(cmd_word in command for cmd_word in cursor.commands.get(cmd_key, [])):
                    action()
                    queue.put(("tts", response))
                    command_handled = True
                    break
            
            if command_handled:
                continue

            # 2. Komplexné lokálne príkazy (vyžaduj��ce parametre z príkazu)
            if "volume_set" in cursor.commands and any(cmd in command for cmd in cursor.commands["volume_set"]):
                if match := re.search(r"\b(\d{1,3})\b", command):
                    value = int(match.group(1))
                    cursor.volume_set(value)
                    queue.put(("tts", f"Hlasitosť nastavená na {value}%."))
                    command_handled = True
            elif "open_application" in cursor.commands and any(cmd in command for cmd in cursor.commands["open_application"]):
                parts = command.split()
                if len(parts) > 1:
                    app_name = " ".join(parts[1:])
                    success = cursor.open_application(app_name)
                    queue.put(("tts", f"Aplikácia '{app_name}' {'otvorená' if success else 'nenájdená'}."))
                    command_handled = True

            if command_handled:
                continue

            # 3. OCR a Gemini (najnižšia priorita)
            # Získanie aktívneho okna pre OCR
            active_window = gw.getActiveWindow()
            region = None
            if active_window and active_window.width > 0 and active_window.height > 0:
                region = (active_window.left, active_window.top, active_window.width, active_window.height)

            success, message = process_voice_command(command, region=region)
            if success:
                queue.put(("tts", message))
            elif cursor.gemini:
                try:
                    response = cursor.gemini.process_command(command)
                    if response["status"] == "success":
                        message = response["message"]
                        if message.get("type") == "action":
                            action_info = message.get("action", {})
                            action_name = action_info.get("name")
                            action_params = action_info.get("parameters", {})
                            if action_name and hasattr(cursor, action_name):
                                getattr(cursor, action_name)(**action_params)
                                queue.put(("tts", message.get("response", "Akcia vykonaná.")))
                            else:
                                queue.put(("tts", "Nerozumiem príkazu na akciu."))
                        elif message.get("type") == "answer":
                            queue.put(("tts", message.get("answer", "Nenašla som odpoveď.")))
                        else:
                            queue.put(("tts", "Dostala som neznámy typ odpovede."))
                    else:
                        queue.put(("tts", response.get("message", "Nepodarilo sa spracovať príkaz.")))
                except Exception as e:
                    logger.error(f"Chyba pri spracovaní Gemini odpovede: {e}")
                    queue.put(("tts", "Chyba pri spracovaní príkazu."))
            else:
                queue.put(("tts", "Príkaz nebol rozpoznaný."))

        except sr.UnknownValueError:
            pass  # Ignorovať nerozpoznané príkazy
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            logger.error(f"Chyba pri spracovaní hlasu: {e}")
            queue.put(("tts", "Nastala chyba."))


if __name__ == "__main__":
    cursor = Cursor()
    q = Queue()
    voice_command_listener(cursor, q)