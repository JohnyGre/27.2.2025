#!/usr/bin/env python3
"""
Rozšírená hlavná aplikácia s ML gesture recognition
Fáza 4 - Úloha 2: Integrácia do hlavnej aplikácie
"""

import tkinter as tk
import threading
import cv2
from queue import Queue
# Odstránené multiprocessing import - používame threading
from air_cursor import AirCursor
from voice_commands import Cursor, voice_command_listener
from config_manager import ConfigManager
from gtts import gTTS
from playsound3 import playsound
import os
from PIL import Image, ImageTk
import logging
from pathlib import Path
import time
import pyautogui

# Vypnutie PyAutoGUI fail-safe pre lepšie fungovanie
pyautogui.FAILSAFE = False

# Import ML gesture recognition
try:
    from gesture_recognition_ml import ml_gesture_recognition_process, MLGestureRecognizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML gesture recognition nie je dostupný")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def speak(text: str, lang: str = "sk", headphones_enabled: bool = True):
    """Prehrá text ako reč, ak sú slúchadlá povolené."""
    if not headphones_enabled:
        return
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        logger.error(f"Chyba pri prehrávaní TTS: {e}")

def run_cursor_tracking(air_cursor: AirCursor, cap: cv2.VideoCapture, queue: Queue):
    """Spustí sledovanie kurzora."""
    while air_cursor.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        if air_cursor.tracking_enabled:
            finger_pos = air_cursor.process_frame(frame)

            if finger_pos and air_cursor.calibrated:
                air_cursor.update_cursor(finger_pos)

            if not air_cursor.calibrated:
                if finger_pos:
                    cv2.putText(frame, f"Kalibrácia {len(air_cursor.calibration_sets) + 1}/3 - Bod {len(air_cursor.corners) + 1}/4",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Sledovanie kurzora je vypnuté", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("AirCursor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            air_cursor.running = False
            break
        elif key == ord('a'):
            air_cursor.auto_calibrate()
        elif key == ord('r'):
            air_cursor.reset_calibration()

    cap.release()
    cv2.destroyAllWindows()

class ModernGUI:
    """Moderné GUI s ML gesture recognition podporou."""
    
    def __init__(self, root, queue, cursor, air_cursor):
        self.root = root
        self.queue = queue
        self.cursor = cursor
        self.air_cursor = air_cursor
        
        # Stav aplikácie
        self.microphone_enabled = True
        self.headphones_enabled = True
        self.gesture_recognition_enabled = False
        self.ml_gesture_enabled = False
        
        # ML gesture recognition process
        self.gesture_process = None
        self.gesture_queue = Queue()
        
        self.setup_gui()
        self.process_queue()
        
        # Spustenie gesture recognition monitoring
        if ML_AVAILABLE:
            self.start_gesture_monitoring()
    
    def setup_gui(self):
        """Nastavenie GUI rozhrania."""
        self.root.title("AirCursor Smart Assistant s ML Gesture Recognition")
        self.root.geometry("800x700")
        self.root.configure(bg="#2c3e50")
        
        # Hlavný frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Nadpis
        title_label = tk.Label(main_frame, text="🤖 AirCursor Smart Assistant", 
                              font=("Arial", 24, "bold"), fg="#ecf0f1", bg="#2c3e50")
        title_label.pack(pady=(0, 20))
        
        # Ovládacie tlačidlá
        controls_frame = tk.Frame(main_frame, bg="#2c3e50")
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Mikrofón tlačidlo
        self.mic_button = tk.Button(controls_frame, text="🎤 Mikrofón ON", 
                                   command=self.toggle_microphone,
                                   font=("Arial", 12, "bold"), fg="white", bg="#27ae60",
                                   activebackground="#2ecc71", relief=tk.FLAT, padx=20, pady=10)
        self.mic_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Slúchadlá tlačidlo
        self.headphones_button = tk.Button(controls_frame, text="🎧 Slúchadlá ON", 
                                          command=self.toggle_headphones,
                                          font=("Arial", 12, "bold"), fg="white", bg="#3498db",
                                          activebackground="#5dade2", relief=tk.FLAT, padx=20, pady=10)
        self.headphones_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Kurzor tlačidlo
        self.cursor_button = tk.Button(controls_frame, text="👆 Kurzor OFF", 
                                      command=self.toggle_cursor,
                                      font=("Arial", 12, "bold"), fg="white", bg="#e74c3c",
                                      activebackground="#ec7063", relief=tk.FLAT, padx=20, pady=10)
        self.cursor_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # ML Gesture Recognition tlačidlo
        if ML_AVAILABLE:
            self.gesture_button = tk.Button(controls_frame, text="🤲 ML Gestá OFF", 
                                           command=self.toggle_ml_gestures,
                                           font=("Arial", 12, "bold"), fg="white", bg="#9b59b6",
                                           activebackground="#bb8fce", relief=tk.FLAT, padx=20, pady=10)
            self.gesture_button.pack(side=tk.LEFT)
        
        # Informačný panel
        info_frame = tk.LabelFrame(main_frame, text="📊 Informácie o systéme", 
                                  font=("Arial", 14, "bold"), fg="#ecf0f1", bg="#34495e",
                                  labelanchor="n", relief=tk.FLAT, bd=2)
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Systémové informácie
        self.info_text = tk.Text(info_frame, height=8, font=("Consolas", 10), 
                                bg="#2c3e50", fg="#ecf0f1", relief=tk.FLAT, bd=0)
        self.info_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Gesture Recognition panel
        if ML_AVAILABLE:
            gesture_frame = tk.LabelFrame(main_frame, text="🤲 ML Gesture Recognition", 
                                         font=("Arial", 14, "bold"), fg="#ecf0f1", bg="#34495e",
                                         labelanchor="n", relief=tk.FLAT, bd=2)
            gesture_frame.pack(fill=tk.X, pady=(0, 20))
            
            self.gesture_text = tk.Text(gesture_frame, height=6, font=("Consolas", 10), 
                                       bg="#2c3e50", fg="#ecf0f1", relief=tk.FLAT, bd=0)
            self.gesture_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Hlasové príkazy panel
        speech_frame = tk.LabelFrame(main_frame, text="🎤 Hlasové príkazy", 
                                    font=("Arial", 14, "bold"), fg="#ecf0f1", bg="#34495e",
                                    labelanchor="n", relief=tk.FLAT, bd=2)
        speech_frame.pack(fill=tk.BOTH, expand=True)
        
        self.speech_text = tk.Text(speech_frame, font=("Consolas", 10), 
                                  bg="#2c3e50", fg="#ecf0f1", relief=tk.FLAT, bd=0)
        self.speech_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Aktualizácia systémových informácií
        self.update_system_info()
    
    def update_system_info(self):
        """Aktualizuje systémové informácie."""
        info = []
        info.append(f"🎤 Mikrofón: {'ON' if self.microphone_enabled else 'OFF'}")
        info.append(f"🎧 Slúchadlá: {'ON' if self.headphones_enabled else 'OFF'}")
        info.append(f"👆 Kurzor: {'ON' if self.air_cursor.tracking_enabled else 'OFF'}")
        info.append(f"📐 Kalibrácia: {'✅ Dokončená' if self.air_cursor.calibrated else '❌ Potrebná'}")
        
        if ML_AVAILABLE:
            info.append(f"🤲 ML Gestá: {'ON' if self.ml_gesture_enabled else 'OFF'}")
            model_exists = Path("gesture_model.pth").exists()
            info.append(f"🧠 ML Model: {'✅ Dostupný' if model_exists else '❌ Nedostupný'}")
        else:
            info.append("🤲 ML Gestá: ❌ Nedostupné")
        
        info.append(f"🔄 Stav: {'🟢 Aktívny' if self.air_cursor.running else '🔴 Zastavený'}")
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "\n".join(info))
        
        # Aktualizácia každé 2 sekundy
        self.root.after(2000, self.update_system_info)
    
    def toggle_microphone(self):
        """Prepne stav mikrofónu."""
        self.microphone_enabled = not self.microphone_enabled
        self.cursor.microphone_enabled = self.microphone_enabled
        
        if self.microphone_enabled:
            self.mic_button.config(text="🎤 Mikrofón ON", bg="#27ae60")
            self.queue.put(("tts", "Mikrofón zapnutý"))
        else:
            self.mic_button.config(text="🎤 Mikrofón OFF", bg="#e74c3c")
            self.queue.put(("tts", "Mikrofón vypnutý"))
    
    def toggle_headphones(self):
        """Prepne stav slúchadiel."""
        self.headphones_enabled = not self.headphones_enabled
        
        if self.headphones_enabled:
            self.headphones_button.config(text="🎧 Slúchadlá ON", bg="#3498db")
            speak("Slúchadlá zapnuté", headphones_enabled=True)
        else:
            self.headphones_button.config(text="🎧 Slúchadlá OFF", bg="#e74c3c")
    
    def toggle_cursor(self):
        """Prepne sledovanie kurzora."""
        self.air_cursor.tracking_enabled = not self.air_cursor.tracking_enabled
        
        if self.air_cursor.tracking_enabled:
            self.cursor_button.config(text="👆 Kurzor ON", bg="#27ae60")
            self.queue.put(("tts", "Sledovanie kurzora zapnuté"))
        else:
            self.cursor_button.config(text="👆 Kurzor OFF", bg="#e74c3c")
            self.queue.put(("tts", "Sledovanie kurzora vypnuté"))
    
    def toggle_ml_gestures(self):
        """Prepne ML gesture recognition."""
        if not ML_AVAILABLE:
            return
        
        self.ml_gesture_enabled = not self.ml_gesture_enabled
        
        if self.ml_gesture_enabled:
            self.start_ml_gesture_recognition()
            self.gesture_button.config(text="🤲 ML Gestá ON", bg="#27ae60")
            self.queue.put(("tts", "ML rozpoznávanie giest zapnuté"))
        else:
            self.stop_ml_gesture_recognition()
            self.gesture_button.config(text="🤲 ML Gestá OFF", bg="#9b59b6")
            self.queue.put(("tts", "ML rozpoznávanie giest vypnuté"))
    
    def start_ml_gesture_recognition(self):
        """Spustí ML gesture recognition thread."""
        if self.gesture_process is None or not self.gesture_process.is_alive():
            model_path = "gesture_model.pth"
            use_ml = Path(model_path).exists()

            self.gesture_process = threading.Thread(
                target=ml_gesture_recognition_process,
                args=(self.gesture_queue, model_path, use_ml),
                daemon=True
            )
            self.gesture_process.start()
            logger.info("ML gesture recognition thread spustený")
    
    def stop_ml_gesture_recognition(self):
        """Zastaví ML gesture recognition thread."""
        if self.gesture_process and self.gesture_process.is_alive():
            # Pre thread nemôžeme použiť terminate, len označíme že má skončiť
            self.gesture_process = None
            logger.info("ML gesture recognition thread označený na zastavenie")
    
    def start_gesture_monitoring(self):
        """Spustí monitoring gesture recognition queue."""
        def monitor_gestures():
            while True:
                try:
                    if not self.gesture_queue.empty():
                        gesture_type, gesture_info = self.gesture_queue.get_nowait()
                        if gesture_type == "gesture":
                            self.handle_gesture(gesture_info)
                except:
                    pass
                time.sleep(0.01)
        
        gesture_thread = threading.Thread(target=monitor_gestures, daemon=True)
        gesture_thread.start()
    
    def handle_gesture(self, gesture_info):
        """Spracuje rozpoznané gesto."""
        gesture = gesture_info["gesture"]
        confidence = gesture_info["confidence"]
        method = gesture_info["method"]
        
        # Zobrazenie v GUI
        if ML_AVAILABLE and hasattr(self, 'gesture_text'):
            timestamp = time.strftime("%H:%M:%S")
            message = f"[{timestamp}] {gesture} (conf: {confidence:.2f}, {method})\n"
            self.gesture_text.insert(tk.END, message)
            self.gesture_text.see(tk.END)
            
            # Udržanie len posledných 20 riadkov
            lines = self.gesture_text.get(1.0, tk.END).split('\n')
            if len(lines) > 20:
                self.gesture_text.delete(1.0, f"{len(lines)-20}.0")
        
        # Spracovanie gesta
        self.execute_gesture_action(gesture, confidence)
    
    def execute_gesture_action(self, gesture, confidence):
        """Vykoná akciu na základe rozpoznaného gesta."""
        # Minimálna confidence pre vykonanie akcie
        if confidence < 0.6:
            return
        
        actions = {
            "pest": "Klik",
            "otvorena_dlan": "Stop",
            "palec_hore": "Scroll hore",
            "ukazovak": "Ukazovanie",
            "peace": "Screenshot"
        }
        
        action = actions.get(gesture, "Neznáme gesto")
        
        # TTS oznámenie
        if self.headphones_enabled:
            speak(f"Gesto {gesture}: {action}", headphones_enabled=True)
        
        # Výpis do konzoly
        logger.info(f"Vykonávam akciu pre gesto '{gesture}': {action}")
        
        # Tu by boli konkrétne akcie pre každé gesto
        # Napríklad:
        # if gesture == "pest":
        #     pyautogui.click()
        # elif gesture == "palec_hore":
        #     pyautogui.scroll(3)
    
    def process_queue(self):
        """Spracováva správy z queue."""
        try:
            while not self.queue.empty():
                msg_type, msg = self.queue.get_nowait()
                if msg_type == "tts":
                    speak(msg, headphones_enabled=self.headphones_enabled)
                elif msg_type == "stt":
                    self.speech_text.insert(tk.END, f"Rozpoznaný príkaz: {msg}\n")
                    self.speech_text.see(tk.END)
                self.queue.task_done()
        except:
            pass
        self.root.after(100, self.process_queue)

def main():
    """Hlavná funkcia aplikácie s ML gesture recognition."""
    config = ConfigManager()
    os.environ["GEMINI_API_KEY"] = config.get("gemini", "api_key")
    cap = cv2.VideoCapture(config.get("camera", "device_id"))
    air_cursor = AirCursor()
    cursor = Cursor(air_cursor)
    root = tk.Tk()
    queue = Queue()
    gui = ModernGUI(root, queue, cursor, air_cursor)

    def on_closing():
        # Zastavenie ML gesture recognition
        if hasattr(gui, 'stop_ml_gesture_recognition'):
            gui.stop_ml_gesture_recognition()
        
        air_cursor.running = False
        cursor.running = False
        cap.release()
        cv2.destroyAllWindows()
        queue.put(("tts", "Aplikácia bola ukončená."))
        speak("Aplikácia bola ukončená.", headphones_enabled=gui.headphones_enabled)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    cursor_thread = threading.Thread(target=run_cursor_tracking, args=(air_cursor, cap, queue))
    voice_thread = threading.Thread(target=voice_command_listener, args=(cursor, queue))
    cursor_thread.start()
    voice_thread.start()
    
    logger.info("AirCursor aplikácia s ML gesture recognition spustená")
    root.mainloop()
    
    cursor_thread.join()
    voice_thread.join()

if __name__ == "__main__":
    main()
