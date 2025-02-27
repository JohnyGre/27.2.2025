import tkinter as tk
import threading
import cv2
from queue import Queue
from air_cursor import AirCursor
from voice_commands import Cursor, voice_command_listener
from config_manager import ConfigManager
from gtts import gTTS
from playsound3 import playsound
import os
from PIL import Image, ImageTk
import logging

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
        finger_pos = air_cursor.process_frame(frame)

        if finger_pos and air_cursor.calibrated:
            air_cursor.update_cursor(finger_pos)

        if not air_cursor.calibrated:
            if finger_pos:
                cv2.putText(frame, f"Kalibrácia {len(air_cursor.calibration_sets) + 1}/3 - Bod {len(air_cursor.corners) + 1}/4",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for i, corner in enumerate(air_cursor.corners):
            cv2.circle(frame, corner, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (corner[0] - 10, corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if i > 0:
                cv2.line(frame, air_cursor.corners[i - 1], corner, (0, 255, 255), 1)
        if len(air_cursor.corners) >= 4:
            cv2.line(frame, air_cursor.corners[-1], air_cursor.corners[0], (0, 255, 255), 1)

        cv2.imshow("AirCursor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("a"):
            air_cursor.auto_calibrate(frame)
        elif key == ord("r"):
            air_cursor.corners = []
            air_cursor.calibration_sets = []
            air_cursor.calibrated = False
            air_cursor.prev_cursor = None
            queue.put(("tts", "Kalibrácia zresetovaná"))

class ModernGUI:
    def __init__(self, root: tk.Tk, queue: Queue, cursor: Cursor, air_cursor: AirCursor):
        self.root = root
        self.queue = queue
        self.cursor = cursor
        self.air_cursor = air_cursor
        self.root.title("AirCursor Assistant")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2E3440")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # GUI prvky (bez zmien vzhľadu)
        self.speech_frame = tk.Frame(self.root, bg="#3B4252", bd=2, relief=tk.SUNKEN)
        self.speech_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tk.Label(self.speech_frame, text="Google Speech-to-Text", bg="#3B4252", fg="#D8DEE9", font=("Helvetica", 14, "bold")).pack(fill=tk.X, padx=10, pady=5)
        self.speech_text = tk.Text(self.speech_frame, wrap=tk.WORD, bg="#3B4252", fg="#D8DEE9", font=("Helvetica", 12))
        self.speech_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tts_frame = tk.Frame(self.root, bg="#3B4252", bd=2, relief=tk.SUNKEN)
        self.tts_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        tk.Label(self.tts_frame, text="TTS Output", bg="#3B4252", fg="#D8DEE9", font=("Helvetica", 14, "bold")).pack(fill=tk.X, padx=10, pady=5)
        self.tts_text = tk.Text(self.tts_frame, wrap=tk.WORD, bg="#3B4252", fg="#D8DEE9", font=("Helvetica", 12))
        self.tts_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.icons_frame = tk.Frame(self.root, bg="#2E3440")
        self.icons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.mic_icon = ImageTk.PhotoImage(Image.open("microphone.png").resize((32, 32), Image.LANCZOS))
        self.headphone_icon = ImageTk.PhotoImage(Image.open("headphones.png").resize((32, 32), Image.LANCZOS))
        tk.Label(self.icons_frame, image=self.mic_icon, bg="#2E3440").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(self.icons_frame, image=self.headphone_icon, bg="#2E3440").pack(side=tk.LEFT, padx=10, pady=5)

        self.buttons_frame = tk.Frame(self.root, bg="#2E3440")
        self.buttons_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.mic_button = tk.Button(self.buttons_frame, text="ON", command=self.toggle_mic, bg="green", fg="white", font=("Helvetica", 12))
        self.mic_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.headphone_button = tk.Button(self.buttons_frame, text="ON", command=self.toggle_headphones, bg="green", fg="white", font=("Helvetica", 12))
        self.headphone_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.mic_enabled = True
        self.headphones_enabled = True
        self.process_queue()

    def toggle_mic(self):
        """Prepnutie stavu mikrofónu."""
        self.mic_enabled = not self.mic_enabled
        self.cursor.mic_enabled = self.mic_enabled
        self.mic_button.config(text="ON" if self.mic_enabled else "OFF", bg="green" if self.mic_enabled else "red")

    def toggle_headphones(self):
        """Prepnutie stavu slúchadiel."""
        self.headphones_enabled = not self.headphones_enabled
        self.cursor.headphones_enabled = self.headphones_enabled
        self.headphone_button.config(text="ON" if self.headphones_enabled else "OFF", bg="green" if self.headphones_enabled else "red")

    def process_queue(self):
        """Spracovanie správ z fronty."""
        try:
            while not self.queue.empty():
                msg_type, msg = self.queue.get_nowait()
                if msg_type == "tts":
                    self.tts_text.insert(tk.END, f"{msg}\n")
                    self.tts_text.see(tk.END)
                    speak(msg, headphones_enabled=self.headphones_enabled)
                elif msg_type == "stt":
                    self.speech_text.insert(tk.END, f"Rozpoznaný príkaz: {msg}\n")
                    self.speech_text.see(tk.END)
                self.queue.task_done()
        except Queue.Empty:
            pass
        self.root.after(100, self.process_queue)

def main():
    """Hlavná funkcia aplikácie."""
    config = ConfigManager()
    os.environ["GEMINI_API_KEY"] = config.get("gemini", "api_key")
    cap = cv2.VideoCapture(config.get("camera", "device_id"))
    cursor = Cursor()
    air_cursor = AirCursor()
    root = tk.Tk()
    queue = Queue()
    gui = ModernGUI(root, queue, cursor, air_cursor)

    def on_closing():
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
    root.mainloop()
    cursor_thread.join()
    voice_thread.join()

if __name__ == "__main__":
    main()