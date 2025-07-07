import tkinter as tk
import speech_recognition as sr
from gtts import gTTS
from playsound3 import playsound
import os
import logging
from typing import Optional

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SmartAssistant:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Assistant")
        self.root.geometry("600x400")
        self.root.configure(bg="#2E3440")

        self.tts_text = tk.Text(self.root, height=10, width=50, bg="#3B4252", fg="#D8DEE9", font=("Helvetica", 12))
        self.tts_text.pack(pady=10)

        self.speech_text = tk.Text(self.root, height=10, width=50, bg="#3B4252", fg="#D8DEE9", font=("Helvetica", 12))
        self.speech_text.pack(pady=10)

        self.listen_button = tk.Button(self.root, text="Listen", command=self.listen, bg="green", fg="white", font=("Helvetica", 12))
        self.listen_button.pack(pady=5)

        self.speak_button = tk.Button(self.root, text="Speak", command=self.speak_text, bg="blue", fg="white", font=("Helvetica", 12))
        self.speak_button.pack(pady=5)

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False

    def listen(self):
        """Listen for voice input and display it."""
        if self.is_listening:
            return
        self.is_listening = True
        self.speech_text.insert(tk.END, "Listening...\n")
        self.root.update()

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5)
            text = self.recognizer.recognize_google(audio, language="sk-SK")
            self.speech_text.delete("1.0", tk.END)
            self.speech_text.insert(tk.END, f"You said: {text}\n")
            logger.info(f"Recognized: {text}")
        except sr.UnknownValueError:
            logger.warning("Speech not recognized")
        except sr.RequestError as e:
            self.speech_text.insert(tk.END, f"Speech service error: {e}\n")
            logger.error(f"Speech service error: {e}")
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
        finally:
            self.is_listening = False

    def speak_text(self):
        """Convert text to speech using gTTS."""
        text = self.tts_text.get("1.0", tk.END).strip()
        if not text:
            logger.warning("No text to speak")
            return
        try:
            tts = gTTS(text=text, lang="sk")
            filename = "temp_audio.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
            logger.info(f"Spoke: {text}")
        except Exception as e:
            logger.error(f"Error in TTS: {e}")

    def on_key_press(self, event):
        """Handle key press events."""
        if event.char == 'q':
            self.root.quit()

    def run(self):
        """Run the assistant application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.mainloop()

    def on_closing(self):
        """Handle window closing."""
        logger.info("Smart Assistant shutting down")
        self.root.destroy()

if __name__ == "__main__":
    assistant = SmartAssistant()
    assistant.run()