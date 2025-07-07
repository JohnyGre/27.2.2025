import json
from pathlib import Path
import cv2
import mediapipe as mp
import pyautogui
from screeninfo import get_monitors
from config_manager import ConfigManager
import logging

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AirCursor:
    def __init__(self, config_manager=None):
        """Inicializácia AirCursor s konfiguráciou a logovaním."""
        self.config = config_manager or ConfigManager()
        self._setup_mediapipe()
        self._setup_screen()
        self._setup_calibration()
        self.calibration_file = Path("calibration.json")
        self.load_calibration_points()
        self.running = True

    def _setup_mediapipe(self):
        """Nastavenie MediaPipe Hands s konfiguráciou."""
        hand_config = self.config.get("hand_tracking")
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=hand_config.get("max_hands", 1),
            min_detection_confidence=hand_config.get("detection_confidence", 0.7),
            min_tracking_confidence=hand_config.get("tracking_confidence", 0.5),
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _setup_screen(self):
        """Nastavenie rozmerov obrazovky a PyAutoGUI."""
        screen = get_monitors()[0]  # Predpokladáme primárny monitor
        self.monitor_width = screen.width
        self.monitor_height = screen.height
        pyautogui.FAILSAFE = self.config.get("cursor", "failsafe", default=True)

    def _setup_calibration(self):
        """Inicializácia atribútov pre kalibráciu."""
        self.corners = []
        self.calibration_sets = []
        self.calibrated = False
        self.prev_cursor = None
        cursor_config = self.config.get("cursor")
        self.cursor_smoothing = cursor_config.get("smoothing", 0.7)
        self.sensitivity_x = cursor_config.get("sensitivity_x", 1.3)
        self.sensitivity_y = cursor_config.get("sensitivity_y", 1.3)

    def load_calibration_points(self):
        """Načítanie kalibračných bodov zo súboru."""
        if not self.calibration_file.exists():
            logger.info("Kalibračný súbor neexistuje, inicializujem nové nastavenie.")
            return
        try:
            with self.calibration_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
                self.corners = data.get("corners", [])
                self.calibrated = len(self.corners) >= 4
                logger.info("Kalibračné body načítané zo súboru.")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Chyba pri načítaní kalibračných bodov: {e}")

    def save_calibration_points(self):
        """Uloženie kalibračných bodov do súboru."""
        try:
            with self.calibration_file.open("w", encoding="utf-8") as file:
                json.dump({"corners": self.corners}, file, indent=2)
            logger.info("Kalibračné body uložené do súboru.")
        except IOError as e:
            logger.error(f"Chyba pri ukladaní kalibračných bodov: {e}")

    def process_frame(self, frame):
        """Spracovanie snímku z kamery a získanie pozície prsta."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        index_tip = landmarks.landmark[8]  # Špička ukazováka
        return (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

    def auto_calibrate(self, frame):
        """Automatická kalibrácia s priemerovaním."""
        if len(self.calibration_sets) >= 3:
            self._finalize_calibration()
            return

        if len(self.corners) < 4:
            self._add_calibration_corner(frame)
            if len(self.corners) == 4:
                self.calibration_sets.append(self.corners.copy())
                self.corners = []
                logger.info(f"Kalibrácia {len(self.calibration_sets)}/3 dokončená.")

    def _add_calibration_corner(self, frame):
        """Pridanie kalibračného rohu."""
        width, height = frame.shape[1], frame.shape[0]
        corners = [
            (width // 4, height // 4),       # Ľavý horný
            (3 * width // 4, height // 4),  # Pravý horný
            (width // 4, 3 * height // 4),  # Ľavý dolný
            (3 * width // 4, 3 * height // 4),  # Pravý dolný
        ]
        self.corners.append(corners[len(self.corners)])

    def _finalize_calibration(self):
        """Finalizácia kalibrácie priemerovaním."""
        self.corners = [
            (
                sum(c[i][0] for c in self.calibration_sets) // len(self.calibration_sets),
                sum(c[i][1] for c in self.calibration_sets) // len(self.calibration_sets),
            )
            for i in range(4)
        ]
        self.calibrated = True
        self.save_calibration_points()
        logger.info("Kalibrácia dokončená a uložená.")

    def update_cursor(self, finger_pos):
        """Aktualizácia pozície kurzora na základe polohy prsta."""
        if not (self.calibrated and finger_pos and len(self.corners) >= 4):
            return

        try:
            x, y = finger_pos
            if not self._are_corners_valid():
                logger.warning("Neplatné kalibračné rohy.")
                return

            # Relatívna pozícia vzhľadom na kalibračné rohy
            rel_x = (x - self.corners[0][0]) / (self.corners[1][0] - self.corners[0][0])
            rel_y = (y - self.corners[0][1]) / (self.corners[2][1] - self.corners[0][1])

            # Ošetrenie hodnôt mimo rozsahu
            rel_x = max(0.01, min(rel_x, 0.99))  # Zabránime úplnému dosiahnutiu rohov (0 alebo 1)
            rel_y = max(0.01, min(rel_y, 0.99))

            cursor_x = int(rel_x * self.monitor_width * self.sensitivity_x)
            cursor_y = int(rel_y * self.monitor_height * self.sensitivity_y)

            if self.prev_cursor:
                cursor_x = int(
                    self.prev_cursor[0] + (cursor_x - self.prev_cursor[0]) * self.cursor_smoothing
                )
                cursor_y = int(
                    self.prev_cursor[1] + (cursor_y - self.prev_cursor[1]) * self.cursor_smoothing
                )

            # Ošetrenie hraníc obrazovky
            cursor_x = max(1, min(cursor_x, self.monitor_width - 1))  # Necháme 1px rezervu
            cursor_y = max(1, min(cursor_y, self.monitor_height - 1))

            self.prev_cursor = (cursor_x, cursor_y)
            pyautogui.moveTo(cursor_x, cursor_y)
            logger.debug(f"Kurzor presunutý na: ({cursor_x}, {cursor_y})")
        except Exception as e:
            logger.error(f"Chyba pri aktualizácii kurzora: {e}")
    def _are_corners_valid(self):
        """Kontrola, či sú kalibračné rohy platné."""
        return (
            len(self.corners) >= 4 and
            self.corners[1][0] != self.corners[0][0] and
            self.corners[2][1] != self.corners[0][1]
        )

if __name__ == "__main__":
    air_cursor = AirCursor()
    cap = cv2.VideoCapture(0)
    while air_cursor.running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        finger_pos = air_cursor.process_frame(frame)
        air_cursor.update_cursor(finger_pos)
        cv2.imshow("AirCursor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()