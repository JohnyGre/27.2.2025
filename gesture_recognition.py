import mediapipe as mp
import cv2
import pyautogui
from queue import Queue
import logging
import time
from multiprocessing import Process

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s") # Pre detailnejšie logovanie, ak je potrebné

class GestureRecognizer:
    def __init__(self, output_queue: Queue):
        self.output_queue = output_queue
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0) # 0 - predvolená webkamera
        if not self.cap.isOpened(): # Kontrola, či sa kamera úspešne otvorila
            raise IOError("Nemôžem otvoriť kameru")
        self.running = True
        self.last_gesture_time = time.time()
        logger.info("GestureRecognizer inicializovaný.")


    def run(self):
        try:
            while self.running:
                success, image = self.cap.read()
                if not success:
                    logger.error("Ignoring empty camera frame.")
                    continue

                image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesture = self.recognize_gesture(hand_landmarks)
                        if gesture and self.is_gesture_rate_limited():
                            self.output_queue.put(("gesture", gesture))
                            logger.info(f"Rozpoznané gesto: {gesture}")
                            self.last_gesture_time = time.time()

                # Optional: Vizualizácia pre ladenie - odkomentuj ak chceš vidieť obraz z kamery a detekciu rúk
                # if results.multi_hand_landmarks:
                #     for hand_landmarks in results.multi_hand_landmarks:
                #         self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # cv2.imshow('Hand Tracking', image)
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break
        except Exception as e:
            logger.error(f"Chyba v run() metóde GestureRecognizer: {e}", exc_info=True) # Logovanie chyby aj s tracebackom
        finally:
            self.hands.close()
            self.cap.release()
            logger.info("GestureRecognizer zastavený a kamera uvoľnená.")


    def is_gesture_rate_limited(self):
        return (time.time() - self.last_gesture_time) >= 0.5 # Znížený rate limit na 0.5 sekundy pre rýchlejšie testovanie


    def recognize_gesture(self, hand_landmarks):
        # ------------------------ THUMBS UP --------------------
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        if thumb_tip.y < index_finger_tip.y and \
           index_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
           middle_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
           ring_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y and \
           pinky_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y:
            return "thumbs_up"

        # ------------------------ THUMBS DOWN --------------------
        if thumb_tip.y > index_finger_tip.y and \
           index_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
           middle_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
           ring_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y and \
           pinky_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y:
            return "thumbs_down"

        # ------------------------ VOLUME UP (FIST UP) --------------------
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        if self.is_fist(hand_landmarks) and index_finger_tip.y < wrist.y: # Fist a prsty hore
            return "volume_up"

        # ------------------------ VOLUME DOWN (FIST DOWN) --------------------
        if self.is_fist(hand_landmarks) and index_finger_tip.y > wrist.y: # Fist a prsty dole
            return "volume_down"

        # ------------------------ PEACE GESTURE --------------------
        if self.is_peace_gesture(hand_landmarks):
            return "peace"

        # ------------------------ FIST GESTURE --------------------
        if self.is_fist(hand_landmarks):
            return "fist"

        # ------------------------ POINTING GESTURE --------------------
        if self.is_pointing_gesture(hand_landmarks):
            return "point_index"

        return None # Gesto nerozpoznané


    def is_fist(self, hand_landmarks):
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Podmienky pre "fist" gesto - všetky prsty ohnuté do dlane, zjednodušené
        return index_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y and \
               middle_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and \
               ring_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y and \
               pinky_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y


    def is_peace_gesture(self, hand_landmarks): # Peace gesture (dva prsty) - zjednodušené
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]


        return index_finger_tip.y < index_finger_mcp.y and \
               middle_finger_tip.y < middle_finger_mcp.y and \
               ring_finger_tip.y > ring_finger_mcp.y and \
               pinky_finger_tip.y > pinky_finger_mcp.y


    def is_pointing_gesture(self, hand_landmarks): # Pointing gesture (ukazovák) - zjednodušené
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        return index_finger_tip.y < index_finger_mcp.y and \
               middle_finger_tip.y > middle_finger_mcp.y and \
               ring_finger_tip.y > ring_finger_mcp.y and \
               pinky_finger_tip.y > pinky_finger_mcp.y


    def stop(self):
        self.running = False



def gesture_recognition_process(output_queue: Queue):
    try:
        recognizer = GestureRecognizer(output_queue)
        recognizer.run()
    except IOError as e:
        logger.error(f"Chyba pri inicializácii GestureRecognizer: {e}")
    except Exception as e:
        logger.error(f"Neočekávaná chyba v gesture_recognition_process: {e}", exc_info=True)


def main(): # HLAVNÁ FUNKCIA pre testovanie gesture_recognition.py
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # Nastavenie logovania pre main funkciu

    q = Queue()
    p = Process(target=gesture_recognition_process, args=(q,))
    p.start()
    recognizer = None # DEFINUJEME recognizer PRED try blokom, aby bol dostupný aj vo finally bloku

    try:
        recognizer = GestureRecognizer(q) # Inicializácia recognizer v try bloku
        while True:
            if not q.empty():
                gesture_info = q.get()
                print(f"Rozpoznané gesto: {gesture_info}") # Tlač gesta na konzolu
            time.sleep(0.01) # Krátka pauza
    except KeyboardInterrupt:
        print("Ukončujem rozpoznávanie gest...")
        if recognizer: # Teraz už môžeme jednoducho kontrolovať, či je recognizer inicializovaný
            recognizer.stop() # Zastavenie GestureRecognizer ak bol inicializovaný
        p.terminate()
        p.join()
        print("Rozpoznávanie gest ukončené.")
    except Exception as e: # Zachytenie prípadných chýb v main cykle
        logger.error(f"Chyba v main cykle: {e}", exc_info=True)
    finally:
        if p.is_alive(): # Poistka pre ukončenie procesu
            p.terminate()
            p.join()
            print("Proces rozpoznávania gest nútene ukončený.")


if __name__ == "__main__":
    main() # Spustenie main funkcie, ak sa spúšťa gesture_recognition.py priamo