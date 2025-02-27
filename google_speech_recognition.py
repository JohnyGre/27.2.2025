import os
from google.cloud import speech
import pyaudio
from queue import Queue
import logging
from typing import Generator, Optional

# Nastavenie logovania
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Parametre nahrávania zvuku
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream:
    def __init__(self, rate: int = RATE, chunk: int = CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        logger.info("Microphone stream opened")
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()
        logger.info("Microphone stream closed")

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self) -> Generator[bytes, None, None]:
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get_nowait()
                    if chunk is None:
                        return
                    data.append(chunk)
                except Queue.Empty:
                    break
            yield b''.join(data)

def listen_print_loop(responses, callback: Optional[callable] = None):
    """Process streaming responses and optionally call a callback."""
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        logger.info(f"Transcript: {transcript}")
        if callback:
            callback(transcript)

def main(callback: Optional[callable] = None):
    """Main function for streaming speech recognition."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/jangr/canvas asistent/PythonProject3/google_credentials.json"
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='sk-SK',
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses, callback)

if __name__ == "__main__":
    main()