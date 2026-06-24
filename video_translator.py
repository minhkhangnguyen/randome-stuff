import sys
import threading
import time
from collections import deque

import numpy as np
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont

# ===================== CONFIG =====================
SOURCE_LANG = "zh"
TARGET_LANG = "vi"
WHISPER_MODEL = "base"
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 1.0  # Reduced delay so subtitles appear much faster!

def ensure_translation_package(source_lang, target_lang):
    try:
        argostranslate.translate.translate("test", source_lang, target_lang)
        return
    except:
        pass

    print(f"Downloading translation model: {source_lang} → {target_lang} ...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    package = next((p for p in available if p.from_code == source_lang and p.to_code == target_lang), None)
    if package:
        argostranslate.package.install_from_path(package.download())
        print(f"✅ Direct model installed")
        return

    pkg1 = next((p for p in available if p.from_code == source_lang and p.to_code == "en"), None)
    pkg2 = next((p for p in available if p.from_code == "en" and p.to_code == target_lang), None)
    if pkg1: argostranslate.package.install_from_path(pkg1.download())
    if pkg2: argostranslate.package.install_from_path(pkg2.download())
    print("✅ Bridge models installed")

def translate_text(text, source_lang, target_lang):
    try:
        return argostranslate.translate.translate(text, source_lang, target_lang)
    except:
        try:
            en_text = argostranslate.translate.translate(text, source_lang, "en")
            return argostranslate.translate.translate(en_text, "en", target_lang)
        except:
            return text

class AudioTranslator(QObject):
    translation_ready = pyqtSignal(str)

    def __init__(self, source_lang):
        super().__init__()
        self.source_lang = source_lang
        ensure_translation_package(source_lang, TARGET_LANG)
        
        self.model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="default")
        self.audio_buffer = []
        self.text_history = deque(maxlen=4)  # Stores up to ~100 words of history for context

    def start(self):
        def record_loop():
            import soundcard as sc
            import warnings
            warnings.filterwarnings("ignore", message="data discontinuity in recording")
            
            try:
                speaker = sc.default_speaker()
                mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
                print(f"✅ Successfully opened WASAPI Loopback on: {speaker.name}")
                with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
                    while True:
                        data = recorder.record(numframes=int(SAMPLE_RATE * 0.1))
                        if len(data.shape) > 1 and data.shape[1] > 1:
                            mono = np.mean(data, axis=1)
                        else:
                            mono = data.flatten()
                        self.audio_buffer.extend(mono.astype(np.float32).tolist())
            except Exception as e:
                print(f"❌ Failed to capture system audio: {e}")

        def process_loop():
            min_samples = int(SAMPLE_RATE * MIN_AUDIO_SECONDS)
            silence_threshold = 0.005
            
            while True:
                current_len = len(self.audio_buffer)
                if current_len >= min_samples:
                    # Take snapshot of current audio
                    audio = np.array(self.audio_buffer[:current_len], dtype=np.float32)
                    
                    if np.max(np.abs(audio)) > silence_threshold:
                        segments, _ = self.model.transcribe(
                            audio, 
                            language=self.source_lang,
                            beam_size=2,
                            vad_filter=True
                        )
                        current_text = " ".join([s.text for s in segments]).strip()
                        
                        # Detect if speaker stopped talking (last 0.8s is silence) or block is > 8 seconds
                        silence_frames = int(SAMPLE_RATE * 0.8)
                        is_silence = current_len > silence_frames and np.max(np.abs(audio[-silence_frames:])) < silence_threshold
                        is_too_long = current_len > int(SAMPLE_RATE * 8)
                        
                        # ✨ MAGIC HAPPENS HERE ✨
                        # Combine past 4 sentences + current words to give translator perfect context
                        history_text = " ".join(list(self.text_history))
                        full_context = f"{history_text} {current_text}".strip()
                        
                        if full_context:
                            translated = translate_text(full_context, self.source_lang, TARGET_LANG)
                            self.translation_ready.emit(f"🎙️ {translated}")
                        
                        if is_silence or is_too_long:
                            if current_text:
                                self.text_history.append(current_text) # Lock in confirmed translation
                            # Clear processed audio to start a new sentence
                            self.audio_buffer = self.audio_buffer[current_len:]
                    else:
                        # Flush silence to prevent memory leaks
                        if current_len > int(SAMPLE_RATE * 2):
                            self.audio_buffer = self.audio_buffer[current_len:]
                
                time.sleep(0.3)

        try:
            threading.Thread(target=record_loop, daemon=True).start()
            threading.Thread(target=process_loop, daemon=True).start()
        except Exception as e:
            print(f"❌ Failed to start threads: {e}")

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 800, 150) # Made window slightly wider

        self.label = QLabel("Ready - Capturing system audio")
        self.label.setFont(QFont("Segoe UI", 16))
        self.label.setStyleSheet("color: #00ffcc; background-color: rgba(0,0,0,200); padding: 20px; border-radius: 10px;")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True) # VERY IMPORTANT: Allows text to wrap multiple lines instead of going off screen

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def show_translation(self, text):
        self.label.setText(text)
        self.adjustSize() # Auto-grow/shrink the overlay height based on text length

def main():
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()

    source_lang = "zh"

    audio = AudioTranslator(source_lang)
    audio.translation_ready.connect(overlay.show_translation)
    audio.start()

    print("✅ Translator running")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
