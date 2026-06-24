import sys
import threading
import time
from collections import deque

import numpy as np
import mss
import pytesseract
from PIL import Image
import sounddevice as sd
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont

# ===================== CONFIG =====================
SOURCE_LANG = "zh"      # "zh" = Chinese, "ja" = Japanese
TARGET_LANG = "vi"      # Vietnamese
WHISPER_MODEL = "small"
OCR_REGION = None
SAMPLE_RATE = 16000
CHUNK_DURATION = 5

def ensure_translation_package(source_lang, target_lang):
    """Install translation package if missing. Falls back to two-step (via English)"""
    try:
        argostranslate.translate.translate("test", source_lang, target_lang)
        return
    except:
        pass

    print(f"Downloading translation model: {source_lang} → {target_lang} ...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    # Try direct package
    package = next((p for p in available if p.from_code == source_lang and p.to_code == target_lang), None)
    if package:
        argostranslate.package.install_from_path(package.download())
        print(f"✅ Direct model installed: {source_lang} → {target_lang}")
        return

    # Fallback: use English as bridge (zh → en → vi)
    print("Direct package not found. Using English bridge (zh → en → vi)...")
    pkg1 = next((p for p in available if p.from_code == source_lang and p.to_code == "en"), None)
    pkg2 = next((p for p in available if p.from_code == "en" and p.to_code == target_lang), None)
    
    if pkg1:
        argostranslate.package.install_from_path(pkg1.download())
    if pkg2:
        argostranslate.package.install_from_path(pkg2.download())
    print("✅ Bridge models installed (via English)")

def translate_text(text, source_lang, target_lang):
    """Translate with fallback to English bridge"""
    try:
        return argostranslate.translate.translate(text, source_lang, target_lang)
    except:
        # Try via English
        try:
            en_text = argostranslate.translate.translate(text, source_lang, "en")
            return argostranslate.translate.translate(en_text, "en", target_lang)
        except Exception as e:
            return f"[Translation failed] {text[:40]}"

class AudioTranslator(QObject):
    translation_ready = pyqtSignal(str)

    def __init__(self, source_lang):
        super().__init__()
        self.source_lang = source_lang
        ensure_translation_package(source_lang, TARGET_LANG)
        self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * CHUNK_DURATION))

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_buffer.extend(indata[:, 0])

    def start(self):
        def loop():
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                callback=self.audio_callback,
                                blocksize=int(SAMPLE_RATE * 0.1)):
                while True:
                    if len(self.audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
                        audio = np.array(list(self.audio_buffer), dtype=np.float32)
                        self.audio_buffer.clear()

                        segments, _ = self.model.transcribe(audio, language=self.source_lang)
                        text = " ".join([s.text for s in segments]).strip()
                        if text:
                            translated = translate_text(text, self.source_lang, TARGET_LANG)
                            self.translation_ready.emit(f"🎙️ {translated}")

                    time.sleep(0.5)
        threading.Thread(target=loop, daemon=True).start()

class SubtitleTranslator(QObject):
    translation_ready = pyqtSignal(str)

    def __init__(self, source_lang):
        super().__init__()
        self.source_lang = source_lang
        ensure_translation_package(source_lang, TARGET_LANG)
        self.sct = mss.MSS()

    def capture_and_translate(self):
        if OCR_REGION is None:
            return
        monitor = {"top": OCR_REGION[1], "left": OCR_REGION[0],
                   "width": OCR_REGION[2], "height": OCR_REGION[3]}
        img = np.array(self.sct.grab(monitor))
        lang = "chi_sim+eng" if self.source_lang == "zh" else "jpn+eng"
        text = pytesseract.image_to_string(img, lang=lang).strip()
        if text:
            translated = translate_text(text, self.source_lang, TARGET_LANG)
            self.translation_ready.emit(f"📺 {translated}")

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 650, 140)

        self.label = QLabel("Ready - Chinese/Japanese → Vietnamese")
        self.label.setFont(QFont("Segoe UI", 14))
        self.label.setStyleSheet("color: #00ffcc; background-color: rgba(0,0,0,200); padding: 15px; border-radius: 8px;")
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def show_translation(self, text):
        self.label.setText(text)

def main():
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()

    source_lang = "zh"

    audio = AudioTranslator(source_lang)
    audio.translation_ready.connect(overlay.show_translation)
    audio.start()

    subtitle = SubtitleTranslator(source_lang)
    subtitle.translation_ready.connect(overlay.show_translation)

    timer = QTimer()
    timer.timeout.connect(subtitle.capture_and_translate)
    timer.start(3000)

    print("✅ Translator is running!")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
