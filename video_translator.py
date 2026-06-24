import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont

# ===================== CONFIG =====================
SOURCE_LANG = "zh"
TARGET_LANG = "vi"
WHISPER_MODEL = "tiny"
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 1.2

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
        
        self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        self.buffer = deque(maxlen=int(SAMPLE_RATE * 12))

    def audio_callback(self, indata, frames, time_info, status):
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        self.buffer.extend(mono)

    def start(self):
        def loop():
            min_samples = int(SAMPLE_RATE * MIN_AUDIO_SECONDS)
            
            # Try to find a loopback device first
            devices = sd.query_devices()
            loopback_device = None
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0 and ('loopback' in dev['name'].lower() or 'stereo mix' in dev['name'].lower()):
                    loopback_device = i
                    break

            # Fallback: use default output device with 2 channels
            if loopback_device is None:
                try:
                    loopback_device = sd.default.device[1]
                except:
                    loopback_device = None

            print(f"🎙️ Using audio device: {loopback_device}")

            try:
                # Force 2 channels (most common for system audio)
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=2,                    # ← Force stereo
                    callback=self.audio_callback,
                    blocksize=int(SAMPLE_RATE * 0.1),
                    device=loopback_device,
                    dtype='float32'
                ):
                    while True:
                        if len(self.buffer) >= min_samples:
                            audio = np.array(list(self.buffer)[-min_samples:], dtype=np.float32)
                            
                            if np.max(np.abs(audio)) > 0.003:
                                segments, _ = self.model.transcribe(
                                    audio, 
                                    language=self.source_lang,
                                    beam_size=1,
                                    vad_filter=True
                                )
                                text = " ".join([s.text for s in segments]).strip()
                                
                                if text and len(text) > 1:
                                    translated = translate_text(text, self.source_lang, TARGET_LANG)
                                    self.translation_ready.emit(f"🎙️ {translated}")
                        
                        time.sleep(0.25)
            except Exception as e:
                print(f"Audio capture error: {e}")
                
        threading.Thread(target=loop, daemon=True).start()

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 650, 140)

        self.label = QLabel("Ready - Capturing system audio")
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

    print("✅ Translator running (System audio mode)")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
