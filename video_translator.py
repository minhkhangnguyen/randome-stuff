import sys
import os
import threading
import time
from collections import deque
from pathlib import Path

# Keep HuggingFace/faster-whisper downloads inside this project folder.
# This avoids Windows privilege/symlink problems in the default HF cache.
BASE_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = BASE_DIR / ".cache" / "huggingface"
LOCAL_MODEL_DIR = BASE_DIR / "models" / "faster-whisper"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(LOCAL_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_CACHE_DIR / "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
import argostranslate.package
import argostranslate.translate
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont

# ===================== CONFIG =====================
# Override with: python video_translator.py zh  OR  python video_translator.py ja
SOURCE_LANG = os.environ.get("SOURCE_LANG", "zh")
TARGET_LANG = os.environ.get("TARGET_LANG", "vi")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 1.0
SILENCE_THRESHOLD = 0.005


def _installed_translation_codes():
    return {
        (translation.from_lang.code, translation.to_lang.code)
        for language in argostranslate.translate.get_installed_languages()
        for translation in language.translations_from
    }


def ensure_translation_package(source_lang, target_lang):
    """Install a direct Argos model or source->English->target bridge."""
    installed = _installed_translation_codes()
    if (source_lang, target_lang) in installed:
        return
    if (source_lang, "en") in installed and ("en", target_lang) in installed:
        return

    print(f"Downloading translation model: {source_lang} → {target_lang} ...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    direct = next(
        (p for p in available if p.from_code == source_lang and p.to_code == target_lang),
        None,
    )
    if direct:
        argostranslate.package.install_from_path(direct.download())
        print("✅ Direct translation model installed")
        return

    bridge_packages = [
        next((p for p in available if p.from_code == source_lang and p.to_code == "en"), None),
        next((p for p in available if p.from_code == "en" and p.to_code == target_lang), None),
    ]
    missing = [p for p in bridge_packages if p is None]
    if missing:
        raise RuntimeError(
            f"No Argos Translate model path found for {source_lang} → {target_lang}. "
            f"Tried direct and {source_lang} → en → {target_lang}."
        )

    for package in bridge_packages:
        argostranslate.package.install_from_path(package.download())
    print("✅ Bridge translation models installed")


def translate_text(text, source_lang, target_lang):
    if not text.strip():
        return ""

    try:
        return argostranslate.translate.translate(text, source_lang, target_lang)
    except Exception:
        try:
            en_text = argostranslate.translate.translate(text, source_lang, "en")
            return argostranslate.translate.translate(en_text, "en", target_lang)
        except Exception:
            return text


def load_whisper_model():
    """Download/load Whisper safely on Windows without requiring admin symlink rights."""
    model_dir = LOCAL_MODEL_DIR / WHISPER_MODEL
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_path = download_model(WHISPER_MODEL, output_dir=str(model_dir))
    except TypeError:
        # Compatibility with older faster-whisper versions.
        model_path = str(model_dir)
        if not any(model_dir.iterdir()):
            model_path = download_model(WHISPER_MODEL, output_dir=str(model_dir))

    try:
        return WhisperModel(model_path, device="auto", compute_type="default")
    except Exception as first_error:
        # Some PCs do not have a usable CUDA/GPU setup. CPU int8 is slower but reliable.
        print(f"⚠️ GPU/auto Whisper load failed, falling back to CPU int8: {first_error}")
        return WhisperModel(model_path, device="cpu", compute_type="int8")


class AudioTranslator(QObject):
    translation_ready = pyqtSignal(str)

    def __init__(self, source_lang):
        super().__init__()
        self.source_lang = source_lang
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.text_history = deque(maxlen=4)

        ensure_translation_package(source_lang, TARGET_LANG)
        self.model = load_whisper_model()

    def start(self):
        def record_loop():
            import soundcard as sc
            import warnings

            warnings.filterwarnings("ignore", message="data discontinuity in recording")

            try:
                speaker = sc.default_speaker()
                mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
                print(f"✅ Successfully opened WASAPI Loopback on: {speaker.name}")
                self.translation_ready.emit("Ready - Capturing system audio")

                with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
                    while True:
                        data = recorder.record(numframes=int(SAMPLE_RATE * 0.1))
                        if len(data.shape) > 1 and data.shape[1] > 1:
                            mono = np.mean(data, axis=1)
                        else:
                            mono = data.flatten()

                        with self.buffer_lock:
                            self.audio_buffer.extend(mono.astype(np.float32).tolist())
            except Exception as e:
                message = f"❌ Failed to capture system audio: {e}"
                print(message)
                self.translation_ready.emit(message)

        def process_loop():
            min_samples = int(SAMPLE_RATE * MIN_AUDIO_SECONDS)
            silence_frames = int(SAMPLE_RATE * 0.8)
            max_samples = int(SAMPLE_RATE * 8)

            while True:
                with self.buffer_lock:
                    current_len = len(self.audio_buffer)
                    audio = (
                        np.array(self.audio_buffer[:current_len], dtype=np.float32)
                        if current_len >= min_samples
                        else None
                    )

                if audio is not None:
                    if np.max(np.abs(audio)) > SILENCE_THRESHOLD:
                        try:
                            segments, _ = self.model.transcribe(
                                audio,
                                language=self.source_lang,
                                beam_size=2,
                                vad_filter=True,
                            )
                            current_text = " ".join(s.text.strip() for s in segments).strip()
                        except Exception as e:
                            message = f"❌ Whisper transcription failed: {e}"
                            print(message)
                            self.translation_ready.emit(message)
                            current_text = ""

                        is_silence = (
                            current_len > silence_frames
                            and np.max(np.abs(audio[-silence_frames:])) < SILENCE_THRESHOLD
                        )
                        is_too_long = current_len > max_samples

                        history_text = " ".join(self.text_history)
                        full_context = f"{history_text} {current_text}".strip()

                        if full_context:
                            translated = translate_text(full_context, self.source_lang, TARGET_LANG)
                            self.translation_ready.emit(f"🎙️ {translated}")

                        if is_silence or is_too_long:
                            if current_text:
                                self.text_history.append(current_text)
                            with self.buffer_lock:
                                del self.audio_buffer[:current_len]
                    else:
                        # Flush silence to prevent memory growth.
                        if current_len > int(SAMPLE_RATE * 2):
                            with self.buffer_lock:
                                del self.audio_buffer[:current_len]

                time.sleep(0.3)

        threading.Thread(target=record_loop, daemon=True).start()
        threading.Thread(target=process_loop, daemon=True).start()


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 900, 170)

        self.label = QLabel("Starting translator...")
        self.label.setFont(QFont("Segoe UI", 16))
        self.label.setStyleSheet(
            "color: #00ffcc; background-color: rgba(0,0,0,200); "
            "padding: 20px; border-radius: 10px;"
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setFixedWidth(860)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def show_translation(self, text):
        self.label.setText(text)
        self.adjustSize()


def main():
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()

    source_lang = sys.argv[1].lower() if len(sys.argv) > 1 else SOURCE_LANG
    if source_lang not in {"zh", "ja"}:
        overlay.show_translation(f"Unsupported source language: {source_lang}. Use zh or ja.")
        return app.exec_()

    try:
        audio = AudioTranslator(source_lang)
        audio.translation_ready.connect(overlay.show_translation)
        audio.start()
        print(f"✅ Translator running ({source_lang} → {TARGET_LANG})")
    except Exception as e:
        message = f"❌ Translator startup failed: {e}"
        print(message)
        overlay.show_translation(message)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
