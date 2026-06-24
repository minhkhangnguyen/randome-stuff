import sys
import os
import html
import re
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
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None
try:
    from google.cloud import speech
except Exception:
    speech = None
try:
    from google.cloud import translate_v2 as google_translate_v2
except Exception:
    google_translate_v2 = None
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont

# ===================== CONFIG =====================
# Override with: python video_translator.py zh  OR  python video_translator.py ja
SOURCE_LANG = os.environ.get("SOURCE_LANG", "zh")
TARGET_LANG = os.environ.get("TARGET_LANG", "vi")
# google = online Google Translate, argos/local = offline Argos Translate
TRANSLATOR = os.environ.get("TRANSLATOR", "google").lower()
# whisper = local faster-whisper, google = Google Cloud Speech-to-Text
SPEECH_ENGINE = os.environ.get("SPEECH_ENGINE", "whisper").lower()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 1.0
SILENCE_THRESHOLD = 0.005
SENTENCE_ENDINGS = ".!?…。！？"
GOOGLE_LANG_CODES = {"zh": "zh-CN", "ja": "ja", "vi": "vi", "en": "en"}
GOOGLE_SPEECH_LANG_CODES = {"zh": "zh-CN", "ja": "ja-JP", "vi": "vi-VN", "en": "en-US"}
GOOGLE_TRANSLATE_CLIENT = None


def clean_subtitle_text(text):
    """Make subtitle text easier to read on one overlay."""
    text = re.sub(r"\s+", " ", text).strip()
    # Remove accidental spaces before punctuation, common after ASR/translation.
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    return text


def add_reading_punctuation(text, final=False):
    """Add a sentence ending when the translator returns plain text only."""
    text = clean_subtitle_text(text)
    if not text:
        return ""
    if text[-1] in SENTENCE_ENDINGS:
        return text
    return f"{text}." if final else f"{text}…"


def format_subtitles(completed_lines, live_line=""):
    """Show completed sentences on separate lines plus the current live sentence."""
    lines = [line for line in completed_lines if line]
    if live_line:
        lines.append(live_line)
    if not lines:
        return "🎙️"
    return "🎙️ " + "\n".join(lines[-4:])


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


def translate_with_google(text, source_lang, target_lang):
    if GoogleTranslator is None:
        raise RuntimeError("deep-translator is not installed")

    google_source = GOOGLE_LANG_CODES.get(source_lang, source_lang)
    google_target = GOOGLE_LANG_CODES.get(target_lang, target_lang)
    return GoogleTranslator(source=google_source, target=google_target).translate(text)


def translate_with_google_cloud(text, source_lang, target_lang):
    """Use official Google Cloud Translation API. Requires GOOGLE_APPLICATION_CREDENTIALS."""
    global GOOGLE_TRANSLATE_CLIENT
    if google_translate_v2 is None:
        raise RuntimeError(
            "google-cloud-translate is not installed. Run pip install -r requirements.txt"
        )
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "Google Cloud Translation needs a Google Cloud service account key. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to the JSON key file path."
        )
    if GOOGLE_TRANSLATE_CLIENT is None:
        GOOGLE_TRANSLATE_CLIENT = google_translate_v2.Client()

    result = GOOGLE_TRANSLATE_CLIENT.translate(
        text,
        source_language=GOOGLE_LANG_CODES.get(source_lang, source_lang),
        target_language=GOOGLE_LANG_CODES.get(target_lang, target_lang),
        format_="text",
    )
    return html.unescape(result.get("translatedText", "")).strip()


def translate_with_argos(text, source_lang, target_lang):
    try:
        return argostranslate.translate.translate(text, source_lang, target_lang)
    except Exception:
        en_text = argostranslate.translate.translate(text, source_lang, "en")
        return argostranslate.translate.translate(en_text, "en", target_lang)


def translate_text(text, source_lang, target_lang):
    if not text.strip():
        return ""

    if TRANSLATOR in {"google_cloud", "googlecloud", "cloud_translate"}:
        try:
            return translate_with_google_cloud(text, source_lang, target_lang)
        except Exception as e:
            print(f"⚠️ Google Cloud Translation failed, trying local Argos fallback: {e}")

    if TRANSLATOR in {"google", "googletranslate", "google_translate"}:
        try:
            return translate_with_google(text, source_lang, target_lang)
        except Exception as e:
            print(f"⚠️ Google Translate failed, trying local Argos fallback: {e}")

    try:
        return translate_with_argos(text, source_lang, target_lang)
    except Exception as e:
        print(f"⚠️ Local translation failed: {e}")
        return text


def load_google_speech_client():
    """Load Google Cloud Speech-to-Text client. Requires GOOGLE_APPLICATION_CREDENTIALS."""
    if speech is None:
        raise RuntimeError(
            "google-cloud-speech is not installed. Run pip install -r requirements.txt"
        )
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "Google Speech-to-Text needs a Google Cloud service account key. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to the JSON key file path."
        )
    return speech.SpeechClient()


def google_recognize_audio(client, audio, source_lang):
    """Send a short audio block to Google Speech-to-Text."""
    if audio.size == 0:
        return ""

    # Convert float32 -1..1 mono audio to 16-bit PCM LINEAR16.
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16).tobytes()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=GOOGLE_SPEECH_LANG_CODES.get(source_lang, source_lang),
        enable_automatic_punctuation=True,
        model="latest_long",
    )
    recognition_audio = speech.RecognitionAudio(content=pcm16)
    response = client.recognize(config=config, audio=recognition_audio)

    return " ".join(
        result.alternatives[0].transcript.strip()
        for result in response.results
        if result.alternatives
    ).strip()

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
        self.translated_history = deque(maxlen=4)

        if TRANSLATOR in {"argos", "local", "offline"}:
            ensure_translation_package(source_lang, TARGET_LANG)

        self.model = None
        self.speech_client = None
        if SPEECH_ENGINE in {"google", "google_speech", "googlecloud"}:
            self.speech_client = load_google_speech_client()
        else:
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
                            if self.speech_client is not None:
                                current_text = google_recognize_audio(
                                    self.speech_client, audio, self.source_lang
                                )
                            else:
                                initial_prompt = " ".join(self.text_history) or None
                                segments, _ = self.model.transcribe(
                                    audio,
                                    language=self.source_lang,
                                    beam_size=2,
                                    vad_filter=True,
                                    initial_prompt=initial_prompt,
                                )
                                current_text = " ".join(s.text.strip() for s in segments).strip()
                        except Exception as e:
                            message = f"❌ Speech recognition failed: {e}"
                            print(message)
                            self.translation_ready.emit(message)
                            current_text = ""

                        is_silence = (
                            current_len > silence_frames
                            and np.max(np.abs(audio[-silence_frames:])) < SILENCE_THRESHOLD
                        )
                        is_too_long = current_len > max_samples

                        if current_text:
                            if is_silence or is_too_long:
                                # The speaker paused (or the block is long), so treat this as
                                # a finished sentence. Add punctuation and move it to history.
                                translated = translate_text(current_text, self.source_lang, TARGET_LANG)
                                final_line = add_reading_punctuation(translated, final=True)
                                self.text_history.append(current_text)
                                self.translated_history.append(final_line)
                                self.translation_ready.emit(format_subtitles(self.translated_history))
                            else:
                                # Live/unfinished subtitle: use an ellipsis so it is clear the
                                # sentence is still continuing instead of looking like one long run-on.
                                translated = translate_text(current_text, self.source_lang, TARGET_LANG)
                                live_line = add_reading_punctuation(translated, final=False)
                                self.translation_ready.emit(
                                    format_subtitles(self.translated_history, live_line)
                                )

                        if is_silence or is_too_long:
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
        print(f"✅ Translator running ({source_lang} → {TARGET_LANG}, speech={SPEECH_ENGINE}, translator={TRANSLATOR})")
    except Exception as e:
        message = f"❌ Translator startup failed: {e}"
        print(message)
        overlay.show_translation(message)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
