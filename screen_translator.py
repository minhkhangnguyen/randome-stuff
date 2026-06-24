import os
import re
import sys
import threading
from pathlib import Path
from dataclasses import dataclass

import mss
from PIL import Image, ImageOps, ImageFilter
import pytesseract
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont, QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QRubberBand


BASE_DIR = Path(__file__).resolve().parent
TESSDATA_DIR = Path(os.environ.get("TESSDATA_DIR", str(BASE_DIR / "tessdata")))

TARGET_LANG = os.environ.get("TARGET_LANG", "vi")
# Tesseract languages. Use chi_sim for simplified Chinese, chi_tra for traditional Chinese, jpn for Japanese.
OCR_LANG = os.environ.get("OCR_LANG", "chi_sim+chi_tra+jpn+eng")
CAPTURE_INTERVAL_MS = int(os.environ.get("CAPTURE_INTERVAL_MS", "1200"))
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "2"))
SCREEN_HISTORY_WORDS = int(os.environ.get("SCREEN_HISTORY_WORDS", "300"))


COMMON_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"D:\Tesseract-OCR\tesseract.exe",
]


def _windows_safe_path(path):
    # Tesseract on Windows can misread backslash escapes such as \t in \tessdata.
    # Forward slashes are accepted by Windows and avoid that problem.
    return str(path).replace("\\", "/")


def configure_tesseract():
    configured = os.environ.get("TESSERACT_CMD")
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    else:
        for path in COMMON_TESSERACT_PATHS:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    # Use project-local language files downloaded by install_tesseract.bat.
    # Do NOT wrap this path in quotes. Tesseract receives the quote as part of
    # the path on some Windows/Python setups and then looks for:
    #   "D:/randome-stuff/tessdata"/chi_sim.traineddata
    if TESSDATA_DIR.exists():
        os.environ["TESSDATA_PREFIX"] = _windows_safe_path(TESSDATA_DIR)


def tesseract_config():
    if TESSDATA_DIR.exists():
        return f"--tessdata-dir {_windows_safe_path(TESSDATA_DIR)} --oem 3 --psm 6"
    return "--oem 3 --psm 6"


def missing_tessdata_languages():
    if not TESSDATA_DIR.exists():
        return OCR_LANG.split("+")
    return [
        lang for lang in OCR_LANG.split("+")
        if lang and not (TESSDATA_DIR / f"{lang}.traineddata").exists()
    ]


def clean_ocr_text(text):
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_for_compare(text):
    return re.sub(r"\s+", "", text).strip()


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Upscale + grayscale + contrast/threshold helps subtitles and manga/game text.
    img = img.convert("RGB")
    w, h = img.size
    scale = 2 if max(w, h) < 1400 else 1
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img



def count_words(text):
    # Vietnamese/English use spaces. For CJK text without spaces this still
    # falls back to a rough character count so very long text is trimmed.
    words = re.findall(r"\S+", text)
    if len(words) <= 1 and len(text) > 80:
        return max(1, len(text) // 2)
    return len(words)


def trim_to_word_limit(lines, limit):
    kept = []
    total = 0
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        wc = count_words(line)
        if kept and total + wc > limit:
            break
        kept.append(line)
        total += wc
    return list(reversed(kept))

def translate_text(text):
    if not text.strip():
        return ""
    if GoogleTranslator is None:
        return "deep-translator is not installed. Run pip install -r requirements.txt"
    try:
        return GoogleTranslator(source="auto", target=TARGET_LANG).translate(text)
    except Exception as e:
        return f"Translate failed: {e}\n\nOCR text:\n{text}"


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


class WorkerSignals(QObject):
    result = pyqtSignal(str, str)  # ocr_text, translated_text
    error = pyqtSignal(str)


class RegionSelector(QWidget):
    region_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowFullScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.origin = QPoint()
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))
        painter.setPen(QPen(QColor(0, 255, 204), 2))
        painter.setFont(QFont("Segoe UI", 18))
        painter.drawText(
            self.rect(),
            Qt.AlignTop | Qt.AlignHCenter,
            "Drag to select the screen area to translate   |   ESC = cancel",
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            rect = self.rubber_band.geometry().normalized()
            self.rubber_band.hide()
            if rect.width() >= 20 and rect.height() >= 20:
                # QWidget coords are global because selector is fullscreen on the virtual desktop.
                top_left = self.mapToGlobal(rect.topLeft())
                region = CaptureRegion(
                    left=top_left.x(),
                    top=top_left.y(),
                    width=rect.width(),
                    height=rect.height(),
                )
                self.hide()
                self.region_selected.emit(region)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            QApplication.quit()


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 1040, 360)

        self.label = QLabel("Select an area to translate")
        self.label.setFont(QFont("Segoe UI", 15))
        self.label.setStyleSheet(
            "color: #00ffcc; background-color: rgba(0,0,0,210); "
            "padding: 18px; border-radius: 10px;"
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setFixedWidth(1000)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def show_text(self, text):
        self.label.setText(text)
        self.adjustSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            QApplication.quit()


class ScreenTranslator(QObject):
    def __init__(self, region: CaptureRegion, overlay: Overlay):
        super().__init__()
        self.region = region
        self.overlay = overlay
        self.signals = WorkerSignals()
        self.signals.result.connect(self.on_result)
        self.signals.error.connect(self.on_error)
        self.last_norm_text = ""
        self.last_translated_text = ""
        self.translated_history = []
        self.busy = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_once)
        self.timer.start(CAPTURE_INTERVAL_MS)

        self.overlay.show_text(
            "Screen translator running...\n"
            "Right-click this subtitle box to close."
        )

    def capture_once(self):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._capture_ocr_translate, daemon=True).start()

    def _capture_ocr_translate(self):
        try:
            with mss.mss() as sct:
                monitor = {
                    "left": self.region.left,
                    "top": self.region.top,
                    "width": self.region.width,
                    "height": self.region.height,
                }
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb)

            missing_langs = missing_tessdata_languages()
            if missing_langs:
                raise RuntimeError(
                    "Missing Tesseract language files: "
                    + ", ".join(missing_langs)
                    + f"\nExpected folder: {_windows_safe_path(TESSDATA_DIR)}\n"
                    + "Run install_tesseract.bat again."
                )

            img = preprocess_for_ocr(img)
            ocr_text = pytesseract.image_to_string(
                img,
                lang=OCR_LANG,
                config=tesseract_config(),
            )
            ocr_text = clean_ocr_text(ocr_text)
            norm = normalize_for_compare(ocr_text)

            if len(norm) < MIN_TEXT_CHARS or norm == self.last_norm_text:
                return

            self.last_norm_text = norm
            translated = translate_text(ocr_text)
            self.signals.result.emit(ocr_text, translated)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.busy = False

    def on_result(self, ocr_text, translated):
        translated = translated.strip()
        if not translated:
            return

        # Keep a rolling reading buffer, similar to the speech translator.
        # This makes screen text easier to follow when subtitles/dialogue update
        # quickly, while avoiding an endlessly growing overlay.
        if translated != self.last_translated_text:
            self.last_translated_text = translated
            self.translated_history.append(translated)

        visible_lines = trim_to_word_limit(self.translated_history, SCREEN_HISTORY_WORDS)
        self.translated_history = visible_lines
        self.overlay.show_text("🖼️ " + "\n\n".join(visible_lines))

    def on_error(self, message):
        self.overlay.show_text(
            "❌ Screen OCR failed:\n"
            f"{message}\n\n"
            "Run install_tesseract.bat once, then run_screen.bat again.\n"
            "If Tesseract is already installed somewhere else, set TESSERACT_CMD to tesseract.exe."
        )


def main():
    configure_tesseract()
    app = QApplication(sys.argv)

    overlay = Overlay()
    selector = RegionSelector()
    state = {}

    def start(region):
        overlay.show()
        overlay.move(region.left, max(0, region.top - 230))
        state["translator"] = ScreenTranslator(region, overlay)

    selector.region_selected.connect(start)
    selector.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
