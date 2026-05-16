import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_APP_DIR = PROJECT_ROOT / "rcgan_qt_gui_app_v1"
IMAGE_APP_SCRIPT = IMAGE_APP_DIR / "qt_gui_app_updated.py"

DATA_APP_DIR = PROJECT_ROOT / "akilli_veri_arttirimi"
DATA_APP_SCRIPT = DATA_APP_DIR / "main.py"
DATA_APP_VENV_PYTHON = DATA_APP_DIR / "otonom_env" / "bin" / "python"


class MainLauncher(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentetik Veri Platformu")
        self.resize(940, 620)

        self.processes = []

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(16)

        title = QLabel("Sentetik Veri Platformu")
        title.setObjectName("title")
        root.addWidget(title)

        subtitle = QLabel("Çalışmak istediğin pipeline'ı seç.")
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.setSpacing(16)

        image_card = self._make_card(
            title="Görüntü Robustness Pipeline",
            body=(
                "Clean kamera frame'lerinden RCGAN ile bozulmuş görüntü üretir, "
                "EDSR ile upscale eder, YOLO ve SegFormer ile dayanıklılık analizi yapar."
            ),
            primary_text="Görüntü Modelini Aç",
            primary_action=self.open_image_app,
            secondary_text="Görüntü Klasörünü Aç",
            secondary_action=lambda: self.open_folder(IMAGE_APP_DIR),
        )

        data_card = self._make_card(
            title="Akıllı Veri Artırımı",
            body=(
                "CSV/tabular/yörünge verisini damıtır, veri tipine göre RCGAN, "
                "CTGAN veya SMOTE ile sentetik veri üretir ve utility/fidelity raporlar."
            ),
            primary_text="Veri Artırımı Modelini Aç",
            primary_action=self.open_data_app,
            secondary_text="Veri Artırımı Klasörünü Aç",
            secondary_action=lambda: self.open_folder(DATA_APP_DIR),
        )

        cards.addWidget(image_card, 1)
        cards.addWidget(data_card, 1)
        root.addLayout(cards, 2)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Başlatılan uygulamaların durumları burada görünecek...")
        root.addWidget(self.log_box, 1)

        footer = QLabel(
            "Not: İki uygulama ayrı süreç olarak açılır. Ana ekranı kapatmadan ikisini de çalıştırabilirsin."
        )
        footer.setObjectName("hint")
        footer.setWordWrap(True)
        root.addWidget(footer)

    def _make_card(
        self,
        title,
        body,
        primary_text,
        primary_action,
        secondary_text,
        secondary_action,
    ):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        text = QLabel(body)
        text.setWordWrap(True)
        text.setObjectName("cardBody")
        layout.addWidget(text)
        layout.addStretch(1)

        primary = QPushButton(primary_text)
        primary.setObjectName("primaryButton")
        primary.clicked.connect(primary_action)
        layout.addWidget(primary)

        secondary = QPushButton(secondary_text)
        secondary.clicked.connect(secondary_action)
        layout.addWidget(secondary)

        return group

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
                color: #20242a;
                background: #f4f6f8;
            }

            QLabel#title {
                font-size: 30px;
                font-weight: 800;
                color: #17202a;
            }

            QLabel#subtitle {
                font-size: 16px;
                color: #53606d;
            }

            QLabel#cardBody {
                color: #4b5563;
                line-height: 1.35;
            }

            QLabel#hint {
                color: #59636f;
                padding: 10px;
                background: #eef2f5;
                border: 1px solid #d9e0e7;
                border-radius: 8px;
            }

            QGroupBox {
                font-size: 17px;
                font-weight: 800;
                margin-top: 12px;
                padding: 18px 14px 14px 14px;
                border: 1px solid #d3dbe4;
                border-radius: 10px;
                background: #fbfcfd;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #2b3a48;
            }

            QPushButton {
                padding: 11px 14px;
                border-radius: 8px;
                border: 1px solid #b8c3cf;
                background: #ffffff;
                color: #17202a;
            }

            QPushButton:hover {
                background: #edf5ff;
                border-color: #7aa7d9;
            }

            QPushButton#primaryButton {
                background: #1f6feb;
                color: #ffffff;
                border-color: #1f6feb;
                font-weight: 700;
            }

            QPushButton#primaryButton:hover {
                background: #155fc9;
            }

            QTextEdit {
                border: 1px solid #c8d0da;
                border-radius: 8px;
                background: #ffffff;
            }
        """)

    def open_image_app(self):
        self.launch_app(
            label="Görüntü Robustness Pipeline",
            script=IMAGE_APP_SCRIPT,
            cwd=IMAGE_APP_DIR,
            python=sys.executable,
        )

    def open_data_app(self):
        python = str(DATA_APP_VENV_PYTHON) if DATA_APP_VENV_PYTHON.exists() else sys.executable

        if not DATA_APP_VENV_PYTHON.exists():
            self.log("Akıllı veri artırımı için otonom_env bulunamadı.")

        missing = self.missing_modules(
            python=python,
            modules=["uvicorn", "fastapi", "webview", "torch", "pandas", "sklearn"],
        )

        if missing:
            message = (
                "Akıllı veri artırımı uygulaması için Python ortamı hazır değil.\n\n"
                f"Kullanılan Python:\n{python}\n\n"
                f"Eksik modüller: {', '.join(missing)}\n\n"
                "Kurmak için terminalde:\n"
                "cd /Users/ozcan/Desktop/projects/akilli_veri_arttirimi\n"
                "python3 -m venv otonom_env\n"
                "source otonom_env/bin/activate\n"
                "pip install -r requirements.txt"
            )
            self.log(message)
            QMessageBox.warning(self, "Ortam hazır değil", message)
            return

        self.launch_app(
            label="Akıllı Veri Artırımı",
            script=DATA_APP_SCRIPT,
            cwd=DATA_APP_DIR,
            python=python,
        )

    def missing_modules(self, python, modules):
        missing = []

        for module in modules:
            result = subprocess.run(
                [python, "-c", f"import {module}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )

            if result.returncode != 0:
                missing.append(module)

        return missing

    def launch_app(self, label, script, cwd, python):
        if not script.exists():
            QMessageBox.warning(self, "Dosya bulunamadı", f"Uygulama dosyası yok:\n{script}")
            return

        process = QProcess(self)
        process.setProgram(python)
        process.setArguments([str(script)])
        process.setWorkingDirectory(str(cwd))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        process.readyReadStandardOutput.connect(
            lambda proc=process, name=label: self.read_process_output(name, proc)
        )
        process.errorOccurred.connect(
            lambda error, name=label: self.log(f"{name} süreç hatası: {error}")
        )
        process.finished.connect(
            lambda code, status, name=label: self.log(
                f"{name} kapandı. Exit code: {code}, status: {status}"
            )
        )

        process.start()

        if not process.waitForStarted(3000):
            QMessageBox.critical(self, "Başlatılamadı", f"{label} başlatılamadı.")
            return

        self.processes.append(process)
        self.log(f"{label} başlatıldı. PID: {process.processId()}")

    def read_process_output(self, label, process):
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")

        if text.strip():
            self.log(f"[{label}] {text.rstrip()}")

    def open_folder(self, folder):
        folder = Path(folder)

        if not folder.exists():
            QMessageBox.warning(self, "Klasör bulunamadı", str(folder))
            return

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def log(self, message):
        self.log_box.append(str(message))


def main():
    app = QApplication(sys.argv)
    window = MainLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
