import os
import shutil
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from generate import (
    FAULT_MAP,
    SEVERITY_MAP,
    collect_images_from_folder,
    generate_sequence,
    natural_sort_key,
)
from pipeline_bridge import copy_images, prepare_detector_dataset
from pipeline_bridge import prepare_project_clean_dataset


APP_TITLE = "RCGAN Robustness Pipeline"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_CLEAN_DIR = PROJECT_ROOT / "clean"
DEFAULT_OUTPUT_DIR = str(DEFAULT_OUTPUTS_ROOT / "gan_generated")
DEFAULT_DETECTOR_DIR = str(PROJECT_ROOT / "detector")


class ImagePreview(QLabel):
    def __init__(self, placeholder="Önizleme"):
        super().__init__(placeholder)

        self._image_path = None
        self._placeholder = placeholder

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 300)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setObjectName("preview")

    def set_image(self, path):
        self._image_path = str(path)
        self._refresh_pixmap()

    def clear_image(self):
        self._image_path = None
        self.clear()
        self.setText(self._placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self._image_path:
            self._refresh_pixmap()

    def _refresh_pixmap(self):
        pixmap = QPixmap(self._image_path)

        if pixmap.isNull():
            self.clear()
            self.setText("Görüntü açılamadı")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        scaled = pixmap.scaled(
            max(1, self.width() - 12),
            max(1, self.height() - 12),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.setPixmap(scaled)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class InferenceWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        checkpoint,
        image_paths,
        output_dir,
        fault,
        severity,
        all_conditions,
        conditions=None,
        image_size=256,
    ):
        super().__init__()

        self.checkpoint = checkpoint
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.fault = fault
        self.severity = severity
        self.all_conditions = all_conditions
        self.conditions = conditions
        self.image_size = image_size

    def run(self):
        try:
            output_paths = generate_sequence(
                checkpoint_path=self.checkpoint,
                image_paths=self.image_paths,
                output_dir=self.output_dir,
                fault=self.fault,
                severity=self.severity,
                all_conditions=self.all_conditions,
                conditions=self.conditions,
                image_size=self.image_size,
                log_callback=self.log.emit,
            )

            self.finished_ok.emit(output_paths)

        except Exception:
            self.failed.emit(traceback.format_exc())


class PipelineWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        mode,
        detector_dir,
        clean_image_paths=None,
        generated_image_paths=None,
        checkpoint=None,
        output_dir=None,
        fault=None,
        severity=None,
        all_conditions=False,
        conditions=None,
        image_size=256,
    ):
        super().__init__()

        self.mode = mode
        self.detector_dir = Path(detector_dir)
        self.clean_image_paths = clean_image_paths or []
        self.generated_image_paths = generated_image_paths or []
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.fault = fault
        self.severity = severity
        self.all_conditions = all_conditions
        self.conditions = conditions
        self.image_size = image_size

    def run(self):
        try:
            result = self.run_pipeline_step()
            self.finished_ok.emit(result)

        except Exception:
            self.failed.emit(traceback.format_exc())

    def run_pipeline_step(self):
        detector_dir = self.detector_dir.resolve()
        dataset_root = detector_dir / "robustness_dataset"
        outputs_root = detector_dir.parent / "outputs"
        results_root = detector_dir.parent / "results"

        if self.mode == "full":
            self.log.emit("\n--- Tam pipeline basladi ---")
            self.reset_pipeline_outputs(outputs_root, results_root)

            clean_prepared = prepare_project_clean_dataset(
                clean_image_paths=self.clean_image_paths,
                project_root=detector_dir.parent,
                clear_existing=True,
                log_callback=self.log.emit,
            )
            self.clean_image_paths = [Path(p) for p in clean_prepared["copied_clean"]]

            generated_paths = generate_sequence(
                checkpoint_path=self.checkpoint,
                image_paths=self.clean_image_paths,
                output_dir=self.output_dir,
                fault=self.fault,
                severity=self.severity,
                all_conditions=self.all_conditions,
                conditions=self.conditions,
                image_size=self.image_size,
                log_callback=self.log.emit,
            )

            prepared = self.prepare_dataset(
                detector_dir=detector_dir,
                generated_paths=generated_paths,
            )
            self.log_prepare_summary(prepared)

            generated_dirs = self.generated_dirs_from_prepare_result(
                dataset_root=dataset_root,
                prepared=prepared,
                root_name="generated",
            )

            upscaled = self.run_upscale(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=generated_dirs,
                upscaled_root=outputs_root / "gan_upscaled",
            )
            upscaled_dirs = self.upscaled_dirs_from_result(
                upscaled=upscaled,
            )

            yolo = self.run_yolo(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                results_dir=results_root / "yolo",
            )
            segmentation = self.run_segmentation(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                output_dir=outputs_root / "segmentation_outputs",
                results_dir=results_root / "segmentation",
            )
            yolo_visuals = self.run_yolo_visuals(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                output_dir=outputs_root / "yolo_comparisons",
            )
            segmentation_visuals = self.run_segmentation_visuals(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                seg_output_root=outputs_root / "segmentation_outputs",
                output_dir=outputs_root / "segmentation_comparisons",
            )

            self.log.emit("\n--- Tam pipeline tamamlandi ---")

            return {
                "mode": self.mode,
                "generated_paths": generated_paths,
                "clean_prepared": clean_prepared,
                "prepared": prepared,
                "upscaled": upscaled,
                "yolo": yolo,
                "segmentation": segmentation,
                "yolo_visuals": yolo_visuals,
                "segmentation_visuals": segmentation_visuals,
            }

        if self.mode == "prepare":
            clean_prepared = prepare_project_clean_dataset(
                clean_image_paths=self.clean_image_paths,
                project_root=detector_dir.parent,
                clear_existing=True,
                log_callback=self.log.emit,
            )
            self.clean_image_paths = [Path(p) for p in clean_prepared["copied_clean"]]

            return {
                "mode": self.mode,
                "clean_prepared": clean_prepared,
                "prepared": self.prepare_dataset(
                    detector_dir=detector_dir,
                    generated_paths=self.generated_image_paths,
                ),
            }

        if self.mode == "upscale":
            return {
                "mode": self.mode,
                "upscaled": self.run_upscale(
                    detector_dir=detector_dir,
                    dataset_root=dataset_root,
                    upscaled_root=outputs_root / "gan_upscaled",
                ),
            }

        if self.mode == "yolo":
            generated_dirs = self.generated_dirs_from_root(outputs_root / "gan_upscaled")
            self.ensure_clean_staging(dataset_root)

            return {
                "mode": self.mode,
                "yolo": self.run_yolo(
                    detector_dir=detector_dir,
                    dataset_root=dataset_root,
                    generated_dirs=generated_dirs,
                    results_dir=results_root / "yolo",
                ),
            }

        if self.mode == "segmentation":
            generated_dirs = self.generated_dirs_from_root(outputs_root / "gan_upscaled")
            self.ensure_clean_staging(dataset_root)

            return {
                "mode": self.mode,
                "segmentation": self.run_segmentation(
                    detector_dir=detector_dir,
                    dataset_root=dataset_root,
                    generated_dirs=generated_dirs,
                    output_dir=outputs_root / "segmentation_outputs",
                    results_dir=results_root / "segmentation",
                ),
            }

        raise ValueError(f"Bilinmeyen pipeline modu: {self.mode}")

    def reset_pipeline_outputs(self, outputs_root, results_root):
        for folder in [
            outputs_root / "gan_generated",
            outputs_root / "gan_upscaled",
            outputs_root / "segmentation_outputs",
            outputs_root / "segmentation_comparisons",
            outputs_root / "yolo_comparisons",
            results_root / "yolo",
            results_root / "segmentation",
        ]:
            if folder.exists():
                shutil.rmtree(folder)

            folder.mkdir(parents=True, exist_ok=True)

        self.log.emit("Eski outputs/results pipeline klasorleri temizlendi.")

    def prepare_dataset(self, detector_dir, generated_paths):
        return prepare_detector_dataset(
            clean_image_paths=self.clean_image_paths,
            generated_image_paths=generated_paths,
            detector_root=detector_dir,
            log_callback=self.log.emit,
        )

    def log_prepare_summary(self, prepared):
        self.log.emit(f"Detector clean staging: {prepared['clean_dir']}")
        self.log.emit(f"Detector generated staging: {prepared['generated_root']}")
        self.log.emit(f"Kopyalanan clean sayisi: {len(prepared['copied_clean'])}")

        for condition, paths in prepared["copied_generated"].items():
            self.log.emit(f"Kopyalanan {condition} sayisi: {len(paths)}")

    def ensure_clean_staging(self, dataset_root):
        clean_dir = Path(dataset_root) / "clean"

        if clean_dir.exists() and any(clean_dir.iterdir()):
            return

        if not self.clean_image_paths:
            project_clean_dir = Path(dataset_root).parent.parent / "clean"

            if project_clean_dir.exists():
                self.clean_image_paths = [
                    path for path in sorted(project_clean_dir.iterdir())
                    if path.is_file() and path.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".bmp",
                        ".webp",
                    }
                ]

        if not self.clean_image_paths:
            raise FileNotFoundError(
                "Clean staging klasoru hazir degil. Once clean frame listesini secip "
                "`Detector Veri Setini Hazirla` veya tam pipeline adimini calistirmalisin."
            )

        self.log.emit("Clean staging eksik; secili clean frame listesi kopyalaniyor...")
        copied = copy_images(
            image_paths=self.clean_image_paths,
            target_dir=clean_dir,
            log_callback=self.log.emit,
        )
        self.log.emit(f"Clean staging hazirlandi: {len(copied)} dosya")

    def run_upscale(self, detector_dir, dataset_root, generated_dirs=None, upscaled_root=None):
        run_upscale_generated_images = self.import_detector_function(
            detector_dir,
            "upscale_generated_images",
            "run_upscale_generated_images",
        )

        return run_upscale_generated_images(
            dataset_root=str(dataset_root),
            generated_dirs=generated_dirs,
            upscaled_root=str(upscaled_root) if upscaled_root else None,
            model_path=str(detector_dir / "EDSR_x4.pb"),
            log_callback=self.log.emit,
        )

    def run_yolo(self, detector_dir, dataset_root, generated_dirs, results_dir):
        run_yolo_evaluation = self.import_detector_function(
            detector_dir,
            "yolo_robustness_evaluation",
            "run_yolo_evaluation",
        )

        return run_yolo_evaluation(
            dataset_root=str(dataset_root),
            results_dir=str(results_dir),
            yolo_model_path=str(detector_dir / "yolov8n.pt"),
            generated_dirs=generated_dirs,
            log_callback=self.log.emit,
        )

    def run_segmentation(
        self,
        detector_dir,
        dataset_root,
        generated_dirs,
        output_dir,
        results_dir,
    ):
        run_segmentation_evaluation = self.import_detector_function(
            detector_dir,
            "semantic_segmentation_robustness",
            "run_segmentation_evaluation",
        )

        return run_segmentation_evaluation(
            dataset_root=str(dataset_root),
            output_dir=str(output_dir),
            results_dir=str(results_dir),
            generated_dirs=generated_dirs,
            log_callback=self.log.emit,
        )

    def run_yolo_visuals(self, detector_dir, dataset_root, generated_dirs, output_dir):
        run_yolo_visual_failure_analysis = self.import_detector_function(
            detector_dir,
            "yolo_visual_failure_analysis",
            "run_yolo_visual_failure_analysis",
        )

        return run_yolo_visual_failure_analysis(
            dataset_root=str(dataset_root),
            generated_dirs=generated_dirs,
            output_dir=str(output_dir),
            yolo_model_path=str(detector_dir / "yolov8n.pt"),
            log_callback=self.log.emit,
        )

    def run_segmentation_visuals(
        self,
        detector_dir,
        dataset_root,
        generated_dirs,
        seg_output_root,
        output_dir,
    ):
        run_segmentation_visual_comparison = self.import_detector_function(
            detector_dir,
            "segmentation_visual_comparison",
            "run_segmentation_visual_comparison",
        )

        return run_segmentation_visual_comparison(
            dataset_root=str(dataset_root),
            generated_image_dirs=generated_dirs,
            seg_output_root=str(seg_output_root),
            output_dir=str(output_dir),
            log_callback=self.log.emit,
        )

    def generated_dirs_from_prepare_result(self, dataset_root, prepared, root_name):
        return {
            condition: str(dataset_root / root_name / condition)
            for condition in prepared["copied_generated"]
        }

    def upscaled_dirs_from_result(self, upscaled):
        return {
            condition: str(Path(upscaled["upscaled_root"]) / condition)
            for condition in upscaled["outputs"]
        }

    def generated_dirs_from_root(self, generated_root):
        generated_root = Path(generated_root)

        if generated_root.exists():
            dirs = {
                path.name: str(path)
                for path in sorted(generated_root.iterdir())
                if path.is_dir()
            }

            if dirs:
                return dirs

        return {
            "blur_high": str(generated_root / "blur_high"),
            "brightness_high": str(generated_root / "brightness_high"),
            "occlusion_high": str(generated_root / "occlusion_high"),
        }

    def import_detector_function(self, detector_dir, module_name, function_name):
        detector_path = str(detector_dir)

        if detector_path not in sys.path:
            sys.path.insert(0, detector_path)

        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name)


class RCGANQtApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_TITLE)
        self.resize(1320, 840)

        self.worker = None
        self.pipeline_worker = None
        self.output_paths = []
        self.selected_images = []

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setObjectName("controlScroll")

        left_panel = QWidget()
        left_panel.setObjectName("controlPanel")

        left = QVBoxLayout(left_panel)
        right = QVBoxLayout()
        left.setSpacing(10)
        right.setSpacing(10)

        left_scroll.setWidget(left_panel)

        root.addWidget(left_scroll, 2)
        root.addLayout(right, 3)

        title = QLabel("RCGAN Robustness Pipeline")
        title.setObjectName("title")
        left.addWidget(title)

        file_group = QGroupBox("1) Dosyaları Seç")
        file_layout = QGridLayout(file_group)

        self.checkpoint_edit = QLineEdit("checkpoint_epoch_29.pt")
        self.output_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        self.detector_edit = QLineEdit(DEFAULT_DETECTOR_DIR)

        self._add_file_row(
            file_layout,
            0,
            "Checkpoint",
            self.checkpoint_edit,
            self.pick_checkpoint,
        )

        self.select_images_btn = QPushButton("Çoklu Fotoğraf Seç")
        self.select_folder_btn = QPushButton("Klasörden Fotoğrafları Al")

        self.select_images_btn.clicked.connect(self.pick_multiple_images)
        self.select_folder_btn.clicked.connect(self.pick_image_folder)

        file_layout.addWidget(QLabel("Zaman serisi:"), 1, 0)
        file_layout.addWidget(self.select_images_btn, 1, 1)
        file_layout.addWidget(self.select_folder_btn, 1, 2)

        self.input_list = QListWidget()
        self.input_list.itemSelectionChanged.connect(self.preview_selected_input)

        file_layout.addWidget(QLabel("Seçilen clean frame listesi:"), 2, 0)
        file_layout.addWidget(self.input_list, 2, 1, 1, 2)

        self._add_file_row(
            file_layout,
            3,
            "Output",
            self.output_edit,
            self.pick_output_dir,
            directory=True,
        )

        self._add_file_row(
            file_layout,
            4,
            "Detector",
            self.detector_edit,
            self.pick_detector_dir,
            directory=True,
        )

        left.addWidget(file_group)

        condition_group = QGroupBox("2) Üretim Koşulları")
        condition_layout = QGridLayout(condition_group)

        self.fault_combo = QComboBox()
        self.fault_combo.addItems(list(FAULT_MAP.keys()))

        self.severity_combo = QComboBox()
        self.severity_combo.addItems(list(SEVERITY_MAP.keys()))

        self.blur_severity_combo = QComboBox()
        self.blur_severity_combo.addItems(list(SEVERITY_MAP.keys()))

        self.occlusion_severity_combo = QComboBox()
        self.occlusion_severity_combo.addItems(list(SEVERITY_MAP.keys()))

        self.brightness_severity_combo = QComboBox()
        self.brightness_severity_combo.addItems(list(SEVERITY_MAP.keys()))

        condition_layout.addWidget(QLabel("Tek hata tipi:"), 0, 0)
        condition_layout.addWidget(self.fault_combo, 0, 1)

        condition_layout.addWidget(QLabel("Tek hata seviyesi:"), 1, 0)
        condition_layout.addWidget(self.severity_combo, 1, 1)

        condition_layout.addWidget(QLabel("Blur seviyesi:"), 2, 0)
        condition_layout.addWidget(self.blur_severity_combo, 2, 1)

        condition_layout.addWidget(QLabel("Occlusion seviyesi:"), 3, 0)
        condition_layout.addWidget(self.occlusion_severity_combo, 3, 1)

        condition_layout.addWidget(QLabel("Brightness seviyesi:"), 4, 0)
        condition_layout.addWidget(self.brightness_severity_combo, 4, 1)

        left.addWidget(condition_group)

        action_group = QGroupBox("3) Üret")
        action_layout = QVBoxLayout(action_group)

        self.generate_one_btn = QPushButton("Seçilen Koşulla Zaman Serisi Üret")
        self.generate_selected_faults_btn = QPushButton("3 Hata Tipini Seçilen Seviyelerle Üret")
        self.open_output_btn = QPushButton("GAN Çıktı Klasörünü Aç")
        self.open_outputs_root_btn = QPushButton("Outputs Klasörünü Aç")
        self.open_results_root_btn = QPushButton("Results Klasörünü Aç")
        self.clear_btn = QPushButton("Listeyi Temizle")

        self.generate_one_btn.clicked.connect(self.start_generation_single)
        self.generate_selected_faults_btn.clicked.connect(self.start_generation_selected_faults)
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_outputs_root_btn.clicked.connect(lambda: self.open_folder(DEFAULT_OUTPUTS_ROOT))
        self.open_results_root_btn.clicked.connect(lambda: self.open_folder(DEFAULT_RESULTS_ROOT))
        self.clear_btn.clicked.connect(self.clear_outputs)

        action_layout.addWidget(self.generate_one_btn)
        action_layout.addWidget(self.generate_selected_faults_btn)
        action_layout.addWidget(self.open_output_btn)
        action_layout.addWidget(self.open_outputs_root_btn)
        action_layout.addWidget(self.open_results_root_btn)
        action_layout.addWidget(self.clear_btn)

        left.addWidget(action_group)

        pipeline_group = QGroupBox("4) Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)

        self.pipeline_one_btn = QPushButton("Pipeline: Tek Koşul")
        self.pipeline_selected_faults_btn = QPushButton("Pipeline: 3 Hata Tipi")
        self.prepare_detector_btn = QPushButton("Detector Veri Setini Hazırla")
        self.upscale_btn = QPushButton("Upscale Uygula")
        self.yolo_btn = QPushButton("YOLO Değerlendir")
        self.segmentation_btn = QPushButton("Segmentasyon Değerlendir")

        self.pipeline_one_btn.clicked.connect(self.start_full_pipeline_single)
        self.pipeline_selected_faults_btn.clicked.connect(self.start_full_pipeline_selected_faults)
        self.prepare_detector_btn.clicked.connect(self.start_prepare_detector)
        self.upscale_btn.clicked.connect(self.start_upscale)
        self.yolo_btn.clicked.connect(self.start_yolo)
        self.segmentation_btn.clicked.connect(self.start_segmentation)

        pipeline_layout.addWidget(self.pipeline_one_btn)
        pipeline_layout.addWidget(self.pipeline_selected_faults_btn)
        pipeline_layout.addWidget(self.prepare_detector_btn)
        pipeline_layout.addWidget(self.upscale_btn)
        pipeline_layout.addWidget(self.yolo_btn)
        pipeline_layout.addWidget(self.segmentation_btn)

        left.addWidget(pipeline_group)

        preview_title = QLabel("Önizleme ve Çıktılar")
        preview_title.setObjectName("subtitle")
        right.addWidget(preview_title)

        preview_row = QHBoxLayout()

        self.curr_preview = ImagePreview("Seçilen clean frame önizleme")
        self.out_preview = ImagePreview("Üretilen görüntü önizleme")

        preview_row.addWidget(self.curr_preview, 1)
        preview_row.addWidget(self.out_preview, 1)

        right.addLayout(preview_row, 3)

        self.output_list = QListWidget()
        self.output_list.itemSelectionChanged.connect(self.preview_selected_output)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("İşlem kayıtları burada görünecek...")

        tabs = QTabWidget()
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.addWidget(self.output_list)

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.addWidget(self.log_box)

        tabs.addTab(output_tab, "Çıktılar")
        tabs.addTab(log_tab, "Log")

        right.addWidget(tabs, 2)

        hint = QLabel(
            "Not: Seçilen görüntüler dosya adına göre sıralanır. "
            "İlk görüntü için çıktı üretilmez çünkü önceki frame yoktur. "
            "2. görüntüden itibaren prev + curr şeklinde zaman serisi üretimi yapılır."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")

        right.addWidget(hint)

    def _add_file_row(self, layout, row, label_text, edit, picker, directory=False):
        layout.addWidget(QLabel(label_text + ":"), row, 0)
        layout.addWidget(edit, row, 1)

        btn = QPushButton("Seç")
        btn.clicked.connect(picker)

        layout.addWidget(btn, row, 2)

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
                color: #20242a;
                background: #f4f6f8;
            }

            QWidget#controlPanel {
                background: #f4f6f8;
            }

            QScrollArea#controlScroll {
                border: none;
                background: #f4f6f8;
            }

            QScrollBar:vertical {
                background: #e7ebef;
                width: 10px;
                margin: 0;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #aab5c1;
                min-height: 32px;
                border-radius: 5px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }

            QLabel#title {
                font-size: 25px;
                font-weight: 700;
                margin: 2px 0 8px 0;
                color: #17202a;
            }

            QLabel#subtitle {
                font-size: 19px;
                font-weight: 700;
                color: #17202a;
            }

            QLabel#hint {
                color: #59636f;
                padding: 10px;
                background: #eef2f5;
                border: 1px solid #d9e0e7;
                border-radius: 8px;
            }

            QLabel#preview {
                border: 1px solid #c9d2dc;
                border-radius: 8px;
                background: #ffffff;
            }

            QPushButton {
                padding: 9px 12px;
                border-radius: 7px;
                border: 1px solid #b8c3cf;
                background: #ffffff;
                color: #17202a;
            }

            QPushButton:hover {
                background: #edf5ff;
                border-color: #7aa7d9;
            }

            QPushButton:pressed {
                background: #dcecff;
            }

            QPushButton:disabled {
                color: #9aa3ad;
                background: #edf0f3;
            }

            QLineEdit,
            QComboBox {
                padding: 7px;
                border-radius: 6px;
                border: 1px solid #c8d0da;
                background: #ffffff;
            }

            QTextEdit,
            QListWidget {
                border: 1px solid #c8d0da;
                border-radius: 8px;
                background: #ffffff;
            }

            QTabWidget::pane {
                border: 1px solid #c8d0da;
                border-radius: 8px;
                background: #ffffff;
            }

            QTabBar::tab {
                padding: 8px 14px;
                border: 1px solid #c8d0da;
                border-bottom: none;
                background: #e8edf2;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }

            QTabBar::tab:selected {
                background: #ffffff;
                font-weight: 700;
            }

            QGroupBox {
                font-weight: 700;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                border: 1px solid #d3dbe4;
                border-radius: 8px;
                background: #fbfcfd;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2b3a48;
            }
        """)

    def pick_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Checkpoint seç",
            os.getcwd(),
            "PyTorch Checkpoint (*.pt *.pth);;All Files (*)",
        )

        if path:
            self.checkpoint_edit.setText(path)

    def pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Output klasörü seç",
            os.getcwd(),
        )

        if path:
            self.output_edit.setText(path)

    def pick_detector_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Detector klasörü seç",
            os.getcwd(),
        )

        if path:
            self.detector_edit.setText(path)

    def pick_multiple_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Zaman serisi görüntülerini seç",
            os.getcwd(),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )

        if paths:
            self.selected_images = self.prepare_project_clean_images([Path(p) for p in paths])
            self.refresh_input_list()

    def pick_image_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Görüntü klasörü seç",
            os.getcwd(),
        )

        if folder:
            try:
                images = collect_images_from_folder(folder)
                self.selected_images = self.prepare_project_clean_images(images)
                self.refresh_input_list()

            except Exception as e:
                QMessageBox.warning(self, "Klasör hatası", str(e))

    def prepare_project_clean_images(self, image_paths):
        prepared = prepare_project_clean_dataset(
            clean_image_paths=image_paths,
            project_root=PROJECT_ROOT,
            clear_existing=True,
            log_callback=self.log_box.append,
        )

        return [Path(p) for p in prepared["copied_clean"]]

    def refresh_input_list(self):
        self.selected_images = sorted(
            self.selected_images,
            key=natural_sort_key,
        )

        self.input_list.clear()

        for path in self.selected_images:
            self.input_list.addItem(str(path))

        self.log_box.append(
            f"{len(self.selected_images)} adet görüntü seçildi ve dosya adına göre sıralandı."
        )

        if len(self.selected_images) >= 2:
            expected_outputs = len(self.selected_images) - 1
            self.log_box.append(
                f"Seçili koşulda üretilecek çıktı sayısı: {expected_outputs}"
            )

        if self.selected_images:
            self.input_list.setCurrentRow(0)
            self.show_image(str(self.selected_images[0]), self.curr_preview)

    def preview_selected_input(self):
        items = self.input_list.selectedItems()

        if not items:
            return

        self.show_image(items[0].text(), self.curr_preview)

    def validate_inputs(self):
        checkpoint = self.checkpoint_edit.text().strip()
        out_dir = self.output_edit.text().strip() or DEFAULT_OUTPUT_DIR

        missing = []

        if not Path(checkpoint).exists():
            missing.append(f"Checkpoint bulunamadı: {checkpoint}")

        if len(self.selected_images) < 2:
            missing.append("Zaman serisi üretimi için en az 2 görüntü seçmelisin.")

        for path in self.selected_images:
            if not Path(path).exists():
                missing.append(f"Görüntü bulunamadı: {path}")

        if missing:
            QMessageBox.warning(
                self,
                "Eksik dosya",
                "\n".join(missing),
            )
            return None

        return checkpoint, self.selected_images, out_dir

    def validate_detector_dir(self):
        detector_dir = self.detector_edit.text().strip() or DEFAULT_DETECTOR_DIR
        detector_path = Path(detector_dir)

        missing = []

        if not detector_path.exists():
            missing.append(f"Detector klasörü bulunamadı: {detector_dir}")

        for name in [
            "upscale_generated_images.py",
            "yolo_robustness_evaluation.py",
            "yolo_visual_failure_analysis.py",
            "semantic_segmentation_robustness.py",
            "segmentation_visual_comparison.py",
            "EDSR_x4.pb",
            "yolov8n.pt",
        ]:
            if not (detector_path / name).exists():
                missing.append(f"Detector dosyası bulunamadı: {detector_path / name}")

        if missing:
            QMessageBox.warning(
                self,
                "Detector eksik",
                "\n".join(missing),
            )
            return None

        return detector_path

    def selected_fault_conditions(self):
        return [
            ("blur", self.blur_severity_combo.currentText()),
            ("occlusion", self.occlusion_severity_combo.currentText()),
            ("brightness", self.brightness_severity_combo.currentText()),
        ]

    def start_generation_single(self):
        self.start_generation(
            conditions=[(
                self.fault_combo.currentText(),
                self.severity_combo.currentText(),
            )],
        )

    def start_generation_selected_faults(self):
        self.start_generation(conditions=self.selected_fault_conditions())

    def start_generation(self, conditions):
        values = self.validate_inputs()

        if values is None:
            return

        checkpoint, image_paths, out_dir = values

        self.set_buttons_enabled(False)
        self.output_list.clear()
        self.out_preview.clear_image()

        self.log_box.append("\n--- Yeni zaman serisi üretimi ---")

        total_outputs = (len(image_paths) - 1) * len(conditions)

        self.log_box.append(f"Seçilen clean frame sayısı: {len(image_paths)}")
        self.log_box.append(
            "Koşullar: " + ", ".join(f"{fault}_{severity}" for fault, severity in conditions)
        )
        self.log_box.append(f"Toplam üretilecek çıktı sayısı: {total_outputs}")

        self.worker = InferenceWorker(
            checkpoint=checkpoint,
            image_paths=image_paths,
            output_dir=out_dir,
            fault=self.fault_combo.currentText(),
            severity=self.severity_combo.currentText(),
            all_conditions=False,
            conditions=conditions,
        )

        self.worker.log.connect(self.log_box.append)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.start()

    def start_full_pipeline_single(self):
        self.start_full_pipeline(
            conditions=[(
                self.fault_combo.currentText(),
                self.severity_combo.currentText(),
            )],
        )

    def start_full_pipeline_selected_faults(self):
        self.start_full_pipeline(conditions=self.selected_fault_conditions())

    def start_full_pipeline(self, conditions):
        values = self.validate_inputs()
        detector_dir = self.validate_detector_dir()

        if values is None or detector_dir is None:
            return

        checkpoint, image_paths, out_dir = values

        self.output_list.clear()
        self.out_preview.clear_image()
        self.log_box.append("\n--- Pipeline kuyruğa alındı ---")

        self.pipeline_worker = PipelineWorker(
            mode="full",
            detector_dir=detector_dir,
            clean_image_paths=image_paths,
            checkpoint=checkpoint,
            output_dir=out_dir,
            fault=self.fault_combo.currentText(),
            severity=self.severity_combo.currentText(),
            all_conditions=False,
            conditions=conditions,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_prepare_detector(self):
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        if len(self.selected_images) < 2:
            QMessageBox.warning(
                self,
                "Eksik dosya",
                "Detector veri seti için clean frame listesini seçmelisin.",
            )
            return

        if not self.output_paths:
            QMessageBox.warning(
                self,
                "Eksik çıktı",
                "Önce GAN çıktısı üretmelisin.",
            )
            return

        self.pipeline_worker = PipelineWorker(
            mode="prepare",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
            generated_image_paths=self.output_paths,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_upscale(self):
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        self.pipeline_worker = PipelineWorker(
            mode="upscale",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_yolo(self):
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        self.pipeline_worker = PipelineWorker(
            mode="yolo",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_segmentation(self):
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        self.pipeline_worker = PipelineWorker(
            mode="segmentation",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_pipeline_worker(self, worker):
        self.set_buttons_enabled(False)
        worker.log.connect(self.log_box.append)
        worker.finished_ok.connect(self.on_pipeline_finished)
        worker.failed.connect(self.on_failed)
        worker.start()

    def on_pipeline_finished(self, result):
        self.set_buttons_enabled(True)

        generated_paths = result.get("generated_paths")

        if generated_paths:
            self.output_paths = generated_paths
            self.output_list.clear()

            for path in generated_paths:
                self.output_list.addItem(path)

            self.output_list.setCurrentRow(0)
            self.show_image(generated_paths[0], self.out_preview)

        self.log_box.append("Pipeline adımı tamamlandı.")

    def on_finished(self, paths):
        self.set_buttons_enabled(True)

        self.output_paths = paths
        self.output_list.clear()

        for path in paths:
            self.output_list.addItem(path)

        if paths:
            self.output_list.setCurrentRow(0)
            self.show_image(paths[0], self.out_preview)

        self.log_box.append("İşlem tamamlandı.")

    def on_failed(self, error_text):
        self.set_buttons_enabled(True)

        self.log_box.append(error_text)

        QMessageBox.critical(
            self,
            "Hata",
            error_text[-3000:],
        )

    def set_buttons_enabled(self, enabled):
        self.generate_one_btn.setEnabled(enabled)
        self.generate_selected_faults_btn.setEnabled(enabled)
        self.pipeline_one_btn.setEnabled(enabled)
        self.pipeline_selected_faults_btn.setEnabled(enabled)
        self.prepare_detector_btn.setEnabled(enabled)
        self.upscale_btn.setEnabled(enabled)
        self.yolo_btn.setEnabled(enabled)
        self.segmentation_btn.setEnabled(enabled)
        self.open_output_btn.setEnabled(enabled)
        self.open_outputs_root_btn.setEnabled(enabled)
        self.open_results_root_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
        self.select_images_btn.setEnabled(enabled)
        self.select_folder_btn.setEnabled(enabled)

    def show_image(self, path, label):
        if isinstance(label, ImagePreview):
            label.set_image(path)
            return

        pixmap = QPixmap(path)

        if pixmap.isNull():
            label.setText("Görüntü açılamadı")
            return

        label.setPixmap(
            pixmap.scaled(
                max(1, label.width() - 12),
                max(1, label.height() - 12),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def preview_selected_output(self):
        items = self.output_list.selectedItems()

        if not items:
            return

        self.show_image(items[0].text(), self.out_preview)

    def clear_outputs(self):
        self.output_paths = []
        self.selected_images = []

        self.output_list.clear()
        self.input_list.clear()

        self.out_preview.clear_image()
        self.curr_preview.clear_image()

        self.log_box.append("Listeler temizlendi.")

    def open_output_folder(self):
        out_dir = self.output_edit.text().strip() or DEFAULT_OUTPUT_DIR
        self.open_folder(out_dir)

    def open_folder(self, folder):
        folder = str(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

        if sys.platform == "darwin":
            os.system(f'open "{folder}"')
        elif sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            os.system(f'xdg-open "{folder}"')


def main():
    app = QApplication(sys.argv)

    window = RCGANQtApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
