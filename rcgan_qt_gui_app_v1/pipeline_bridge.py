from pathlib import Path
import re
import shutil


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FAULTS = ("blur", "occlusion", "brightness")
SEVERITIES = ("low", "medium", "high")


def natural_sort_key(path):
    name = Path(path).name
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


def condition_key(fault, severity):
    return f"{fault}_{severity}"


def known_condition_keys():
    return [
        condition_key(fault, severity)
        for fault in FAULTS
        for severity in SEVERITIES
    ]


def detect_generated_condition(path):
    stem = Path(path).stem

    for condition in known_condition_keys():
        if stem.endswith(f"_{condition}"):
            return condition

    return None


def copy_images(image_paths, target_dir, log_callback=None):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    for source in sorted([Path(p) for p in image_paths], key=natural_sort_key):
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        target = target_dir / source.name
        shutil.copy2(source, target)
        copied.append(str(target))

        if log_callback:
            log_callback(f"Kopyalandi: {target}")

    return copied


def reset_dir(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def prepare_project_clean_dataset(
    clean_image_paths,
    project_root,
    clean_dir_name="clean",
    clear_existing=True,
    log_callback=None,
):
    clean_dir = Path(project_root) / clean_dir_name
    clean_image_paths = [Path(path) for path in clean_image_paths]

    existing_project_clean = [
        path for path in clean_image_paths
        if path.parent.resolve() == clean_dir.resolve()
    ]

    if existing_project_clean and len(existing_project_clean) == len(clean_image_paths):
        copied = sorted(
            [str(path) for path in existing_project_clean if path.exists()],
            key=natural_sort_key,
        )

        if len(copied) != len(clean_image_paths):
            raise FileNotFoundError(
                "Proje clean klasorundeki secili dosyalardan bazilari bulunamadi. "
                "Clean veri setini arayuzden tekrar secmelisin."
            )

        if log_callback:
            log_callback(f"Proje clean klasoru zaten hazir: {len(copied)} dosya")

        return {
            "clean_dir": str(clean_dir),
            "copied_clean": copied,
        }

    if clear_existing:
        if log_callback:
            log_callback(f"Proje clean klasoru temizleniyor: {clean_dir}")
        reset_dir(clean_dir)
    else:
        clean_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_images(
        image_paths=clean_image_paths,
        target_dir=clean_dir,
        log_callback=log_callback,
    )

    if log_callback:
        log_callback(f"Proje clean klasoru hazir: {len(copied)} dosya")

    return {
        "clean_dir": str(clean_dir),
        "copied_clean": copied,
    }


def prepare_detector_dataset(
    clean_image_paths,
    generated_image_paths,
    detector_root,
    clear_existing=True,
    log_callback=None,
):
    detector_root = Path(detector_root)
    dataset_root = detector_root / "robustness_dataset"

    clean_dir = dataset_root / "clean"
    generated_root = dataset_root / "generated"

    def log(message):
        if log_callback:
            log_callback(str(message))

    log("Detector veri seti hazirlaniyor...")

    if clear_existing:
        log("Eski detector staging klasorleri temizleniyor...")
        reset_dir(clean_dir)
        reset_dir(generated_root)

    copied_clean = copy_images(
        clean_image_paths,
        clean_dir,
        log_callback=log_callback,
    )

    grouped_generated = {}
    skipped_generated = []

    for path in sorted([Path(p) for p in generated_image_paths], key=natural_sort_key):
        condition = detect_generated_condition(path)

        if condition is None:
            skipped_generated.append(str(path))
            continue

        grouped_generated.setdefault(condition, []).append(path)

    copied_generated = {}

    for condition, paths in grouped_generated.items():
        condition_dir = generated_root / condition
        copied_generated[condition] = copy_images(
            paths,
            condition_dir,
            log_callback=log_callback,
        )

    log(f"Clean kopya sayisi: {len(copied_clean)}")

    for condition, paths in copied_generated.items():
        log(f"{condition} generated kopya sayisi: {len(paths)}")

    if skipped_generated:
        log(f"Kosulu anlasilamayan generated dosya sayisi: {len(skipped_generated)}")

    return {
        "dataset_root": str(dataset_root),
        "clean_dir": str(clean_dir),
        "generated_root": str(generated_root),
        "copied_clean": copied_clean,
        "copied_generated": copied_generated,
        "skipped_generated": skipped_generated,
    }
