import os
import shutil
from pathlib import Path


DATASET_ROOT = "robustness_dataset"
UPSCALED_ROOT = os.path.join(DATASET_ROOT, "generated_upscaled")
MODEL_PATH = "EDSR_x4.pb"
TARGET_WIDTH = 1600
TARGET_HEIGHT = 900
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def cv2_safe_model_path(model_path):
    """OpenCV dnn_superres can fail on Windows paths with non-ASCII chars."""
    model_path = Path(model_path)
    model_text = str(model_path)

    try:
        model_text.encode("ascii")
        return model_text
    except UnicodeEncodeError:
        pass

    public_dir = Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
    cache_dir = public_dir / "sentetik_veri_platformu_cv2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_model = cache_dir / model_path.name
    if (
        not cached_model.exists()
        or cached_model.stat().st_size != model_path.stat().st_size
    ):
        shutil.copyfile(model_path, cached_model)

    return str(cached_model)


def default_generated_dirs(dataset_root=DATASET_ROOT):
    return {
        "blur_high": os.path.join(dataset_root, "generated", "blur_high"),
        "brightness_high": os.path.join(dataset_root, "generated", "brightness_high"),
        "occlusion_high": os.path.join(dataset_root, "generated", "occlusion_high"),
    }


def log_message(message, log_callback=None):
    if log_callback:
        log_callback(str(message))
    else:
        print(message)


def progress_iter(items, log_callback=None):
    if not hasattr(items, "__len__") or len(items) == 0:
        return items

    if log_callback:
        return items

    try:
        from tqdm import tqdm

        return tqdm(items)
    except ImportError:
        return items


def list_images(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)


def load_super_resolution_model(model_path=MODEL_PATH, log_callback=None):
    import cv2

    log_message("\nLoading Super Resolution model...\n", log_callback)

    model_path = cv2_safe_model_path(model_path)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)

    log_message("Super Resolution model loaded.\n", log_callback)

    return sr


def upscale_image_file(
    sr,
    input_path,
    output_path,
    target_size=(TARGET_WIDTH, TARGET_HEIGHT),
):
    import cv2
    import numpy as np

    image_data = np.fromfile(input_path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        return False

    upscaled = sr.upsample(image)

    final = cv2.resize(
        upscaled,
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )

    extension = os.path.splitext(output_path)[1] or ".png"
    ok, encoded = cv2.imencode(extension, final)
    if not ok:
        return False

    encoded.tofile(output_path)
    return True


def run_upscale_generated_images(
    dataset_root=DATASET_ROOT,
    generated_dirs=None,
    upscaled_root=None,
    model_path=MODEL_PATH,
    target_width=TARGET_WIDTH,
    target_height=TARGET_HEIGHT,
    log_callback=None,
):
    if generated_dirs is None:
        generated_dirs = default_generated_dirs(dataset_root)

    if upscaled_root is None:
        upscaled_root = os.path.join(dataset_root, "generated_upscaled")

    sr = load_super_resolution_model(
        model_path=model_path,
        log_callback=log_callback,
    )

    target_size = (target_width, target_height)
    outputs = {}
    skipped = []

    for condition, input_dir in generated_dirs.items():
        if not os.path.isdir(input_dir):
            log_message(f"Skipping missing condition dir: {input_dir}", log_callback)
            continue

        log_message(f"\nUpscaling: {condition}\n", log_callback)

        output_dir = os.path.join(upscaled_root, condition)
        reset_dir(output_dir)

        outputs[condition] = []

        image_files = list_images(input_dir)
        log_message(f"{condition} dosya sayisi: {len(image_files)}", log_callback)

        for index, fname in enumerate(
            progress_iter(image_files, log_callback=log_callback),
            start=1,
        ):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)

            ok = upscale_image_file(
                sr=sr,
                input_path=input_path,
                output_path=output_path,
                target_size=target_size,
            )

            if ok:
                outputs[condition].append(output_path)
            else:
                skipped.append(input_path)

            if log_callback and (index == 1 or index == len(image_files) or index % 10 == 0):
                log_message(
                    f"{condition}: {index}/{len(image_files)} upscale tamamlandi",
                    log_callback,
                )

    log_message("\nUPSCALE FINISHED.\n", log_callback)

    return {
        "upscaled_root": upscaled_root,
        "outputs": outputs,
        "skipped": skipped,
    }


def main():
    run_upscale_generated_images()


if __name__ == "__main__":
    main()
