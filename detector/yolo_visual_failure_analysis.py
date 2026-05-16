import os
from pathlib import Path


DATASET_ROOT = "robustness_dataset"
OUTPUT_DIR = "yolo_visual_failure_outputs"
YOLO_MODEL_PATH = "yolov8n.pt"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def default_generated_dirs(dataset_root=DATASET_ROOT):
    return {
        "blur_high": os.path.join(dataset_root, "generated_upscaled", "blur_high"),
        "brightness_high": os.path.join(
            dataset_root,
            "generated_upscaled",
            "brightness_high",
        ),
        "occlusion_high": os.path.join(
            dataset_root,
            "generated_upscaled",
            "occlusion_high",
        ),
    }


def log_message(message, log_callback=None):
    if log_callback:
        log_callback(str(message))
    else:
        print(message)


def load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    return plt


def list_images(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Goruntu klasoru bulunamadi: {folder}. "
            "Once detector veri setini hazirla veya tam pipeline'i calistir."
        )

    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])


def find_matching_clean(generated_name, clean_files):
    for clean_name in clean_files:
        clean_stem = Path(clean_name).stem

        if clean_stem in generated_name:
            return clean_name

    return None


def run_yolo_and_save(model, image_path, output_path):
    import cv2

    results = model(image_path, verbose=False)
    plotted = results[0].plot()

    cv2.imwrite(output_path, plotted)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return 0, 0.0

    confs = boxes.conf.cpu().numpy()
    return len(confs), float(confs.mean())


def create_comparison_figure(
    model,
    clean_img_path,
    corrupted_img_path,
    condition,
    output_path,
    temp_dir,
):
    import cv2

    plt = load_pyplot()

    os.makedirs(temp_dir, exist_ok=True)

    temp_clean = os.path.join(temp_dir, "temp_clean.jpg")
    temp_corr = os.path.join(temp_dir, "temp_corr.jpg")

    clean_count, clean_conf = run_yolo_and_save(model, clean_img_path, temp_clean)
    corr_count, corr_conf = run_yolo_and_save(model, corrupted_img_path, temp_corr)

    clean_img = cv2.cvtColor(cv2.imread(temp_clean), cv2.COLOR_BGR2RGB)
    corr_img = cv2.cvtColor(cv2.imread(temp_corr), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5), dpi=300)

    plt.subplot(1, 2, 1)
    plt.imshow(clean_img)
    plt.title(f"Clean Image\nObjects: {clean_count}, Avg Conf: {clean_conf:.3f}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(corr_img)
    plt.title(
        f"{condition.replace('_', ' ').title()}\n"
        f"Objects: {corr_count}, Avg Conf: {corr_conf:.3f}"
    )
    plt.axis("off")

    plt.suptitle(
        "YOLO Object Detection Failure Analysis under Synthetic Camera Corruption",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    for temp_path in [temp_clean, temp_corr]:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_yolo_visual_failure_analysis(
    dataset_root=DATASET_ROOT,
    generated_dirs=None,
    output_dir=OUTPUT_DIR,
    yolo_model_path=YOLO_MODEL_PATH,
    limit_per_condition=10,
    log_callback=None,
):
    from ultralytics import YOLO

    clean_dir = os.path.join(dataset_root, "clean")

    if generated_dirs is None:
        generated_dirs = default_generated_dirs(dataset_root)

    os.makedirs(output_dir, exist_ok=True)

    log_message("\nLoading YOLOv8 model for visual comparisons...\n", log_callback)
    model = YOLO(yolo_model_path)

    clean_files = list_images(clean_dir)
    output_paths = []

    for condition, condition_dir in generated_dirs.items():
        if not os.path.isdir(condition_dir):
            log_message(f"Skipping missing condition dir: {condition_dir}", log_callback)
            continue

        condition_output = os.path.join(output_dir, condition)
        os.makedirs(condition_output, exist_ok=True)

        generated_files = list_images(condition_dir)
        selected_files = generated_files[:limit_per_condition]

        for gname in selected_files:
            matched_clean = find_matching_clean(gname, clean_files)

            if matched_clean is None:
                continue

            clean_path = os.path.join(clean_dir, matched_clean)
            generated_path = os.path.join(condition_dir, gname)
            output_path = os.path.join(
                condition_output,
                f"{Path(gname).stem}_yolo_comparison.png",
            )

            create_comparison_figure(
                model=model,
                clean_img_path=clean_path,
                corrupted_img_path=generated_path,
                condition=condition,
                output_path=output_path,
                temp_dir=output_dir,
            )

            output_paths.append(output_path)
            log_message(f"Saved: {output_path}", log_callback)

    return {
        "output_dir": output_dir,
        "output_paths": output_paths,
    }


def main():
    run_yolo_visual_failure_analysis()


if __name__ == "__main__":
    main()
