import os
from pathlib import Path


DATASET_ROOT = "robustness_dataset"
SEG_OUTPUT_ROOT = "segmentation_outputs"
OUTPUT_DIR = "segmentation_visual_comparisons"
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


def default_generated_seg_dirs(seg_output_root=SEG_OUTPUT_ROOT):
    return {
        "blur_high": os.path.join(seg_output_root, "blur_high"),
        "brightness_high": os.path.join(seg_output_root, "brightness_high"),
        "occlusion_high": os.path.join(seg_output_root, "occlusion_high"),
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


def read_rgb(path):
    import cv2

    img = cv2.imread(path)

    if img is None:
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_matching_clean(generated_name, clean_files):
    for clean_name in clean_files:
        clean_stem = Path(clean_name).stem

        if clean_stem in generated_name:
            return clean_name

    return None


def find_segmentation_file(seg_dir, image_stem):
    if not os.path.isdir(seg_dir):
        return None

    files = os.listdir(seg_dir)

    for fname in files:
        if image_stem in fname and fname.lower().endswith(IMAGE_EXTENSIONS):
            return os.path.join(seg_dir, fname)

    return None


def create_segmentation_figure(
    clean_image_path,
    clean_seg_path,
    corrupted_image_path,
    corrupted_seg_path,
    condition,
    output_path,
):
    plt = load_pyplot()

    clean_img = read_rgb(clean_image_path)
    clean_seg = read_rgb(clean_seg_path)
    corr_img = read_rgb(corrupted_image_path)
    corr_seg = read_rgb(corrupted_seg_path)

    if clean_img is None or clean_seg is None or corr_img is None or corr_seg is None:
        return False

    plt.figure(figsize=(14, 7), dpi=300)

    plt.subplot(2, 2, 1)
    plt.imshow(clean_img)
    plt.title("Clean Input Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(clean_seg)
    plt.title("Clean Segmentation Prediction")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(corr_img)
    plt.title(f"{condition.replace('_', ' ').title()} Input Image")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(corr_seg)
    plt.title(f"{condition.replace('_', ' ').title()} Segmentation Prediction")
    plt.axis("off")

    plt.suptitle(
        "Semantic Segmentation Degradation under Synthetic Camera Corruption",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return True


def run_segmentation_visual_comparison(
    dataset_root=DATASET_ROOT,
    generated_image_dirs=None,
    seg_output_root=SEG_OUTPUT_ROOT,
    output_dir=OUTPUT_DIR,
    limit_per_condition=10,
    log_callback=None,
):
    clean_image_dir = os.path.join(dataset_root, "clean")
    clean_seg_dir = os.path.join(seg_output_root, "clean")

    if generated_image_dirs is None:
        generated_image_dirs = default_generated_dirs(dataset_root)

    os.makedirs(output_dir, exist_ok=True)

    clean_files = list_images(clean_image_dir)
    output_paths = []

    for condition, generated_dir in generated_image_dirs.items():
        if not os.path.isdir(generated_dir):
            log_message(f"Skipping missing condition dir: {generated_dir}", log_callback)
            continue

        condition_output = os.path.join(output_dir, condition)
        os.makedirs(condition_output, exist_ok=True)

        generated_files = list_images(generated_dir)
        selected_files = generated_files[:limit_per_condition]

        for gname in selected_files:
            matched_clean = find_matching_clean(gname, clean_files)

            if matched_clean is None:
                continue

            clean_stem = Path(matched_clean).stem
            generated_stem = Path(gname).stem

            clean_image_path = os.path.join(clean_image_dir, matched_clean)
            corrupted_image_path = os.path.join(generated_dir, gname)

            clean_seg_path = find_segmentation_file(clean_seg_dir, clean_stem)
            corrupted_seg_path = find_segmentation_file(
                os.path.join(seg_output_root, condition),
                generated_stem,
            )

            if clean_seg_path is None or corrupted_seg_path is None:
                continue

            output_path = os.path.join(
                condition_output,
                f"{generated_stem}_segmentation_comparison.png",
            )

            ok = create_segmentation_figure(
                clean_image_path=clean_image_path,
                clean_seg_path=clean_seg_path,
                corrupted_image_path=corrupted_image_path,
                corrupted_seg_path=corrupted_seg_path,
                condition=condition,
                output_path=output_path,
            )

            if ok:
                output_paths.append(output_path)
                log_message(f"Saved: {output_path}", log_callback)

    return {
        "output_dir": output_dir,
        "output_paths": output_paths,
    }


def main():
    run_segmentation_visual_comparison()


if __name__ == "__main__":
    main()
