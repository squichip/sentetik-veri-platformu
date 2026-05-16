import os
from collections import Counter
from pathlib import Path


MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DATASET_ROOT = "robustness_dataset"
OUTPUT_DIR = "segmentation_outputs"
RESULTS_DIR = "results"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def default_generated_dirs(dataset_root=DATASET_ROOT):
    return {
        "blur_high": os.path.join(dataset_root, "generated", "blur_high"),
        "brightness_high": os.path.join(dataset_root, "generated", "brightness_high"),
        "occlusion_high": os.path.join(dataset_root, "generated", "occlusion_high"),
    }


def default_upscaled_generated_dirs(dataset_root=DATASET_ROOT):
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


def pick_device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


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


def load_image(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def load_segformer(model_name=MODEL_NAME, device=None, log_callback=None):
    import torch
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    if device is None:
        device = pick_device()

    log_message("\nLoading SegFormer model...", log_callback)

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    log_message("Model loaded.\n", log_callback)

    return processor, model, device, torch


def predict_segmentation(image, processor, model, device, torch_module):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch_module.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch_module.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return pred_seg.cpu().numpy()


def colorize_mask(mask):
    import numpy as np

    np.random.seed(42)

    colors = np.random.randint(
        0,
        255,
        size=(256, 3),
        dtype=np.uint8,
    )

    return colors[mask]


def save_segmentation_visual(image, mask, out_path):
    import cv2
    import numpy as np

    image_np = np.array(image)
    color_mask = colorize_mask(mask)

    overlay = cv2.addWeighted(
        image_np,
        0.5,
        color_mask,
        0.5,
        0,
    )

    cv2.imwrite(
        str(out_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def pixel_agreement(mask1, mask2):
    same = (mask1 == mask2).sum()
    total = mask1.size

    return same / total


def prediction_iou(mask1, mask2):
    import numpy as np

    classes = np.union1d(np.unique(mask1), np.unique(mask2))
    ious = []

    for cls in classes:
        intersection = np.logical_and(mask1 == cls, mask2 == cls).sum()
        union = np.logical_or(mask1 == cls, mask2 == cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return np.mean(ious)


def class_distribution(mask):
    counts = Counter(mask.flatten())
    total = mask.size

    dist = {}

    for cls, cnt in counts.items():
        dist[int(cls)] = cnt / total

    return dist


def distribution_shift(dist1, dist2):
    classes = set(dist1.keys()).union(set(dist2.keys()))

    shift = 0.0

    for cls in classes:
        shift += abs(dist1.get(cls, 0) - dist2.get(cls, 0))

    return shift


def find_matching_clean(generated_name, clean_names):
    for clean_name in clean_names:
        clean_stem = Path(clean_name).stem

        if clean_stem in generated_name:
            return clean_name

    return None


def resize_mask_like(generated_mask, clean_mask):
    if generated_mask.shape == clean_mask.shape:
        return generated_mask

    import cv2
    import numpy as np

    return cv2.resize(
        generated_mask.astype(np.uint8),
        (clean_mask.shape[1], clean_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def evaluate_clean_segmentations(
    clean_dir,
    output_dir,
    processor,
    model,
    device,
    torch_module,
    log_callback=None,
):
    clean_predictions = {}

    log_message("\nRunning clean segmentation predictions...\n", log_callback)

    clean_files = list_images(clean_dir)
    log_message(f"Clean segmentasyon dosya sayisi: {len(clean_files)}", log_callback)

    clean_output_dir = os.path.join(output_dir, "clean")
    os.makedirs(clean_output_dir, exist_ok=True)

    for index, fname in enumerate(
        progress_iter(clean_files, log_callback=log_callback),
        start=1,
    ):
        clean_path = os.path.join(clean_dir, fname)
        image = load_image(clean_path)
        mask = predict_segmentation(image, processor, model, device, torch_module)

        clean_predictions[fname] = mask

        out_path = os.path.join(clean_output_dir, f"{Path(fname).stem}_seg.png")
        save_segmentation_visual(image, mask, out_path)

        if log_callback and (index == 1 or index == len(clean_files) or index % 25 == 0):
            log_message(
                f"Clean segmentasyon: {index}/{len(clean_files)} tamamlandi",
                log_callback,
            )

    return clean_predictions


def evaluate_generated_segmentations(
    generated_dirs,
    output_dir,
    clean_predictions,
    processor,
    model,
    device,
    torch_module,
    log_callback=None,
):
    rows = []

    log_message("\nRunning robustness evaluation...\n", log_callback)

    for condition_name, condition_dir in generated_dirs.items():
        if not os.path.isdir(condition_dir):
            log_message(f"Skipping missing condition dir: {condition_dir}", log_callback)
            continue

        log_message(f"\nCondition: {condition_name}\n", log_callback)

        output_condition_dir = os.path.join(output_dir, condition_name)
        os.makedirs(output_condition_dir, exist_ok=True)

        generated_files = list_images(condition_dir)
        log_message(
            f"{condition_name} segmentasyon dosya sayisi: {len(generated_files)}",
            log_callback,
        )

        for index, gname in enumerate(
            progress_iter(generated_files, log_callback=log_callback),
            start=1,
        ):
            generated_path = os.path.join(condition_dir, gname)
            image = load_image(generated_path)
            generated_mask = predict_segmentation(
                image,
                processor,
                model,
                device,
                torch_module,
            )

            save_path = os.path.join(
                output_condition_dir,
                f"{Path(gname).stem}_seg.png",
            )

            save_segmentation_visual(image, generated_mask, save_path)

            matched_clean = find_matching_clean(gname, clean_predictions.keys())

            if matched_clean is None:
                continue

            clean_mask = clean_predictions[matched_clean]
            generated_mask = resize_mask_like(generated_mask, clean_mask)

            agreement = pixel_agreement(clean_mask, generated_mask)
            pred_iou = prediction_iou(clean_mask, generated_mask)

            clean_dist = class_distribution(clean_mask)
            gen_dist = class_distribution(generated_mask)
            shift = distribution_shift(clean_dist, gen_dist)

            rows.append({
                "condition": condition_name,
                "generated_file": gname,
                "clean_reference": matched_clean,
                "pixel_agreement": agreement,
                "prediction_iou": pred_iou,
                "distribution_shift": shift,
                "robustness_drop": 1.0 - pred_iou,
            })

            if log_callback and (
                index == 1 or index == len(generated_files) or index % 25 == 0
            ):
                log_message(
                    f"{condition_name}: {index}/{len(generated_files)} segmentasyon tamamlandi",
                    log_callback,
                )

    return rows


def save_segmentation_outputs(rows, results_dir=RESULTS_DIR, log_callback=None):
    import pandas as pd

    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(results_dir, "robustness_metrics.csv")
    df.to_csv(csv_path, index=False)
    log_message(f"\nCSV saved: {csv_path}", log_callback)

    if df.empty:
        log_message(
            "No matched segmentation rows found; summary and plots skipped.",
            log_callback,
        )
        return {
            "metrics_csv": csv_path,
            "summary_csv": None,
            "plots": [],
        }

    summary = df.groupby("condition").mean(numeric_only=True)

    summary_csv = os.path.join(results_dir, "robustness_summary.csv")
    summary.to_csv(summary_csv)
    log_message(f"Summary CSV saved: {summary_csv}", log_callback)

    plots = []

    plots.append(save_bar_plot(
        summary=summary,
        column="prediction_iou",
        title="Prediction IoU by Condition",
        ylabel="Prediction IoU",
        output_path=os.path.join(results_dir, "prediction_iou_comparison.png"),
        log_callback=log_callback,
    ))

    plots.append(save_bar_plot(
        summary=summary,
        column="pixel_agreement",
        title="Pixel Agreement by Condition",
        ylabel="Pixel Agreement",
        output_path=os.path.join(results_dir, "pixel_agreement_comparison.png"),
        log_callback=log_callback,
    ))

    plots.append(save_bar_plot(
        summary=summary,
        column="robustness_drop",
        title="Robustness Drop by Condition",
        ylabel="Robustness Drop",
        output_path=os.path.join(results_dir, "robustness_drop_comparison.png"),
        log_callback=log_callback,
    ))

    return {
        "metrics_csv": csv_path,
        "summary_csv": summary_csv,
        "plots": plots,
    }


def save_bar_plot(summary, column, title, ylabel, output_path, log_callback=None):
    plt = load_pyplot()

    plt.figure(figsize=(10, 5))
    summary[column].plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    log_message(f"Plot saved: {output_path}", log_callback)
    return output_path


def run_segmentation_evaluation(
    dataset_root=DATASET_ROOT,
    output_dir=OUTPUT_DIR,
    results_dir=RESULTS_DIR,
    model_name=MODEL_NAME,
    generated_dirs=None,
    device=None,
    log_callback=None,
):
    clean_dir = os.path.join(dataset_root, "clean")

    if generated_dirs is None:
        generated_dirs = default_generated_dirs(dataset_root)

    processor, model, device, torch_module = load_segformer(
        model_name=model_name,
        device=device,
        log_callback=log_callback,
    )

    clean_predictions = evaluate_clean_segmentations(
        clean_dir=clean_dir,
        output_dir=output_dir,
        processor=processor,
        model=model,
        device=device,
        torch_module=torch_module,
        log_callback=log_callback,
    )

    rows = evaluate_generated_segmentations(
        generated_dirs=generated_dirs,
        output_dir=output_dir,
        clean_predictions=clean_predictions,
        processor=processor,
        model=model,
        device=device,
        torch_module=torch_module,
        log_callback=log_callback,
    )

    outputs = save_segmentation_outputs(
        rows=rows,
        results_dir=results_dir,
        log_callback=log_callback,
    )

    log_message("\nROBUSTNESS EVALUATION FINISHED.\n", log_callback)

    return {
        "rows": rows,
        "outputs": outputs,
    }


def main():
    run_segmentation_evaluation()


if __name__ == "__main__":
    main()
