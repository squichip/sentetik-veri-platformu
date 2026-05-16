import os
from pathlib import Path


DATASET_ROOT = "robustness_dataset"
RESULTS_DIR = "yolo_results"
YOLO_MODEL_PATH = "yolov8n.pt"
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


def get_detection_stats(model, image_path):
    import numpy as np

    results = model(image_path, verbose=False)
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "object_count": 0,
            "avg_confidence": 0.0,
        }

    confidences = boxes.conf.cpu().numpy()

    return {
        "object_count": len(confidences),
        "avg_confidence": float(np.mean(confidences)),
    }


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


def find_matching_clean(generated_name, clean_names):
    for clean_name in clean_names:
        clean_stem = Path(clean_name).stem

        if clean_stem in generated_name:
            return clean_name

    return None


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


def evaluate_clean_images(model, clean_dir, log_callback=None):
    clean_stats = {}

    log_message("\nRunning clean detections...\n", log_callback)

    clean_files = list_images(clean_dir)
    log_message(f"Clean YOLO dosya sayisi: {len(clean_files)}", log_callback)

    for fname in progress_iter(clean_files, log_callback=log_callback):
        path = os.path.join(clean_dir, fname)
        clean_stats[fname] = get_detection_stats(model, path)

    return clean_stats


def evaluate_generated_images(
    model,
    generated_dirs,
    clean_stats,
    log_callback=None,
):
    rows = []

    log_message("\nRunning robustness evaluation...\n", log_callback)

    for condition_name, condition_dir in generated_dirs.items():
        if not os.path.isdir(condition_dir):
            log_message(f"Skipping missing condition dir: {condition_dir}", log_callback)
            continue

        log_message(f"\nCondition: {condition_name}\n", log_callback)

        generated_files = list_images(condition_dir)
        log_message(
            f"{condition_name} YOLO dosya sayisi: {len(generated_files)}",
            log_callback,
        )

        for index, gname in enumerate(
            progress_iter(generated_files, log_callback=log_callback),
            start=1,
        ):
            generated_path = os.path.join(condition_dir, gname)
            generated_stats = get_detection_stats(model, generated_path)

            matched_clean = find_matching_clean(gname, clean_stats.keys())

            if matched_clean is None:
                continue

            clean_result = clean_stats[matched_clean]

            clean_count = clean_result["object_count"]
            generated_count = generated_stats["object_count"]

            clean_conf = clean_result["avg_confidence"]
            generated_conf = generated_stats["avg_confidence"]

            rows.append({
                "condition": condition_name,
                "generated_file": gname,
                "clean_reference": matched_clean,
                "clean_object_count": clean_count,
                "generated_object_count": generated_count,
                "clean_avg_confidence": clean_conf,
                "generated_avg_confidence": generated_conf,
                "detection_drop": clean_count - generated_count,
                "confidence_drop": clean_conf - generated_conf,
            })

            if log_callback and (
                index == 1 or index == len(generated_files) or index % 25 == 0
            ):
                log_message(
                    f"{condition_name}: {index}/{len(generated_files)} YOLO tamamlandi",
                    log_callback,
                )

    return rows


def save_yolo_outputs(rows, results_dir=RESULTS_DIR, log_callback=None):
    import pandas as pd

    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(results_dir, "yolo_robustness_metrics.csv")
    df.to_csv(csv_path, index=False)
    log_message(f"\nCSV saved: {csv_path}", log_callback)

    if df.empty:
        log_message("No matched YOLO rows found; summary and plots skipped.", log_callback)
        return {
            "metrics_csv": csv_path,
            "summary_csv": None,
            "plots": [],
        }

    summary = df.groupby("condition").mean(numeric_only=True)

    summary_csv = os.path.join(results_dir, "yolo_summary.csv")
    summary.to_csv(summary_csv)
    log_message(f"Summary CSV saved: {summary_csv}", log_callback)

    plots = []

    plots.append(save_bar_plot(
        summary=summary,
        column="generated_avg_confidence",
        title="YOLO Average Confidence under Corruptions",
        ylabel="Average Confidence",
        output_path=os.path.join(results_dir, "yolo_confidence.png"),
        log_callback=log_callback,
    ))

    plots.append(save_bar_plot(
        summary=summary,
        column="generated_object_count",
        title="YOLO Detection Count under Corruptions",
        ylabel="Average Detection Count",
        output_path=os.path.join(results_dir, "yolo_detection_count.png"),
        log_callback=log_callback,
    ))

    plots.append(save_bar_plot(
        summary=summary,
        column="confidence_drop",
        title="YOLO Confidence Drop",
        ylabel="Confidence Drop",
        output_path=os.path.join(results_dir, "yolo_confidence_drop.png"),
        log_callback=log_callback,
    ))

    return {
        "metrics_csv": csv_path,
        "summary_csv": summary_csv,
        "plots": plots,
    }


def save_bar_plot(summary, column, title, ylabel, output_path, log_callback=None):
    plt = load_pyplot()

    plt.figure(figsize=(8, 5))
    summary[column].plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    log_message(f"Plot saved: {output_path}", log_callback)
    return output_path


def run_yolo_evaluation(
    dataset_root=DATASET_ROOT,
    results_dir=RESULTS_DIR,
    yolo_model_path=YOLO_MODEL_PATH,
    generated_dirs=None,
    log_callback=None,
):
    clean_dir = os.path.join(dataset_root, "clean")

    if generated_dirs is None:
        generated_dirs = default_generated_dirs(dataset_root)

    from ultralytics import YOLO

    log_message("\nLoading YOLOv8 model...\n", log_callback)
    model = YOLO(yolo_model_path)
    log_message("YOLO model loaded.\n", log_callback)

    clean_stats = evaluate_clean_images(
        model=model,
        clean_dir=clean_dir,
        log_callback=log_callback,
    )

    rows = evaluate_generated_images(
        model=model,
        generated_dirs=generated_dirs,
        clean_stats=clean_stats,
        log_callback=log_callback,
    )

    outputs = save_yolo_outputs(
        rows=rows,
        results_dir=results_dir,
        log_callback=log_callback,
    )

    log_message("\nYOLO ROBUSTNESS EVALUATION FINISHED.\n", log_callback)

    return {
        "rows": rows,
        "outputs": outputs,
    }


def main():
    run_yolo_evaluation()


if __name__ == "__main__":
    main()
