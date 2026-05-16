import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "robustness_metrics.csv")

df = pd.read_csv(CSV_PATH)

condition_order = [
    "blur_high",
    "brightness_high",
    "occlusion_high",
]

condition_labels = {
    "blur_high": "Blur High",
    "brightness_high": "Brightness High",
    "occlusion_high": "Occlusion High",
}

df["condition_label"] = df["condition"].map(condition_labels)

summary = (
    df.groupby("condition_label")
    .agg(
        prediction_iou_mean=("prediction_iou", "mean"),
        prediction_iou_std=("prediction_iou", "std"),
        pixel_agreement_mean=("pixel_agreement", "mean"),
        pixel_agreement_std=("pixel_agreement", "std"),
        robustness_drop_mean=("robustness_drop", "mean"),
        robustness_drop_std=("robustness_drop", "std"),
        distribution_shift_mean=("distribution_shift", "mean"),
        distribution_shift_std=("distribution_shift", "std"),
    )
    .reindex([condition_labels[c] for c in condition_order])
)

summary.to_csv(os.path.join(RESULTS_DIR, "academic_summary_statistics.csv"))


def academic_bar_plot(
    values,
    errors,
    ylabel,
    title,
    output_name,
    ylim=None,
):
    plt.figure(figsize=(8, 5), dpi=300)

    bars = plt.bar(
        summary.index,
        values,
        yerr=errors,
        capsize=5,
        edgecolor="black",
        linewidth=1,
    )

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Synthetic Corruption Type", fontsize=12)
    plt.title(title, fontsize=13, pad=12)

    if ylim:
        plt.ylim(ylim)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


academic_bar_plot(
    values=summary["prediction_iou_mean"],
    errors=summary["prediction_iou_std"],
    ylabel="Prediction Consistency IoU",
    title="Semantic Segmentation Consistency under Synthetic Corruptions",
    output_name="academic_prediction_iou.png",
    ylim=(0, 1),
)

academic_bar_plot(
    values=summary["pixel_agreement_mean"],
    errors=summary["pixel_agreement_std"],
    ylabel="Pixel Agreement",
    title="Pixel-Level Agreement between Clean and Corrupted Predictions",
    output_name="academic_pixel_agreement.png",
    ylim=(0, 1),
)

academic_bar_plot(
    values=summary["robustness_drop_mean"],
    errors=summary["robustness_drop_std"],
    ylabel="Robustness Drop",
    title="Segmentation Robustness Degradation by Corruption Type",
    output_name="academic_robustness_drop.png",
    ylim=(0, 1),
)

academic_bar_plot(
    values=summary["distribution_shift_mean"],
    errors=summary["distribution_shift_std"],
    ylabel="Class Distribution Shift",
    title="Predicted Class Distribution Shift under Synthetic Corruptions",
    output_name="academic_distribution_shift.png",
)
