# lfa/viz.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_summary(an, save_path=None):
    """
    2-panel summary visualization:
      Panel 1: Original image with midpoint line
      Panel 2: Corrected image with detected band rows overlaid
    Plus a second figure showing the background subtraction stages.
    """
    if an.corrected_image is None:
        raise ValueError("Run the full pipeline first.")

    H = an.corrected_image.shape[0]
    mid = H // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Panel 1: Original
    ax1.imshow(cv2.cvtColor(an.original_image, cv2.COLOR_BGR2RGB))
    ax1.axhline(y=mid, color="yellow", linewidth=2, linestyle="--", label="Midpoint")
    ax1.set_title("Original Image", fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Corrected + band mask overlay
    ax2.imshow(an.corrected_image, cmap="hot")
    ax2.axhline(y=mid, color="cyan", linewidth=1.5, linestyle="--", alpha=0.6, label="Midpoint")

    if an.binary_mask is not None and hasattr(an, "_rowwise_debug"):
        from lfa.image_processing import _runs_from_hits
        hits = an._rowwise_debug["row_hits_clean"].astype(bool)
        runs = _runs_from_hits(hits)
        for s, e in runs:
            center = 0.5 * (s + e)
            color = "cyan" if center < mid else "lime"
            label = "Control" if center < mid else "Test"
            ax2.axhspan(s, e, alpha=0.35, color=color, label=label)

    ax2.set_title("Corrected + Detected Bands", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.axis("off")

    fig.suptitle(
        Path(an.image_path).name,
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()

    # Second figure: background subtraction stages
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(an.inverted_image, cmap="hot")
    axes[0].set_title("Inverted")
    axes[0].axis("off")
    axes[1].imshow(an.background_image, cmap="hot")
    axes[1].set_title("Estimated Background")
    axes[1].axis("off")
    axes[2].imshow(an.corrected_image, cmap="hot")
    axes[2].set_title("Corrected (Inverted - Background)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

    return fig


def plot_rowwise_threshold_debug(an):
    """
    5-panel debug figure for the rowwise pipeline:
      0) Original image
      1) Histogram of row scores
      2) Row score trace with baseline and threshold
      3) Corrected image with mask overlay
      4) Binary mask (lines dark)
    """
    if an.corrected_image is None:
        raise ValueError("Run subtract_background() first.")
    if not hasattr(an, "_rowwise_debug"):
        raise ValueError("Run rowwise_binarize_corrected() first.")

    dbg = an._rowwise_debug
    row_score = dbg["row_score"]
    baseline = dbg["baseline"]
    T_row = dbg["T_row"]
    sigma = dbg["sigma"]
    k = dbg["k"]

    fig, axes = plt.subplots(1, 5, figsize=(26, 4))

    # 0) Original
    axes[0].imshow(cv2.cvtColor(an.original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 1) Histogram of row scores
    typical_T = float(np.median(baseline) + k * sigma)
    axes[1].hist(row_score, bins=60, density=True, histtype="bar", alpha=0.85)
    axes[1].axvline(typical_T, linestyle="--", linewidth=2, color="red",
                    label=f"Typical T ≈ {typical_T:.1f}")
    axes[1].set_title(f"Row {dbg['stat']} scores")
    axes[1].set_xlabel("Row score")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # 2) Row score trace
    y = np.arange(len(row_score))
    axes[2].plot(row_score, y, linewidth=1.5, label="row_score")
    axes[2].plot(baseline, y, linewidth=1.5, label="baseline")
    axes[2].plot(T_row, y, linewidth=1.5, label="threshold")
    axes[2].invert_yaxis()
    axes[2].set_title("Row-wise threshold profile")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Row index")
    axes[2].legend()

    # 3) Corrected + mask overlay
    axes[3].imshow(an.corrected_image, cmap="hot")
    if an.binary_mask is not None:
        axes[3].imshow(an.binary_mask, alpha=0.35)
    axes[3].set_title("Corrected + Mask Overlay")
    axes[3].axis("off")

    # 4) Binary mask
    binary_vis = cv2.bitwise_not(an.binary_mask)
    bordered = cv2.copyMakeBorder(binary_vis, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    axes[4].imshow(bordered, cmap="gray", vmin=0, vmax=255)
    axes[4].set_title("Binarized (Lines Dark)")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def plot_inverted_vs_corrected(an, cmap="hot"):
    """
    Simple 2-panel sanity check:
    inverted image vs background-subtracted image.
    """
    if an.inverted_image is None:
        raise ValueError("Run preprocess() first.")
    if an.corrected_image is None:
        raise ValueError("Run subtract_background() first.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(an.inverted_image, cmap=cmap)
    axes[0].set_title("Inverted Image\n(Dark lines → Bright)", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(an.corrected_image, cmap=cmap)
    axes[1].set_title("After Background Subtraction", fontsize=12)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return fig
