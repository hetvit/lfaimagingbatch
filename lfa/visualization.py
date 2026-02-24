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
    typical_T = float(np.median(baseline) + k * sigma) # typical_T is the median baseline across all rows
    axes[1].hist(row_score, bins=60, density=True, histtype="bar", alpha=0.85)
    axes[1].axvline(typical_T, linestyle="--", linewidth=2, color="red",
                    label=f"Median T across all rows = {typical_T:.1f}")
    axes[1].set_title(f"Row {dbg['stat']} scores")
    axes[1].set_xlabel("Row score")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # # 2) Row score trace
    # y = np.arange(len(row_score))
    # axes[2].plot(row_score, y, linewidth=1.5, label="row_score")
    # axes[2].plot(baseline, y, linewidth=1.5, label="baseline")
    # axes[2].plot(T_row, y, linewidth=1.5, label="threshold")
    # axes[2].invert_yaxis()
    # axes[2].set_title("Row-wise threshold profile")
    # axes[2].set_xlabel("Value")
    # axes[2].set_ylabel("Row index")
    # axes[2].legend()
    
    # 2) Row score trace
    y = np.arange(len(row_score))

    axes[2].plot(row_score, y, linewidth=1.5, label="row_score")  # solid
    axes[2].plot(baseline,  y, linestyle="--", linewidth=2.0, label="baseline")  # dashed, thicker
    axes[2].plot(T_row,     y, linestyle=":",  linewidth=2.0, label="threshold")  # dotted

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
    # plt.show()
    return fig


def plot_inverted_vs_corrected(an, cmap="hot"):
    """
    Simple 2-panel sanity check:
    inverted image vs background-subtracted image
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



# Used in batch_analyze
def plot_lfa_debug_panels(an, save_dir="lfa_strip_panels"):
    """
    7-panel debug figure combining rowwise threshold info with image stages
    for a single LFA strip:

      0) Original image (RGB)                     [matches plot_rowwise_threshold_debug panel 0]
      1) Row-wise threshold profile (trace)       [matches plot_rowwise_threshold_debug panel 2]
      2) Inverted image                           (preprocessed)
      3) Background image that was subtracted
      4) Corrected image (no mask)
      5) Corrected image + mask overlay           [matches plot_rowwise_threshold_debug panel 3]
      6) Binary mask (lines dark, bordered)       [matches plot_rowwise_threshold_debug panel 4]
      7) Original image + Mask overlay

    This mirrors plot_rowwise_threshold_debug while also exposing the
    background estimate used for correction, the inverted intermediate,
    and both corrected-with/without-mask views.
    """

    # Sanity checks
    if an.original_image is None:
        raise ValueError("original_image is None; did SimpleLFAAnalyzer load correctly?")
    if an.inverted_image is None:
        raise ValueError("inverted_image is None; run preprocess() first.")
    if an.corrected_image is None or an.background_image is None:
        raise ValueError("corrected_image or background_image is None; run subtract_background() first.")
    if an.binary_mask is None:
        raise ValueError("binary_mask is None; run rowwise_binarize_corrected() first.")
    if not hasattr(an, "_rowwise_debug"):
        raise ValueError("Missing _rowwise_debug; run rowwise_binarize_corrected() first.")

    # Unpack rowwise debug for the row trace panel
    dbg       = an._rowwise_debug
    row_score = dbg["row_score"]
    baseline  = dbg["baseline"]
    T_row     = dbg["T_row"]

    orig_bgr = an.original_image
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    inv      = an.inverted_image
    corr     = an.corrected_image
    mask     = an.binary_mask
    bg       = an.background_image

    H, W = corr.shape[:2]

    # 1×7 figure, with constrained layout so plots sit closer together
    fig, axes = plt.subplots(1, 8, figsize=(24, 4), constrained_layout=True)

    # 0) Original image (RGB) -- matches plot_rowwise_threshold_debug panel 0
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 1) Row score trace (row-wise threshold profile) -- matches panel 2
    y = np.arange(len(row_score))
    axes[1].plot(row_score, y, linewidth=1.5, label="row_score")           # solid
    axes[1].plot(baseline,  y, linestyle="--", linewidth=2.0, label="baseline")   # dashed, thicker
    axes[1].plot(T_row,     y, linestyle=":",  linewidth=2.0, label="threshold")  # dotted

    axes[1].invert_yaxis()
    axes[1].set_title("Row-wise\nthreshold profile")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Row index")
    axes[1].legend(fontsize=8)

    # 2) Inverted image (preprocessed)
    axes[2].imshow(inv, cmap="hot")
    axes[2].set_title("Inverted\n(Dark → Bright)", fontsize=10)
    axes[2].axis("off")

    # 3) Background image that was subtracted
    axes[3].imshow(bg, cmap="hot")
    axes[3].set_title("Background\n(Estimated)", fontsize=10)
    axes[3].axis("off")

    # 4) Corrected image (no mask)
    axes[4].imshow(corr, cmap="hot")
    axes[4].set_title("Corrected", fontsize=10)
    axes[4].axis("off")

    # 5) Corrected + mask overlay -- matches plot_rowwise_threshold_debug panel 3
    axes[5].imshow(corr, cmap="hot")
    axes[5].imshow(mask, alpha=0.35)
    axes[5].set_title("Corrected\n+ Mask Overlay", fontsize=10)
    axes[5].axis("off")

    # 6) Binary mask (lines dark) -- matches plot_rowwise_threshold_debug panel 4
    binary_vis = cv2.bitwise_not(mask)  # lines dark
    bordered   = cv2.copyMakeBorder(
        binary_vis, 2, 2, 2, 2,
        cv2.BORDER_CONSTANT, value=0
    )
    axes[6].imshow(bordered, cmap="gray", vmin=0, vmax=255)
    axes[6].set_title("Binarized\n(Lines Dark)", fontsize=10)
    axes[6].axis("off")
    
    # # 7) Original + mask overlay (only highlight mask==1 pixels)
    # axes[7].imshow(orig_rgb)
    # # Create RGBA overlay for mask highlighting
    # overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    # # Choose highlight color (red here; change if you prefer)
    # color = np.array([0.0, 1.0, 0.0])  # RGB normalized 0–1
    # # Boolean mask for "1" regions
    # mask_1 = (mask > 0).astype(np.float32)
    # # Assign color only where mask==1
    # overlay[..., :3] = color
    # overlay[..., 3] = 0.25 * mask_1     # alpha applied ONLY to mask==1
    # axes[7].imshow(overlay)
    # axes[7].set_title("Original\n+ Mask Overlay", fontsize=10)
    # axes[7].axis("off")
    
     # 7) Original + mask overlay + sampling regions
    axes[7].imshow(orig_rgb)

    # Light green overlay for the full binary mask (where mask == 1)
    overlay_full = np.zeros((*mask.shape, 4), dtype=np.float32)
    mask_1 = (mask > 0).astype(np.float32)
    overlay_full[..., :3] = np.array([0.0, 1.0, 0.0])   # green
    overlay_full[..., 3] = 0.20 * mask_1
    axes[7].imshow(overlay_full)

    # --- Sampling regions using make_band_sampling_mask() ---
    from lfa.image_processing import _runs_from_hits, make_band_sampling_mask

    hits = an._rowwise_debug["row_hits_clean"].astype(bool)
    runs = _runs_from_hits(hits)
    mid = H // 2

    # These should match band_mean_intensity_on_original defaults
    row_radius = 10
    edge_margin_frac = 0.01

    for (s, e) in runs:
        # Build a sampling mask for this specific band
        sample_mask = make_band_sampling_mask(
            an,
            (s, e),
            edge_margin_frac=edge_margin_frac,
            row_radius=row_radius,
        )

        # Choose color based on half: top = control, bottom = test
        center = 0.5 * (s + e)
        color = np.array([0.0, 1.0, 1.0]) if center < mid else np.array([1.0, 0.0, 1.0])  # cyan / magenta

        sample_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        sample_overlay[..., :3] = color
        sample_overlay[..., 3] = 0.40 * (sample_mask > 0).astype(np.float32)

        axes[7].imshow(sample_overlay)

    axes[7].set_title("Original\n+ Mask + ROIs", fontsize=10)
    axes[7].axis("off")
    

    # Save one PNG per strip
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(an.image_path).stem
    out_path = save_dir / f"{stem}_lfa_debug_panels.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[DEBUG] Saved LFA debug panels to {out_path}")
    return out_path