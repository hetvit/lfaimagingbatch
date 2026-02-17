# lfa/viz.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_summary(an, save_path=None):
    """Create simple 2-panel visualization: original and enhanced only"""
    if an.inverted_image is None:
        raise ValueError("Run analyze() (or preprocess()) first so an.inverted_image exists.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    height = an.inverted_image.shape[0]
    mid_point = height // 2
    skip = 3

    # Panel 1: Original image with split line
    ax1.imshow(cv2.cvtColor(an.original_image, cv2.COLOR_BGR2RGB))
    ax1.axhline(y=mid_point, color='yellow', linewidth=3, linestyle='--', label='Split line')
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.axis('off')

    # Panel 2: Enhanced image with detected lines
    ax2.imshow(an.corrected_image if an.corrected_image is not None else an.inverted_image, cmap='hot')
    ax2.axhline(y=mid_point, color='cyan', linewidth=2, linestyle='--', label='Split', alpha=0.7)

    # Show skipped regions
    ax2.axhspan(0, skip, alpha=0.3, color='gray', label='Skipped (edge)')
    ax2.axhspan(mid_point-skip, mid_point+skip, alpha=0.3, color='gray')
    ax2.axhspan(height-skip, height, alpha=0.3, color='gray')

    # Mark detected control line
    if an.control_line_pos is not None:
        # Control line position is relative to top half (which starts at skip)
        actual_control_pos = skip + an.control_line_pos
        ax2.axhline(y=actual_control_pos, color='cyan', linewidth=3, label='Control Line')

    # Mark detected test line
    if an.test_line_pos is not None:
        # Test line position is relative to bottom half (which starts at mid_point+skip)
        actual_test_pos = mid_point + skip + an.test_line_pos
        if not an.is_negative:
            ax2.axhline(y=actual_test_pos, color='lime', linewidth=3, label='Test Line')
        else:
            ax2.axhline(y=actual_test_pos, color='red', linewidth=3, linestyle=':', label='Test (below threshold)')

    ax2.set_title('Enhanced Image\n(Brighter = darker lines)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.axis('off')

    # --- Put legends outside on the right ---
    fig.subplots_adjust(right=0.78)
    ax1.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    ax2.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    # ------------------------------------------

    # Overall title    
    status = "NEGATIVE" if getattr(an, "is_negative", False) else "POSITIVE"

    ri = getattr(an, "relative_intensity", None)
    ri_str = "N/A" if (ri is None) else f"{ri:.4f}"

    fig.suptitle(
        f"{Path(an.image_path).name} - Relative Intensity: {ri_str} ({status})",
        fontsize=16, fontweight="bold", y=0.98
    )
    
    

    # WL: Panel 3 -> print the inverted image, the estimated background, and the corrected image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Inverted"); plt.imshow(an.inverted_image, cmap="hot"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Estimated Background"); plt.imshow(an.background_image, cmap="hot"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Corrected (Inverted - Background)"); plt.imshow(an.corrected_image, cmap="hot"); plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")

    return fig


def plot_histogram_and_otsu(an, bins=60, show_overlay=True):
    """
    Plot histograms of inverted and corrected pixel intensities and show Otsu threshold.
    Uses thick connected bars (not thin spiky lines).
    Also displays the corrected image and the resulting binary mask.
    """
    if an.inverted_image is None:
        raise ValueError("Run preprocess() first.")
    if an.corrected_image is None:
        raise ValueError("Run subtract_background() first.")
    if an.binary_mask is None or an.otsu_threshold is None:
        raise ValueError("Run otsu_binarize_corrected() first.")

    inv = an.inverted_image.ravel()
    cor = an.corrected_image.ravel()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # ----------------------------
    # Histogram: inverted (raw)
    # ----------------------------
    axes[0].hist(inv, bins=bins, density=True, histtype='bar',
                 edgecolor=None, linewidth=0, alpha=0.85)

    # Show same numeric threshold for reference
    axes[0].axvline(an.otsu_threshold, linestyle="--", linewidth=2, color="red",
                    label=f"Otsu T (corrected) = {an.otsu_threshold:.1f}")

    axes[0].set_title("Histogram: Inverted (raw)")
    axes[0].set_xlabel("Pixel intensity")
    axes[0].set_ylabel("Density")

    # ----------------------------
    # Histogram: corrected + Otsu
    # ----------------------------
    axes[1].hist(cor, bins=bins, density=True, histtype='bar',
                 edgecolor=None, linewidth=0, alpha=0.85)

    axes[1].axvline(an.otsu_threshold, linestyle="--", linewidth=2,
                    label=f"Otsu T = {an.otsu_threshold:.1f}")

    axes[1].set_title("Histogram: Corrected (bg-subtracted)")
    axes[1].set_xlabel("Pixel intensity")
    axes[1].legend()

    # ----------------------------
    # Show corrected + mask overlay
    # ----------------------------
    axes[2].imshow(an.corrected_image, cmap="hot")
    if show_overlay:
        axes[2].imshow(an.binary_mask, alpha=0.35)
    axes[2].set_title("Corrected + Otsu Overlay")
    axes[2].axis("off")

    # Panel 4: Clean binary visualization
    binary_vis = cv2.bitwise_not(an.binary_mask)

    # Add black border for visibility
    bordered = cv2.copyMakeBorder(binary_vis, 2, 2, 2, 2,
                                  borderType=cv2.BORDER_CONSTANT, value=0)

    axes[3].imshow(bordered, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Binarized (Lines Dark)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    return fig


def plot_rowwise_threshold_debug(an, bins=60, show_overlay=True):
    if an.corrected_image is None:
        raise ValueError("Run subtract_background() first.")
    if not hasattr(an, "_rowwise_debug"):
        raise ValueError("Run rowwise_binarize_corrected() first.")

    dbg = an._rowwise_debug
    row_score = dbg["row_score"]
    baseline = dbg["baseline"]
    T_row = dbg["T_row"]
    sigma = dbg["sigma"]

    # 5 panels now
    fig, axes = plt.subplots(1, 5, figsize=(26, 4))

    # 0) ORIGINAL image
    orig_rgb = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 1) Histogram of row scores (not pixels)
    axes[1].hist(row_score, bins=bins, density=True, histtype="bar", alpha=0.85)
    typical_T = float(np.median(baseline) + dbg["k"] * sigma)
    axes[1].axvline(typical_T, linestyle="--", linewidth=2, color="red",
                    label=f"Typical T ≈ {typical_T:.1f}")
    axes[1].set_title(f"Histogram: Row {dbg['stat']} scores")
    axes[1].set_xlabel("Row score")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # 2) Row score trace + baseline + threshold
    y = np.arange(len(row_score))
    axes[2].plot(row_score, y, linewidth=1.5, label="row_score")
    axes[2].plot(baseline, y, linewidth=1.5, label="baseline (smoothed)")
    axes[2].plot(T_row, y, linewidth=1.5, label="T_row = baseline + k*sigma")
    axes[2].invert_yaxis()
    axes[2].set_title("Row-wise threshold profile")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Row index")
    axes[2].legend()

    # 3) Corrected + mask overlay
    axes[3].imshow(an.corrected_image, cmap="hot")
    if show_overlay and an.binary_mask is not None:
        axes[3].imshow(an.binary_mask, alpha=0.35)
    axes[3].set_title("Corrected + Row-wise Mask Overlay")
    axes[3].axis("off")
    
    # V2:
    # axes[3].imshow(self.corrected_image, cmap="hot")
    # axes[3].contour(self.binary_mask > 0, levels=[0.5], linewidths=2)
    # axes[3].set_title("Corrected + Mask Outline")
    # axes[3].axis("off")
    
    
    # V3:
    # img_vis = cv2.normalize(self.corrected_image, None, 0, 255, cv2.NORM_MINMAX)

    # axes[3].imshow(img_vis, cmap="hot")
    # axes[3].imshow(self.binary_mask, alpha=0.35, cmap="gray", vmin=0, vmax=255)
    # axes[3].set_title("Overlay (normalized background)")
    # axes[3].axis("off")

    # 4) Binary visualization (lines dark)
    if an.binary_mask is None:
        raise ValueError("No binary_mask found. Run rowwise_binarize_corrected().")
    binary_vis = cv2.bitwise_not(an.binary_mask)
    bordered = cv2.copyMakeBorder(binary_vis, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    axes[4].imshow(bordered, cmap="gray", vmin=0, vmax=255)
    axes[4].set_title("Row-wise Binarized (Lines Dark)")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def plot_inverted_vs_corrected(an, cmap="hot"):
    """
    Show side-by-side comparison:
      - Inverted image (after preprocess)
      - Background-subtracted image (after subtract_background)
    """

    if an.inverted_image is None:
        raise ValueError("Run preprocess() first.")
    if an.corrected_image is None:
        raise ValueError("Run subtract_background() first.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Inverted
    axes[0].imshow(an.inverted_image, cmap=cmap)
    axes[0].set_title("Inverted Image\n(Dark lines → Bright)", fontsize=13)
    axes[0].axis("off")

    # Panel 2: Background-subtracted
    axes[1].imshow(an.corrected_image, cmap=cmap)
    axes[1].set_title("After Background Subtraction", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return fig