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



from pathlib import Path





def plot_lfa_final_panels(an, save_dir="lfa_strip_panels", uncropped_image=None):
    """
    Debug figure combining rowwise threshold info with image stages
    for a single LFA strip, arranged in TWO rows.

    Order (with uncropped):
      Row 1: [0] uncropped, [1] cropped, [2] preprocessed, [3] background, [4] BG-subtracted
      Row 2: [5] rowwise trace, [6] binarized mask, [7] BG+mask overlay, [8] original + ROIs

    Without uncropped, the first row is 4 panels and the second row is 4 panels.
    """

    # --- Sanity checks for main pipeline ---
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

    # --- Unpack rowwise debug for the row trace panel ---
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

    # --- Output path setup ---
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(an.image_path).stem

    # --- Decide number of panels and grid shape ---
    has_uncropped = uncropped_image is not None
    if has_uncropped:
        n_panels = 9   # uncropped + 8 existing
        n_cols   = 5   # 5 on top, 4 on bottom (last axis unused)
        base_idx = 1   # cropped/raw starts at panel index 1
    else:
        n_panels = 8   # original layout
        n_cols   = 4   # 4 on top, 4 on bottom
        base_idx = 0   # cropped/raw starts at panel index 0

    n_rows = 2

    fig_width  = 3 * n_cols
    fig_height = 5 * n_rows
    # fig, axes_grid = plt.subplots(
    #     n_rows, n_cols,
    #     figsize=(fig_width, fig_height),
    #     constrained_layout=True,
    # )

    # # Make sure axes_grid is 2D array
    # axes_grid = np.atleast_2d(axes_grid)
    
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = fig.add_gridspec(
        2, n_cols,
        height_ratios=[1, 1],   # equal row heights
        hspace=0.25 # increase vertical spacing between rows
    )
    axes_grid = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(2)])

    # Helper to map panel index -> axis (row-major)
    def get_axis_for_panel(k: int):
        """Return the axis corresponding to logical panel index k."""
        if k < n_cols:
            r, c = 0, k
        else:
            r, c = 1, k - n_cols
        return axes_grid[r, c]

    # Build linear list of axes in panel order, to reuse your existing logic
    axes = [get_axis_for_panel(k) for k in range(n_panels)]
    
    # if has_uncropped:
    #     # 1-based: (2,3),(3,4),(4,5),(7,8),(8,9)
    #     tight_pairs = [(1, 2), (2, 3), (3, 4), (6, 7), (7, 8)]
    # else:
    #     # Without uncropped, everything shifts left by 1:
    #     # visual 1-based: (1,2),(2,3),(3,4),(5,6),(6,7)
    #     tight_pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]

    
    # # === Reduce spacing between specific horizontal panel pairs ===
    # TIGHT_FRACTION = 0.4  # 0.0 = no change, 1.0 = panels touch
    # fig.canvas.draw()     # make sure positions are up to date

    # for left_idx, right_idx in tight_pairs:
    #     if left_idx >= len(axes) or right_idx >= len(axes):
    #         continue  # skip if panel doesn't exist (e.g., no uncropped mode)

    #     ax_left  = axes[left_idx]
    #     ax_right = axes[right_idx]

    #     # Current positions
    #     bL = ax_left.get_position()
    #     bR = ax_right.get_position()

    #     # Compute new closer positions
    #     # Move right-axis leftwards slightly
    #     new_left_xR = bR.x0 - TIGHT_FRACTION
    #     shift = bR.x0 - new_left_xR

    #     # Apply shift to right panel
    #     ax_right.set_position([
    #         bR.x0 - shift,  # left
    #         bR.y0,          # bottom
    #         bR.width,       # width
    #         bR.height       # height
    #     ])

    #     # Apply a symmetric shift to the left panel (move it slightly right)
    #     ax_left.set_position([
    #         bL.x0 + shift,
    #         bL.y0,
    #         bL.width,
    #         bL.height
    #     ])
    

    # If we have an extra unused axis (e.g. 9 panels in a 2x5 grid), turn it off
    if has_uncropped and n_cols * n_rows > n_panels:
        unused_ax = axes_grid[1, n_cols - 1]
        unused_ax.axis("off")

    # --- 0) Optional uncropped panel ---
    if has_uncropped:
        uncropped_rgb = cv2.cvtColor(uncropped_image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(uncropped_rgb)
        axes[0].set_title("Raw Image\n(Uncropped)", fontsize=10)
        axes[0].axis("off")

    # Convenience aliases for indices (into axes list)
    i_cropped   = base_idx + 0
    i_preproc   = base_idx + 1
    i_bg        = base_idx + 2
    i_corr      = base_idx + 3
    i_rowtrace  = base_idx + 4
    i_binmask   = base_idx + 5
    i_corr_mask = base_idx + 6
    i_final     = base_idx + 7

    # 1) Cropped raw image (RGB)
    axes[i_cropped].imshow(orig_rgb)
    axes[i_cropped].set_title("Auto-cropped Image", fontsize=10)
    axes[i_cropped].axis("off")

    # 2) Inverted image (preprocessed)
    axes[i_preproc].imshow(inv, cmap="hot")
    axes[i_preproc].set_title("Preprocessed", fontsize=10)
    axes[i_preproc].axis("off")

    # 3) Background image that was subtracted
    axes[i_bg].imshow(bg, cmap="hot")
    axes[i_bg].set_title("Background\n(Morphological Opening)", fontsize=10)
    axes[i_bg].axis("off")

    # 4) Corrected image (no mask)
    axes[i_corr].imshow(corr, cmap="hot")
    axes[i_corr].set_title("Background Subtracted\n(BG) Image", fontsize=10)
    axes[i_corr].axis("off")

    # 5) Row score trace (row-wise threshold profile)
    y = np.arange(len(row_score))
    axes[i_rowtrace].plot(row_score, y, linewidth=1.5, label="row_score")
    axes[i_rowtrace].plot(baseline,  y, linestyle="--", linewidth=2.0, label="baseline")
    axes[i_rowtrace].plot(T_row,     y, linestyle=":",  linewidth=2.0, label="threshold")

    axes[i_rowtrace].invert_yaxis()
    axes[i_rowtrace].set_title("Row-wise Threshold Profile", fontsize=10)
    axes[i_rowtrace].set_xlabel("Average Row Intensity Value")
    axes[i_rowtrace].set_ylabel("Row index")
    axes[i_rowtrace].legend(fontsize=8)

    # 6) Binary mask (lines dark)
    binary_vis = cv2.bitwise_not(mask)
    bordered   = cv2.copyMakeBorder(
        binary_vis, 2, 2, 2, 2,
        cv2.BORDER_CONSTANT, value=0
    )
    axes[i_binmask].imshow(bordered, cmap="gray", vmin=0, vmax=255)
    axes[i_binmask].set_title("Binarized Mask", fontsize=10)
    axes[i_binmask].axis("off")

    # 7) Corrected + mask overlay (colored)
    axes[i_corr_mask].imshow(corr, cmap="hot")
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    color = np.array([0.0, 1.0, 0.0])  # green
    m = (mask > 0)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = 0.35 * m
    axes[i_corr_mask].imshow(overlay)
    axes[i_corr_mask].set_title("BG Image\n+ Mask Overlay", fontsize=10)
    axes[i_corr_mask].axis("off")

    # 8) Original + mask overlay + sampling regions
    axes[i_final].imshow(orig_rgb)

    # full (rowwise) mask → GREEN overlay
    overlay_full = np.zeros((*mask.shape, 4), dtype=np.float32)
    full_hits = (mask > 0).astype(np.float32)
    overlay_full[..., :3] = np.array([0.0, 1.0, 0.0])
    overlay_full[..., 3] = 0.20 * full_hits
    axes[i_final].imshow(overlay_full)

    from lfa.image_processing import _runs_from_hits, make_band_sampling_mask
    hits = an._rowwise_debug["row_hits_clean"].astype(bool)
    runs = _runs_from_hits(hits)

    row_radius = 10
    edge_margin_frac = 0.01

    for (s, e) in runs:
        sample_mask = make_band_sampling_mask(
            an,
            (s, e),
            edge_margin_frac=edge_margin_frac,
            row_radius=row_radius,
        )
        sample_hits = (sample_mask > 0).astype(np.float32)

        sample_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        sample_overlay[..., :3] = np.array([1.0, 0.0, 0.0])  # red
        sample_overlay[..., 3] = 0.40 * sample_hits
        axes[i_final].imshow(sample_overlay)

    axes[i_final].set_title("Raw Image + Mask\n+ Intensity Window", fontsize=10)
    axes[i_final].axis("off")

    # --- Arrows between panels (now 2 rows) ---
    fig.canvas.draw()

    ARROW_LEN_H = 0.04  # horizontal arrow length (figure fraction)
    ARROW_LEN_V = 0.05  # vertical arrow length (figure fraction)

    for k in range(n_panels - 1):
        # --- skip arrow between panel 5 and 6 ---
        if k == 4:      # pair (5 -> 6)
            continue
        ax_start = axes[k]
        ax_end   = axes[k + 1]

        bbox_start = ax_start.get_position()
        bbox_end   = ax_end.get_position()

        # Centers in figure coords
        x_mid_start = 0.5 * (bbox_start.x0 + bbox_start.x1)
        y_mid_start = 0.5 * (bbox_start.y0 + bbox_start.y1)
        x_mid_end   = 0.5 * (bbox_end.x0 + bbox_end.x1)
        y_mid_end   = 0.5 * (bbox_end.y0 + bbox_end.y1)

        # MUCH stricter row check to avoid false vertical arrows
        same_row = abs(y_mid_start - y_mid_end) < 0.01

        if same_row:
            # Horizontal arrow ONLY
            gap_left   = bbox_start.x1
            gap_right  = bbox_end.x0
            gap_center = 0.5 * (gap_left + gap_right)
            y_mid      = y_mid_start

            half = ARROW_LEN_H / 2.0
            x0 = gap_center - half
            x1 = gap_center + half
            y0 = y1 = y_mid

        else:
            # Vertical arrow ONLY (true row break)
            x_center = 0.5 * (x_mid_start + x_mid_end)
            half = ARROW_LEN_V / 2.0
            y_center = 0.5 * (bbox_start.y0 + bbox_end.y1)
            y0 = y_center + half
            y1 = y_center - half
            x0 = x1 = x_center
            
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            transform=fig.transFigure,
            arrowstyle="->",
            lw=2,
            color="black",
            mutation_scale=13,
        )
        fig.add_artist(arrow)

    # --- Save single combined figure ---
    out_path = save_dir / f"{stem}_lfa_FINAL_panels.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"[DEBUG] Saved LFA debug panels to {out_path}")
    return out_path





# THIS VERSION DOES IT ALL IN 1 ROW
# def plot_lfa_final_panels(an, save_dir="lfa_strip_panels", uncropped_image=None):
#     """
#     Debug figure combining rowwise threshold info with image stages
#     for a single LFA strip.

#     If `uncropped_image` (BGR) is provided, an extra panel is added to the
#     left showing the raw uncropped image, with arrows between all panels.
#     """

#     # --- Sanity checks for main pipeline ---
#     if an.original_image is None:
#         raise ValueError("original_image is None; did SimpleLFAAnalyzer load correctly?")
#     if an.inverted_image is None:
#         raise ValueError("inverted_image is None; run preprocess() first.")
#     if an.corrected_image is None or an.background_image is None:
#         raise ValueError("corrected_image or background_image is None; run subtract_background() first.")
#     if an.binary_mask is None:
#         raise ValueError("binary_mask is None; run rowwise_binarize_corrected() first.")
#     if not hasattr(an, "_rowwise_debug"):
#         raise ValueError("Missing _rowwise_debug; run rowwise_binarize_corrected() first.")

#     # --- Unpack rowwise debug for the row trace panel ---
#     dbg       = an._rowwise_debug
#     row_score = dbg["row_score"]
#     baseline  = dbg["baseline"]
#     T_row     = dbg["T_row"]

#     orig_bgr = an.original_image
#     orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
#     inv      = an.inverted_image
#     corr     = an.corrected_image
#     mask     = an.binary_mask
#     bg       = an.background_image

#     # --- Output path setup ---
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     stem = Path(an.image_path).stem

#     # --- Decide number of panels ---
#     has_uncropped = uncropped_image is not None
#     if has_uncropped:
#         n_panels = 9   # uncropped + 8 existing
#         base_idx = 1   # cropped/raw starts at axes[1]
#     else:
#         n_panels = 8   # original layout
#         base_idx = 0   # cropped/raw starts at axes[0]

#     # Scale width with number of panels
#     fig_width = 3 * n_panels
#     fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4), constrained_layout=True)

#     # --- 0) Optional uncropped panel ---
#     if has_uncropped:
#         # uncropped_image is assumed BGR (cv2-style)
#         uncropped_rgb = cv2.cvtColor(uncropped_image, cv2.COLOR_BGR2RGB)
#         axes[0].imshow(uncropped_rgb)
#         axes[0].set_title("Raw Image\n(Uncropped)", fontsize=10)
#         axes[0].axis("off")

#     # Convenience aliases for indices
#     i_cropped   = base_idx + 0
#     i_preproc   = base_idx + 1
#     i_bg        = base_idx + 2
#     i_corr      = base_idx + 3
#     i_rowtrace  = base_idx + 4
#     i_binmask   = base_idx + 5
#     i_corr_mask = base_idx + 6
#     i_final     = base_idx + 7

#     # 1) Cropped raw image (RGB)
#     axes[i_cropped].imshow(orig_rgb)
#     axes[i_cropped].set_title("Cropped Raw Image", fontsize=10)
#     axes[i_cropped].axis("off")

#     # 2) Inverted image (preprocessed)
#     axes[i_preproc].imshow(inv, cmap="hot")
#     axes[i_preproc].set_title("Preprocessed", fontsize=10)
#     axes[i_preproc].axis("off")

#     # 3) Background image that was subtracted
#     axes[i_bg].imshow(bg, cmap="hot")
#     axes[i_bg].set_title("Background\n(Morphological Opening)", fontsize=10)
#     axes[i_bg].axis("off")

#     # 4) Corrected image (no mask)
#     axes[i_corr].imshow(corr, cmap="hot")
#     axes[i_corr].set_title("Background Subtracted\n(BG) Image", fontsize=10)
#     axes[i_corr].axis("off")

#     # 5) Row score trace (row-wise threshold profile)
#     y = np.arange(len(row_score))
#     axes[i_rowtrace].plot(row_score, y, linewidth=1.5, label="row_score")
#     axes[i_rowtrace].plot(baseline,  y, linestyle="--", linewidth=2.0, label="baseline")
#     axes[i_rowtrace].plot(T_row,     y, linestyle=":",  linewidth=2.0, label="threshold")

#     axes[i_rowtrace].invert_yaxis()
#     axes[i_rowtrace].set_title("Row-wise Threshold Profile", fontsize=10)
#     axes[i_rowtrace].set_xlabel("Average Row Intensity Value")
#     axes[i_rowtrace].set_ylabel("Row index")
#     axes[i_rowtrace].legend(fontsize=8)

#     # 6) Binary mask (lines dark)
#     binary_vis = cv2.bitwise_not(mask)
#     bordered   = cv2.copyMakeBorder(
#         binary_vis, 2, 2, 2, 2,
#         cv2.BORDER_CONSTANT, value=0
#     )
#     axes[i_binmask].imshow(bordered, cmap="gray", vmin=0, vmax=255)
#     axes[i_binmask].set_title("Binarized Mask", fontsize=10)
#     axes[i_binmask].axis("off")

#     # 7) Corrected + mask overlay (colored)
#     axes[i_corr_mask].imshow(corr, cmap="hot")
#     overlay = np.zeros((*mask.shape, 4), dtype=float)
#     color = np.array([0.0, 1.0, 0.0])  # green
#     m = (mask > 0)
#     overlay[..., 0] = color[0]
#     overlay[..., 1] = color[1]
#     overlay[..., 2] = color[2]
#     overlay[..., 3] = 0.35 * m
#     axes[i_corr_mask].imshow(overlay)
#     axes[i_corr_mask].set_title("BG Image\n+ Binarized Mask Overlay", fontsize=10)
#     axes[i_corr_mask].axis("off")

#     # 8) Original + mask overlay + sampling regions
#     axes[i_final].imshow(orig_rgb)

#     # full (rowwise) mask → GREEN overlay
#     overlay_full = np.zeros((*mask.shape, 4), dtype=np.float32)
#     full_hits = (mask > 0).astype(np.float32)
#     overlay_full[..., :3] = np.array([0.0, 1.0, 0.0])
#     overlay_full[..., 3] = 0.20 * full_hits
#     axes[i_final].imshow(overlay_full)

#     from lfa.image_processing import _runs_from_hits, make_band_sampling_mask
#     hits = an._rowwise_debug["row_hits_clean"].astype(bool)
#     runs = _runs_from_hits(hits)

#     row_radius = 10
#     edge_margin_frac = 0.01

#     for (s, e) in runs:
#         sample_mask = make_band_sampling_mask(
#             an,
#             (s, e),
#             edge_margin_frac=edge_margin_frac,
#             row_radius=row_radius,
#         )
#         sample_hits = (sample_mask > 0).astype(np.float32)

#         sample_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
#         sample_overlay[..., :3] = np.array([1.0, 0.0, 0.0])  # red
#         sample_overlay[..., 3] = 0.40 * sample_hits
#         axes[i_final].imshow(sample_overlay)

#     axes[i_final].set_title("Original\n+ Mask + ROIs", fontsize=10)
#     axes[i_final].axis("off")

#     # --- Arrows between panels (including uncropped→cropped if present) ---
#     fig.canvas.draw()
    
#     ARROW_LEN = 0.02   # fixed arrow length (fraction of figure width) — tweak as desired

#     for i in range(n_panels - 1):  # 0→1, ..., (n_panels-2)→(n_panels-1)
#         ax_start = axes[i]
#         ax_end   = axes[i + 1]

#         bbox_start = ax_start.get_position()
#         bbox_end   = ax_end.get_position()

#         # vertical position
#         y_mid = 0.5 * (bbox_start.y0 + bbox_start.y1)

#         # horizontal gap
#         gap_left  = bbox_start.x1
#         gap_right = bbox_end.x0
#         gap_width = gap_right - gap_left
#         gap_center = 0.5 * (gap_left + gap_right)
        
        
#         # Fixed-length arrow centered in the gap
#         half = ARROW_LEN / 2
#         x0 = gap_center - half
#         x1 = gap_center + half

#         from matplotlib.patches import FancyArrowPatch
#         arrow = FancyArrowPatch(
#             (x0, y_mid), (x1, y_mid),
#             transform=fig.transFigure,
#             arrowstyle="->",
#             lw=2,
#             color="black",
#             mutation_scale=13,
#         )
#         fig.add_artist(arrow)


#     # --- Save single combined figure ---
#     out_path = save_dir / f"{stem}_lfa_FINAL_panels.png"
#     fig.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     print(f"[DEBUG] Saved LFA debug panels to {out_path}")
#     return out_path




# Used in final image generation
# def plot_lfa_final_panels(an, save_dir="lfa_strip_panels"):
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
    fig, axes = plt.subplots(1, 8, figsize=(20, 4), constrained_layout=True)

    # 0) Original image (RGB) -- matches plot_rowwise_threshold_debug panel 0
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Cropped Raw Image")
    axes[0].axis("off")

    
    # 1) Inverted image (preprocessed)
    axes[1].imshow(inv, cmap="hot")
    axes[1].set_title("Preprocessed", fontsize=10)
    axes[1].axis("off")

    # 2) Background image that was subtracted
    axes[2].imshow(bg, cmap="hot")
    axes[2].set_title("Background\n(Morphological Opening)", fontsize=10)
    axes[2].axis("off")

    # 3) Corrected image (no mask)
    axes[3].imshow(corr, cmap="hot")
    axes[3].set_title("Background Subtracted (BG) Image\n(Preprocessed - BG)", fontsize=10)
    axes[3].axis("off")
    
    # 4) Row score trace (row-wise threshold profile) -- matches panel 2
    y = np.arange(len(row_score))
    axes[4].plot(row_score, y, linewidth=1.5, label="row_score")           # solid
    axes[4].plot(baseline,  y, linestyle="--", linewidth=2.0, label="baseline")   # dashed, thicker
    axes[4].plot(T_row,     y, linestyle=":",  linewidth=2.0, label="threshold")  # dotted

    axes[4].invert_yaxis()
    axes[4].set_title("Row-wise\nthreshold profile")
    axes[4].set_xlabel("Average Row Intensity Value")
    axes[4].set_ylabel("Row index")
    axes[4].legend(fontsize=8)

    
    # 5) Binary mask (lines dark) -- matches plot_rowwise_threshold_debug panel 4
    binary_vis = cv2.bitwise_not(mask)  # lines dark
    bordered   = cv2.copyMakeBorder(
        binary_vis, 2, 2, 2, 2,
        cv2.BORDER_CONSTANT, value=0
    )
    axes[5].imshow(bordered, cmap="gray", vmin=0, vmax=255)
    axes[5].set_title("Binarized Mask", fontsize=10)
    axes[5].axis("off")

    # 6) Corrected + mask overlay (colored)
    axes[6].imshow(corr, cmap="hot")
    # Build an RGBA overlay the same size as the mask
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    # Choose a color: green here; use [1, 0, 0] for red
    color = np.array([0.0, 1.0, 0.0])  # RGB in 0–1
    # Boolean mask of "on" pixels
    m = (mask > 0)
    # Assign color only where mask == 1
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = 0.35 * m  # alpha only where mask is on
    axes[6].imshow(overlay)
    axes[6].set_title("BG Image\n+ Binarized Mask Overlay", fontsize=10)
    axes[6].axis("off")


    # 7) Original + mask overlay + sampling regions
    axes[7].imshow(orig_rgb)

    # === Full (rowwise) mask → GREEN overlay ===
    overlay_full = np.zeros((*mask.shape, 4), dtype=np.float32)
    full_hits = (mask > 0).astype(np.float32)
    overlay_full[..., :3] = np.array([0.0, 1.0, 0.0])  # GREEN
    overlay_full[..., 3] = 0.20 * full_hits            # light transparency
    axes[7].imshow(overlay_full)

    # === Sampling regions → RED overlay ===
    from lfa.image_processing import _runs_from_hits, make_band_sampling_mask

    hits = an._rowwise_debug["row_hits_clean"].astype(bool)
    runs = _runs_from_hits(hits)

    row_radius = 10
    edge_margin_frac = 0.01

    for (s, e) in runs:
        sample_mask = make_band_sampling_mask(
            an,
            (s, e),
            edge_margin_frac=edge_margin_frac,
            row_radius=row_radius,
        )

        sample_hits = (sample_mask > 0).astype(np.float32)

        sample_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        sample_overlay[..., :3] = np.array([1.0, 0.0, 0.0])  # RED
        sample_overlay[..., 3] = 0.40 * sample_hits # alpha is how transparent it is
        axes[7].imshow(sample_overlay)
        
    axes[7].set_title("Original\n+ Mask + ROIs", fontsize=10)
    axes[7].axis("off")
    
    
    from matplotlib.patches import FancyArrowPatch

    # === Add arrows between panels, in the gaps ===
    fig.canvas.draw()  # make sure positions are computed

    for i in range(7):  # 0→1, ..., 6→7
        ax_start = axes[i]
        ax_end   = axes[i + 1]

        bbox_start = ax_start.get_position()
        bbox_end   = ax_end.get_position()

        # Vertical position: midway between panels (they should be aligned anyway)
        y = 0.5 * (bbox_start.y0 + bbox_start.y1)

        # Horizontal gap between the two panels
        gap_left  = bbox_start.x1
        gap_right = bbox_end.x0
        gap_width = gap_right - gap_left

        # Put the arrow fully inside that gap (20% in from each side)
        x0 = gap_left  + 0.2 * gap_width   # tail
        x1 = gap_right - 0.2 * gap_width   # head

        arrow = FancyArrowPatch(
            (x0, y), (x1, y),
            transform=fig.transFigure,   # figure fraction coords
            arrowstyle="->",
            lw=2,
            color="black",
            mutation_scale=15,
        )
        fig.add_artist(arrow)

    # Save one PNG per strip
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(an.image_path).stem
    out_path = save_dir / f"{stem}_lfa_FINAL_panels.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[DEBUG] Saved LFA debug panels to {out_path}")
    return out_path