
from lfa import SimpleLFAAnalyzer
from lfa.analysis import run_analysis
from lfa.image_processing import preprocess, subtract_background

import numpy as np
import cv2
from . import image_processing as ip
import matplotlib.pyplot as plt

def show_preprocessing_steps(image_path):
    """
    Load an LFA image, run preprocess(), and show:
    [original] [grayscale] [inverted]
    """
    # 1. Make analyzer and run your existing preprocess
    an = SimpleLFAAnalyzer(image_path)
    preprocess(an)   # fills an.gray_image and an.inverted_image

    # 2. Grab images
    orig_bgr = an.original_image
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)  # for correct display
    gray     = an.gray_image
    inv      = an.inverted_image

    # 3. Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(inv, cmap="gray")
    axes[2].set_title("Inverted")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


import cv2
import matplotlib.pyplot as plt
from lfa import SimpleLFAAnalyzer
from lfa.image_processing import preprocess
from lfa.image_processing import subtract_background

def visualize_background_subtraction(image_path,
                                     method="morph",
                                     ksize=51,
                                     normalize=False,
                                     denoise=False,
                                     colormode="gray"): #gray or hot
    """
    Full visualization helper:
    Loads image → preprocess() → subtract_background()
    Shows:
        1) inverted image
        2) background estimate
        3) background-subtracted image
    """

    # 1) Create analyzer + run preprocess
    an = SimpleLFAAnalyzer(image_path)
    preprocess(an)     # populates an.gray_image and an.inverted_image

    if an.inverted_image is None:
        raise ValueError("preprocess() failed — inverted image is None.")

    # 2) Run background subtraction (fills an.background_image + corrected_image)
    corrected = subtract_background(
        an,
        method=method,
        ksize=ksize,
        normalize=normalize,
        denoise=denoise,
    )

    # Extract images for plotting
    inverted   = an.inverted_image
    background = an.background_image
    corrected  = an.corrected_image

    # 3) Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].imshow(inverted, cmap=colormode)
    axes[0].set_title("Inverted Image")
    axes[0].axis("off")

    axes[1].imshow(background, cmap=colormode)
    axes[1].set_title(f"Background Estimate ({method})")
    axes[1].axis("off")

    axes[2].imshow(corrected, cmap=colormode)
    axes[2].set_title("Background-Subtracted Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()



from lfa import SimpleLFAAnalyzer
from lfa.image_processing import (
    preprocess,
    subtract_background,
    detect_band_rows,
    rowwise_binarize_corrected,
)

def visualize_rowwise_thresholding(
    image_path,
    stat="mean",
    smooth_ksize=51,
    k=4.0,
    exclude_center_frac=0.0,
    min_run=3,
    expand=2,
    keep_top_bands=2,
    band_score_mode="auc",
    colormode="gray"
):
    """
    Visualize the full row-wise thresholding pipeline in ONE figure:

        (1) Corrected image
        (2) Row score vs baseline vs T_row
        (3) Residuals
        (4) Raw + cleaned hits
        (5) Final row-wise mask
    """

    # ---------------------------
    # 1) Load + preprocess + subtract background
    # ---------------------------
    an = SimpleLFAAnalyzer(image_path)
    preprocess(an)
    subtract_background(an, method="morph", ksize=51, denoise=False, normalize=False)

    corrected = an.corrected_image.astype(np.float32)
    H, W = corrected.shape

    # ---------------------------
    # 2) Compute row-wise metrics
    # ---------------------------
    row_score, baseline, T_row, row_hits, peak_row, sigma = detect_band_rows(
        corrected,
        stat=stat,
        smooth_ksize=smooth_ksize,
        exclude_center_frac=exclude_center_frac,
        k=k,
    )

    resid = row_score - baseline

    # ---------------------------
    # 3) Build final mask (and cleaned hits)
    # ---------------------------
    _, final_mask = rowwise_binarize_corrected(
        an,
        stat=stat,
        smooth_ksize=smooth_ksize,
        exclude_center_frac=exclude_center_frac,
        k=k,
        min_run=min_run,
        expand=expand,
        invert_mask=False,
        keep_top_bands=keep_top_bands,
        band_score_mode=band_score_mode,
    )
    hits_clean = an._rowwise_debug["row_hits_clean"]

    # ---------------------------
    # 4) Plot 2×3 layout (keeps 4 original plots + final mask)
    # ---------------------------
    fig = plt.figure(figsize=(20, 12))

    # (A) corrected image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(corrected, cmap=colormode)
    ax1.set_title("Corrected Image")
    ax1.axis("off")

    # (B) row_score, baseline, T_row
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(row_score, label="row_score", linewidth=2)
    ax2.plot(baseline, label="baseline", linewidth=2)
    ax2.plot(T_row, label=f"T_row (k={k}, σ={sigma:.3f})", linestyle="--")
    ax2.axvline(peak_row, color="red", linestyle=":", label="Peak row")
    ax2.set_title("Row Score / Baseline / Threshold")
    ax2.legend()

    # (C) residuals
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(resid, label="residuals", linewidth=2)
    ax3.axhline(0, color="gray", linestyle="--")
    ax3.set_title("Residuals")
    ax3.legend()

    # (D) raw + cleaned hits
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(row_hits.astype(int), label="raw hits")
    ax4.plot(hits_clean.astype(int), label="clean hits", linewidth=2)
    ax4.set_title("Hits (Raw & Cleaned)")
    ax4.legend()

    # (E) final mask
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(final_mask, cmap="gray")
    ax5.set_title("Final Row-wise Mask")
    ax5.axis("off")

    # Third column bottom-right is left empty
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "row_score": row_score,
        "baseline": baseline,
        "T_row": T_row,
        "resid": resid,
        "row_hits_raw": row_hits,
        "row_hits_clean": hits_clean,
        "peak_row": peak_row,
        "sigma": sigma,
        "final_mask": final_mask,
    }