# lfa/analysis.py
from pathlib import Path
from . import image_processing as ip
import numpy as np
import matplotlib.pyplot as plt


def run_analysis(an, bg="morph", ksize=51, k=5.0, smooth_ksize=51, normalize=False, denoise=False, binarize_mode="rowwise", debug_plots=False):
    """
    Run the full rowwise LFA pipeline.

    Parameters
    ----------
    an       : SimpleLFAAnalyzer instance
    ksize    : background subtraction kernel size (should exceed band width)
    denoise  : apply median blur after background subtraction
    k        : sigma multiplier for row threshold strictness
    debug_plots : show debug figures if True

    Returns
    -------
    dict with keys: status, runs, top_runs, bottom_runs, mid_row, num_bands,
                    and optionally relative_intensity + components if POSITIVE
    """
    print("=" * 60)
    print(f"Analyzing: {Path(an.image_path).name}")
    print("=" * 60)

    # 1) Preprocess
    ip.preprocess(an) # DONE
    
    # Iteratively trims edges that are bright from poor cropping
    ip.auto_crop_remove_bright_edges(an, bright_delta=80, center_frac=0.9, tol_frac=0.25, min_size=40)

    # 2) Background subtraction
    ip.subtract_background(an, method=bg, ksize=ksize, normalize=normalize, denoise=denoise)

    # 3) Rowwise binarization
    if binarize_mode:
        ip.rowwise_binarize_corrected(
            an,
            stat="mean",
            smooth_ksize=smooth_ksize,
            exclude_center_frac=0.20, # how much from the center to exclude
            k=k, # this is how man SD above the background to binarize
            min_run=8,
            expand=2,
            min_peak_sigma=0.0,
        )

    if debug_plots:
        from .visualization import plot_rowwise_threshold_debug
        import matplotlib.pyplot as plt
        fig = plot_rowwise_threshold_debug(an)
        plt.show()
        
    # 4) Classify
    info = ip.classify_two_band_top_bottom(an)

    # 5) Relative intensity
    if info["status"] == "POSITIVE":
        top_run = info["top_runs"][0]
        bot_run = info["bottom_runs"][0]
        ri = ip.compute_relative_intensity_from_runs(an, top_run, bot_run)
        info.update(ri)

    elif info["status"] == "NEGATIVE":
        # Only the control line was detected. Estimate where the test line should be.
        control_run = info["top_runs"][0]
        ri = ip.compute_negative_relative_intensity(an, control_run)
        info.update(ri)

    return info



def calculate_lod_from_negatives(negative_results):
    """
    Calculate LOD from negative/blank LFA relative intensities.

    LOD = mean(blank RI) + 3 * SD(blank RI)

    Uses sample standard deviation (ddof=1).
    """

    values = []

    for result in negative_results:
        if result.get("status") != "NEGATIVE":
            continue

        ri = result.get("relative_intensity")

        if ri is not None:
            values.append(float(ri))

    if len(values) < 2:
        raise ValueError(
            "At least 2 valid negative relative intensities are required."
        )

    values = np.asarray(values, dtype=float)

    mean_blank = float(np.mean(values))

    # Sample SD
    sd_blank = float(np.std(values, ddof=1))

    lod = mean_blank + 3.0 * sd_blank

    return {
        "negative_relative_intensities": values.tolist(),
        "mean_negative_relative_intensity": mean_blank,
        "sd_negative_relative_intensity": sd_blank,
        "lod_relative_intensity": float(lod),
    }