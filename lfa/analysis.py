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

    # 2) Background subtraction
    ip.subtract_background(an, method=bg, ksize=ksize, normalize=normalize, denoise=denoise)

    # 3) Rowwise binarization
    if binarize_mode:
        ip.rowwise_binarize_corrected(
            an,
            stat="median",
            smooth_ksize=smooth_ksize,
            k=k, # this is how man SD above the background to binarize
            min_run=5,
            expand=2,
        )

    if debug_plots:
        from .visualization import plot_rowwise_threshold_debug
        import matplotlib.pyplot as plt
        fig = plot_rowwise_threshold_debug(an)
        plt.show()
        
    # 4) Classify
    info = ip.classify_two_band_top_bottom(an)

    # 5) Relative intensity (only meaningful for POSITIVE)
    if info["status"] == "POSITIVE":
        top_run = info["top_runs"][0]
        bot_run = info["bottom_runs"][0]
        ri = ip.compute_relative_intensity_from_runs(an, top_run, bot_run)
        info.update(ri)

    return info
