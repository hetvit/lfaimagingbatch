# lfa/analysis.py
from pathlib import Path
from . import image_processing as ip


def run_analysis(an, bg="morph", ksize=71, normalize=False, denoise=False, binarize_mode="rowwise", k=5.0, debug_plots=True):
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
    ip.preprocess(an)

    # 2) Background subtraction
    ip.subtract_background(an, method="morph", ksize=ksize, normalize=normalize, denoise=denoise)

    # 3) Rowwise binarization
    if binarize_mode:
        ip.rowwise_binarize_corrected(
            an,
            stat="median",
            smooth_ksize=51,
            k=k,
            min_run=5,
            expand=2,
        )

    if debug_plots:
        from .visualization import plot_rowwise_threshold_debug
        plot_rowwise_threshold_debug(an)

    # 4) Classify
    info = ip.classify_two_band_top_bottom(an)

    # 5) Relative intensity (only meaningful for POSITIVE)
    if info["status"] == "POSITIVE":
        top_run = info["top_runs"][0]
        bot_run = info["bottom_runs"][0]
        ri = ip.compute_relative_intensity_from_runs(an, top_run, bot_run)
        info.update(ri)

    return info
