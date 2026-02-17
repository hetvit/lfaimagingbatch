# lfa/analysis.py
from pathlib import Path
from . import image_processing as ip


def run_analysis(
    self,
    bg="",
    k=51,
    normalize=False,
    denoise=False,
    binarize_mode="rowwise",
    debug_plots=True,
):
    an = self
    print("=" * 70)
    print(f"Analyzing: {Path(an.image_path).name}")
    print("=" * 70)
    
    # 1) Preprocess
    ip.preprocess(an)

    # 2) Background subtraction
    ip.subtract_background(an, method=bg, ksize=k, normalize=normalize, denoise=denoise)

    # 3) Binarize
    if binarize_mode == "otsu":
        ip.otsu_binarize_corrected(an, blur_ksize=5)
        if debug_plots:
            from .visualization import plot_histogram_and_otsu
            plot_histogram_and_otsu(an)

        # Your band-count classifier relies on rowwise debug info.
        return {
            "status": "INVALID",
            "reason": "Band-count classification expects binarize_mode='rowwise'.",
        }

    elif binarize_mode == "rowwise":
        ip.rowwise_binarize_corrected(
            an,
            stat="mean",
            smooth_ksize=51,
            k=5.0,
            min_run=5,
            expand=2
        )
        if debug_plots:
            from .visualization import plot_rowwise_threshold_debug
            plot_rowwise_threshold_debug(an)
    else:
        raise ValueError("binarize_mode must be 'otsu' or 'rowwise'")

    # 4) Classify using top+bottom expectation
    info = ip.classify_two_band_top_bottom(an)

    if info["status"] == "POSITIVE":
        top_run = info["top_runs"][0]
        bot_run = info["bottom_runs"][0]
        ri = ip.compute_relative_intensity_from_runs(an, top_run, bot_run)
        info.update(ri)  # adds relative_intensity + components

    # Store on analyzer for convenience
    an.is_negative = (info["status"] != "POSITIVE")

    return info