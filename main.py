from lfa import SimpleLFAAnalyzer
from lfa.analysis import run_analysis
from lfa.utils import show_preprocessing_steps, visualize_background_subtraction, visualize_rowwise_thresholding
from __future__ import annotations
from pathlib import Path

def analyze_image(img_path: str):
    """
    Single source-of-truth pipeline entry point for both CLI (main.py)
    and FastAPI (server.py).

    Returns:
        (results_dict, analyzer_instance)
    """
    an = SimpleLFAAnalyzer(img_path)

    results = run_analysis(
        an,
        bg="morph",
        ksize=51,
        k=1.5,
        smooth_ksize=91,
        normalize=False,
        denoise=False,
        binarize_mode="rowwise",
        debug_plots=False,  # IMPORTANT: server is headless; don't generate show() plots
    )

    return results, an
def main():
    img_path = "LFAIMAGES/02-24 device_flashmoved/democrop.PNG"

    results, an = analyze_image(img_path, debug_plots=True)  # CLI only

    print("\nReturned results dict:")
    print(results)
    print_analysis_report(results)

    from lfa.visualization import plot_inverted_vs_corrected
    plot_inverted_vs_corrected(an)
    
    # try:
    #     from lfa.visualization import plot_rowwise_threshold_debug
    #     plot_rowwise_threshold_debug(an)  # uses an.corrected_image, an.binary_mask, an._rowwise_debug
    # except Exception as e:
    #     print(f"(Skipping extra debug plot: {e})")

    # # Your 2-panel + extra WL figure
    # an.visualize(save_path=None)

    
    ###########################################
    # DEBUG/IMAGE VISUALIZATION
    ###########################################
    
    # Preprocess Visual
    # show_preprocessing_steps(img_path)
    
    #BG Subtraction Visual
    # visualize_background_subtraction(img_path, method="morph", ksize=51, normalize=False, denoise=False, colormode="gray")

    # visualize_background_subtraction(img_path, method="morph", ksize=51, normalize=False, denoise=False, colormode="hot")
    
    
    #Rowwise Thresholding Visual
    # visualize_rowwise_thresholding(
    #     img_path,
    #     stat="mean",
    #     smooth_ksize=81,
    #     k=3.0,
    #     exclude_center_frac=0.0,
    #     min_run=3,
    #     expand=2,
    #     keep_top_bands=2,
    #     band_score_mode="auc",
    #     colormode="hot"
    # )

    return

def print_analysis_report(results):
    """
    Nicely formatted CLI-style report for LFA results dict.
    """

    print("\n" + "=" * 60)
    print("LFA ANALYSIS RESULTS")
    print("=" * 60)

    status = results.get("status", "UNKNOWN")
    num_bands = results.get("num_bands", "N/A")
    runs = results.get("runs", [])
    top_runs = results.get("top_runs", [])
    bottom_runs = results.get("bottom_runs", [])
    mid_row = results.get("mid_row", None)
    rel_int = results.get("relative_intensity", None)

    # Status line with symbol
    if status == "POSITIVE":
        symbol = "🟢"
    elif status == "NEGATIVE":
        symbol = "🔴"
    elif status == "INVALID":
        symbol = "⚠️"
    else:
        symbol = "?"

    print(f"Result:              {symbol} {status}")
    print(f"Number of Bands:     {num_bands}")

    if mid_row is not None:
        print(f"Image Midpoint Row:  {mid_row}")

    if runs:
        print(f"Detected Band Runs:  {runs}")

    if top_runs or bottom_runs:
        print(f"Top Half Runs:       {top_runs}")
        print(f"Bottom Half Runs:    {bottom_runs}")

    if rel_int is not None:
        print(f"Relative Intensity:  {rel_int:.4f}")
    else:
        print("Relative Intensity:  N/A")

    print("=" * 60 + "\n")




if __name__ == "__main__":
    main()