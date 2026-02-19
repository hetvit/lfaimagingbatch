from lfa import SimpleLFAAnalyzer
from lfa.analysis import run_analysis

def main():
    # img_path = 'LFAIMAGES/image3-50fold.jpeg'
    img_path = 'LFAIMAGES/image1neg.jpeg'
    # img_path = 'LFAIMAGES/image9-75fold2.jpeg'
    an = SimpleLFAAnalyzer(img_path)


    # use package analysis function (not an.analyze())
    results = run_analysis(
    an,
    ksize=71,
    k=5.0,
    denoise=False,
    debug_plots=True,
    )

    print("\nReturned results dict:")
    # print(results)
    print_analysis_report(results)

    from lfa.visualization import plot_inverted_vs_corrected
    plot_inverted_vs_corrected(an)
    
    try:
        from lfa.visualization import plot_rowwise_threshold_debug
        # plot_rowwise_threshold_debug(an)  # uses an.corrected_image, an.binary_mask, an._rowwise_debug
    except Exception as e:
        print(f"(Skipping extra debug plot: {e})")

    # Your 2-panel + extra WL figure
    an.visualize(save_path=None)

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