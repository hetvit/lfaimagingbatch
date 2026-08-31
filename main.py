from lfa import SimpleLFAAnalyzer
from lfa.analysis import run_analysis
from lfa.utils import show_preprocessing_steps, visualize_background_subtraction, visualize_rowwise_thresholding
from lfa.mask_utils import (
    mean_intensity_under_saved_mask,
    show_saved_mask,
    show_mask_overlay_on_image,
    compute_relative_intensity_with_saved_test_mask,
    visualize_saved_test_and_control_masks
)
from lfa.visualization import *

def main():
    # img_path = 'LFAIMAGES/standard 2-20/no atps 1 REAL.JPG'
    img_path = 'LFAIMAGES/standard 2-20/3e5 + atps 1.JPG'
    # img_path = 'LFAIMAGES/standard 2-20/3e5 + atps 2.JPG'
    # img_path = 'LFAIMAGES/standard 2-20/3e5 + no atps 1.JPG'
    # img_path = 'LFAIMAGES/standard 2-20/3e5 + no atps 2.JPG'
    # img_path = 'LFAIMAGES/75_fold_manual_1.jpeg'
    # # img_path = 'LFAIMAGES/image3-50fold.jpeg'
    # # img_path = 'LFAIMAGES/SP-2-18/3e6_crop.jpg'
    # img_path = 'LFAIMAGES/image9-75fold2.jpeg'
    # img_path = 'LFAIMAGES/2-19/1e6_crop_auto.JPG'
    # img_path = 'LFAIMAGES/original/50_fold_manual_1.jpeg'
    # img_path = 'LFAIMAGES/original/image5-50fold2_COPY.jpeg'
    # img_path = 'LFAIMAGES/SP 2-19/SP_1e7.png'
    # img_path = 'LFAIMAGES/SP 2-19/SP_neg.png'
    # img_path = 'LFAIMAGES/2-27/3e5 no atps 3_cropped.JPG'
    # img_path = 'LFAIMAGES/controls/IMG_8390 - blank 1_cropped.JPEG'
    an = SimpleLFAAnalyzer(img_path)

    # use package analysis function
    results = run_analysis(
        an,
        bg='morph',
        ksize=51,
        k=1.5, # this is how many SD above is a band
        smooth_ksize=91, # 1d median filter smoothing
        normalize=False,
        denoise=False,
        binarize_mode="rowwise",
        # debug_plots=True,
    )
    print("\nReturned results dict:")
    print(results)
    # print_analysis_report(results)
    
    
    
    
    # from lfa.mask_utils import save_test_band_mask
    # if results["status"] == "POSITIVE":
    #     save_test_band_mask(an, results, out_prefix="saved_test_band")
    
    
    ###########################################
    # FINAL FIGURE GENERATION
    ###########################################
    
    # # # --- NEW: generate the big 3-panel debug image for this strip ---
    import cv2
    from lfa.visualization import plot_lfa_final_panels, plot_lfa_debug_panels
    uncropped = cv2.imread("./1e6_precropped.jpg", cv2.IMREAD_COLOR)
    # for debug:
    # plot_lfa_debug_panels(an, save_dir="lfa_strip_panels")
    # for final panel:
    # plot_lfa_final_panels(an, save_dir="lfa_strip_panels", uncropped_image=uncropped)

    # from lfa.visualization import plot_inverted_vs_corrected
    # plot_inverted_vs_corrected(an)
    
    try:
        from lfa.visualization import plot_rowwise_threshold_debug
        plot_rowwise_threshold_debug(an)  # uses an.corrected_image, an.binary_mask, an._rowwise_debug
    except Exception as e:
        print(f"(Skipping extra debug plot: {e})")

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
    
    
    # # Rowwise Thresholding Visual
    # visualize_rowwise_thresholding(
    #     img_path,
    #     stat="mean",
    #     smooth_ksize=91,
    #     k=1.5,
    #     exclude_center_frac_OLD=0.0,
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



# def main_negative():
#     neg_img_path = 'LFAIMAGES/SP 2-19/SP_neg.png'

#     test_bg_mean = mean_intensity_under_saved_mask(
#         img_path=neg_img_path,
#         mask_prefix="saved_test_band",
#     )

#     print("\n=== NEGATIVE STRIP TEST-BAND BACKGROUND INTENSITY ===")
#     print(f"Mean grayscale intensity at test-band location: {test_bg_mean:.3f}")

#     # Visualize the saved mask alone
#     show_saved_mask(mask_prefix="saved_test_band")

#     # Visualize the mask overlaid on the negative image
#     show_mask_overlay_on_image(
#         img_path=neg_img_path,
#         mask_prefix="saved_test_band",
#         alpha=0.35,
#     )
    
    
def main_negative():
    img_path = 'LFAIMAGES/SP 2-19/SP_neg.png'

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
        debug_plots=False,
    )

    # This works for NEGATIVE or POSITIVE:
    ri = compute_relative_intensity_with_saved_test_mask(
        an,
        results,
        mask_prefix="saved_test_band",  # same prefix used when you saved the test mask
    )

    print("\n=== RELATIVE TEST/CONTROL USING SAVED TEST MASK ===")
    print(f"Classifier status:        {results.get('status')}")
    print(f"Background (p90 gray):    {ri['background_gray_p90']:.3f}")
    print(f"Control mean gray:        {ri['control_mean']:.3f}")
    print(f"Test mean gray:           {ri['test_mean']:.3f}")
    print(f"Control signal:           {ri['control_signal']:.3f}")
    print(f"Test signal:              {ri['test_signal']:.3f}")
    print(f"Test / Control signal:    {ri['relative_intensity']:.4f}")

    # Visual overlay: control mask (green) + saved test mask (red)
    visualize_saved_test_and_control_masks(
        an,
        results,
        mask_prefix="saved_test_band",
        alpha_test=0.40,
        alpha_ctrl=0.40,
        save_path=None,   # or "overlay_neg_strip.png"
    )

    # Can also merge these into `results`:
    results.update(ri)



from lfa.analysis import run_analysis, calculate_lod_from_negatives

def main_negative_lod():
    negative_paths = [
        'LFAIMAGES/controls/IMG_8390 - blank 1_cropped.JPEG',
        "LFAIMAGES/controls/IMG_0615 - blank 2_cropped.jpeg",
        "LFAIMAGES/controls/IMG_0796 - blank 3_cropped.jpeg",
    ]

    negative_results = []
    analyzers = []

    for img_path in negative_paths:
        an = SimpleLFAAnalyzer(img_path)
        results = run_analysis(
            an,
            bg='morph',
            ksize=51,
            k=1.5, # this is how many SD above is a band
            smooth_ksize=91, # 1d median filter smoothing
            normalize=False,
            denoise=False,
            binarize_mode="rowwise",
            debug_plots=False,
        )

        analyzers.append(an)
        negative_results.append(results)

        print("\nImage:", img_path)
        print("Control run:", results.get("top_runs"))
        print("Estimated test run:", results.get("estimated_test_run"))
        print("Control signal:", results.get("control_signal"))
        print("Projected test signal:", results.get("test_signal"))
        print("Relative intensity:", results.get("relative_intensity"))

    # Calculate LOD across all negatives
    lod_results = calculate_lod_from_negatives(negative_results)

    print("\n" + "=" * 60)
    print("LOD RESULTS")
    print("=" * 60)

    print("Negative relative intensities:", lod_results["negative_relative_intensities"])
    print("Mean negative RI:", lod_results["mean_negative_relative_intensity"])
    print("SD negative RI:", lod_results["sd_negative_relative_intensity"])
    print("LOD (mean + 3 SD):", lod_results["lod_relative_intensity"])
    print("=" * 60)

    # Show all 3 negative strips together at the end
    from lfa.visualization import visualize_negative_projected_rois_panel

    visualize_negative_projected_rois_panel(
        analyzers, negative_results,
        save_path="negative_LFA_projected_ROIs.png"
    )
            
if __name__ == "__main__":
    # main()
    # main_negative()
    main_negative_lod()
