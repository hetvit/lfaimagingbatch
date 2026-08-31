# from image_processing import make_band_sampling_mask
import numpy as np
import cv2
from . import image_processing as ip
import os
import json
import matplotlib.pyplot as plt


import json
from pathlib import Path
def save_test_band_mask(an, info, out_prefix="test_band"):
    """
    Given a POSITIVE strip and its classification info from run_analysis(),
    create and save a sampling mask for the TEST band.

    - an: SimpleLFAAnalyzer (already processed / cropped)
    - info: dict returned by run_analysis() (must be POSITIVE)
    - out_prefix: base path (without extension) for saving mask + metadata

    Saves:
      out_prefix + "_mask.npy"   (uint8 0/255 mask)
      out_prefix + "_meta.json"  (crop bounds, mask shape, row_radius, edge_margin_frac)
    """
    if info.get("status") != "POSITIVE":
        raise ValueError("save_test_band_mask requires a POSITIVE strip with top+bottom bands.")

    # bottom_run is your TEST band
    bottom_run = info["bottom_runs"][0]

    # same defaults as your band_mean_intensity_on_original
    row_radius = 10
    edge_margin_frac = 0.01

    # use your existing helper to build sampling mask
    mask = ip.make_band_sampling_mask(
        an,
        bottom_run,
        edge_margin_frac=edge_margin_frac,
        row_radius=row_radius,
    )

    # Use pathlib correctly
    out_prefix = Path(out_prefix)
    # e.g. "saved_test_band"      -> saved_test_band_mask.npy
    #      "folder/foo.png"       -> folder/foo_mask.npy
    mask_path = out_prefix.with_name(out_prefix.stem + "_mask.npy")
    meta_path = out_prefix.with_name(out_prefix.stem + "_meta.json")
    
    
    # save mask
    np.save(mask_path, mask)

    # also save crop bounds + mask shape + sampling params
    crop_bounds = getattr(an, "_crop_bounds", None)
    meta = {
        "crop_bounds": crop_bounds,           # (top, bottom, left, right)
        "mask_shape": mask.shape,            # (H, W)
        "row_radius": row_radius,
        "edge_margin_frac": edge_margin_frac,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved test band mask to {mask_path}")
    print(f"Saved metadata to {meta_path}")
    return str(mask_path), str(meta_path)

def mean_intensity_under_saved_mask(img_path, mask_prefix="saved_test_band"):
    import json
    import os
    import numpy as np
    import cv2

    # Build filenames WITHOUT pathlib
    base = str(mask_prefix)
    mask_path = base + "_mask.npy"
    meta_path = base + "_meta.json"

    # Check files
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Load mask + metadata
    mask = np.load(mask_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    crop_bounds = meta.get("crop_bounds", None)
    if crop_bounds is None:
        raise ValueError("No crop_bounds stored in the mask metadata.")

    top, bottom, left, right = crop_bounds

    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    H, W = gray.shape
    # clamp
    top = max(0, min(top, H - 1))
    bottom = max(0, min(bottom, H - 1))
    left = max(0, min(left, W - 1))
    right = max(0, min(right, W - 1))

    # Crop
    gray_crop = gray[top:bottom+1, left:right+1]

    if gray_crop.shape != tuple(meta["mask_shape"]):
        raise ValueError(
            f"Cropped image shape {gray_crop.shape} does not match mask shape {tuple(meta['mask_shape'])}. "
            "Camera framing must be consistent."
        )

    # Extract intensity under mask
    roi = gray_crop[mask > 0]
    if roi.size == 0:
        return float("nan")

    return float(np.mean(roi))


def show_saved_mask(mask_prefix="saved_test_band", save_path=None):
    """
    Visualize the saved test-band mask by itself.

    - mask_prefix: base name used when saving the mask (e.g., "saved_test_band")
    - save_path: if not None, saves the figure instead of (or in addition to) showing it
    """
    base = str(mask_prefix)
    mask_path = base + "_mask.npy"

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = np.load(mask_path)

    plt.figure(figsize=(6, 4))
    plt.imshow(mask, cmap="gray")
    plt.title(f"Saved test-band mask ({mask_path})")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved mask visualization to {save_path}")
    else:
        plt.show()
        
        
def show_mask_overlay_on_image(
    img_path,
    mask_prefix="saved_test_band",
    alpha=0.35,
    save_path=None,
):
    """
    Visualize the saved test-band mask overlaid on a NEW image.

    - img_path: path to the new (e.g., negative) strip
    - mask_prefix: base name used when saving the mask (e.g., "saved_test_band")
    - alpha: transparency of the mask overlay
    - save_path: if not None, saves the figure instead of (or in addition to) showing it
    """
    base = str(mask_prefix)
    mask_path = base + "_mask.npy"
    meta_path = base + "_meta.json"

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    mask = np.load(mask_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    crop_bounds = meta.get("crop_bounds", None)
    if crop_bounds is None:
        raise ValueError("No crop_bounds stored in the mask metadata.")

    top, bottom, left, right = crop_bounds

    # Load the new image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Convert BGR -> RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W, _ = img_rgb.shape
    # clamp crop bounds just in case
    top = max(0, min(top, H - 1))
    bottom = max(0, min(bottom, H - 1))
    left = max(0, min(left, W - 1))
    right = max(0, min(right, W - 1))

    img_crop = img_rgb[top:bottom+1, left:right+1]

    if img_crop.shape[:2] != mask.shape:
        raise ValueError(
            f"Cropped image shape {img_crop.shape[:2]} != mask shape {mask.shape}. "
            "Check that camera framing and crop bounds are consistent."
        )

    # Binary mask → bool
    mask_bool = mask > 0

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_crop)
    plt.title("Cropped new image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_crop)
    # Overlay the mask as a transparent red heatmap
    plt.imshow(mask_bool, cmap="Reds", alpha=alpha)
    plt.title("Mask overlay on new image")
    plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved overlay visualization to {save_path}")
    else:
        plt.show()
        
   
def _reproject_saved_test_mask_to_current(an, mask_prefix="saved_test_band"):
    """
    Reproject the saved test-band mask (from a reference positive image)
    into the coordinate system of the CURRENT analyzer `an`.

    IMPORTANT CHANGE:
      We now treat the saved mask as being in the *cropped* coordinate
      system of the reference positive strip, and we simply warp (resize)
      it to the size of this image's cropped original.

      This avoids assuming the same absolute crop bounds in the raw
      camera frame, which is what was causing the misalignment.
    """
    base = str(mask_prefix)

    # Support both naming conventions:
    test_mask_path_new = base + "_test_mask.npy"
    test_mask_path_old = base + "_mask.npy"
    meta_path = base + "_meta.json"  # still read for debugging if present

    if os.path.exists(test_mask_path_new):
        test_mask_path = test_mask_path_new
    elif os.path.exists(test_mask_path_old):
        test_mask_path = test_mask_path_old
    else:
        raise FileNotFoundError(
            f"No test mask found for prefix '{mask_prefix}'. "
            f"Tried: {test_mask_path_new} and {test_mask_path_old}"
        )

    mask_pos = np.load(test_mask_path).astype(np.uint8)

    if an.original_image is None:
        raise ValueError("an.original_image is None; did SimpleLFAAnalyzer load correctly?")

    # Current cropped image shape
    H_neg, W_neg = an.original_image.shape[:2]

    # If shapes already match, just binarize & return
    if mask_pos.shape == (H_neg, W_neg):
        mask_neg = np.zeros_like(mask_pos, dtype=np.uint8)
        mask_neg[mask_pos > 0] = 255
        return mask_neg

    # Otherwise: warp the saved cropped mask to this cropped image size
    # using nearest-neighbor so the mask stays binary-ish.
    mask_resized = cv2.resize(
        mask_pos,
        (W_neg, H_neg),
        interpolation=cv2.INTER_NEAREST,
    )

    mask_neg = np.zeros((H_neg, W_neg), dtype=np.uint8)
    mask_neg[mask_resized > 0] = 255

    return mask_neg
     
def compute_relative_intensity_with_saved_test_mask(
    an,
    info,
    mask_prefix="saved_test_band",
):
    """
    Compute relative TEST/CONTROL intensity for ANY image (positive or negative)
    by combining:

      - CONTROL band from this image (via info["top_runs"][0])
      - TEST band from a saved mask (from a reference positive image),
        reprojected into this image's cropped coordinates.

    Uses the same convention as compute_relative_intensity_from_runs:
      signal = background_mean - band_mean   (since band is darker)
      ratio  = test_signal / control_signal

    Returns
    -------
    dict with keys:
      - background_gray_p90
      - control_mean
      - test_mean
      - control_signal
      - test_signal
      - relative_intensity
    """
    # 1) Get control band for this image (top run)
    top_runs = info.get("top_runs", [])
    if not top_runs:
        raise ValueError("No top_runs found in info; cannot determine control band.")
    ctrl_run = top_runs[0]  # (s, e) row indices

    if an.original_image is None:
        raise ValueError("an.original_image is None; did SimpleLFAAnalyzer load correctly?")

    # 2) Reproject saved TEST mask into this image's cropped coordinates
    test_mask = _reproject_saved_test_mask_to_current(an, mask_prefix=mask_prefix)

    # 3) Grayscale of the CURRENT cropped strip
    gray = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    if test_mask.shape != (H, W):
        raise ValueError(
            f"Reprojected test mask shape {test_mask.shape} does not match "
            f"current cropped image shape {(H, W)}."
        )

    # 4) Background from this image (bright background)
    bg = float(np.percentile(gray, 90))

    # 5) CONTROL mean using your existing band_mean_intensity_on_original
    ctrl_mean = ip.band_mean_intensity_on_original(an, ctrl_run)

    # 6) TEST mean using the reprojected test mask
    test_roi = gray[test_mask > 0]
    if test_roi.size == 0:
        raise ValueError("Reprojected test mask has no nonzero pixels on this image.")
    test_mean = float(np.mean(test_roi))

    # 7) Signals: darker band => larger signal
    ctrl_signal = max(bg - ctrl_mean, 1e-6)  # avoid divide-by-zero
    test_signal = max(bg - test_mean, 0.0)

    ratio = float(test_signal / ctrl_signal)

    return {
        "background_gray_p90": bg,
        "control_mean": ctrl_mean,
        "test_mean": test_mean,
        "control_signal": ctrl_signal,
        "test_signal": test_signal,
        "relative_intensity": ratio,
    }
    
    
def visualize_saved_test_mask_overlay(
    an,
    mask_prefix="saved_test_band",
    alpha=0.35,
    save_path=None,
):
    """
    Visualize the reprojected TEST-band mask overlaid on the CURRENT
    cropped original image.

    Assumes:
      - `an` has already been processed by your usual pipeline
        (preprocess -> auto_crop_remove_bright_edges -> subtract_background
         -> rowwise_binarize_corrected), so:
          * an.original_image is cropped
          * an._crop_bounds is set by auto_crop_remove_bright_edges
      - A saved test mask + meta exist for the reference positive
        under `mask_prefix`.

    Parameters
    ----------
    an : SimpleLFAAnalyzer
        Analyzer for the CURRENT image (positive or negative).
    mask_prefix : str
        Prefix used when saving the reference-positive test mask,
        e.g. "saved_test_band".
    alpha : float
        Transparency for the overlay (0 = invisible, 1 = fully opaque).
    save_path : str or None
        If given, save the figure to this path instead of showing it.
    """
    # 1) Reproject saved test mask into this image's cropped coordinates
    test_mask = _reproject_saved_test_mask_to_current(an, mask_prefix=mask_prefix)

    # 2) Get cropped original image
    if an.original_image is None:
        raise ValueError("an.original_image is None; did you run the pipeline on `an`?")

    img_rgb = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2RGB)

    H, W, _ = img_rgb.shape
    if test_mask.shape != (H, W):
        raise ValueError(
            f"Reprojected test mask shape {test_mask.shape} != cropped image shape {(H, W)}."
        )

    mask_bool = test_mask > 0

    plt.figure(figsize=(6, 4))
    plt.imshow(img_rgb)
    plt.imshow(mask_bool, cmap="Reds", alpha=alpha)
    plt.title("Saved TEST mask overlay on cropped original")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved overlay visualization to {save_path}")
        plt.close()
    else:
        plt.show()
        
        
def visualize_saved_test_and_control_masks(
    an,
    info,
    mask_prefix="saved_test_band",
    alpha_test=0.40,
    alpha_ctrl=0.40,
    edge_margin_frac=0.01,
    row_radius=10,
    save_path=None,
):
    """
    Visualize BOTH:
      - the saved TEST-band mask (from a reference positive, reprojected), and
      - the CONTROL-band sampling mask for THIS image (from top_runs[0])

    overlaid on the CURRENT cropped original image.

    TEST ROI  -> red overlay
    CONTROL ROI -> green overlay

    Parameters
    ----------
    an : SimpleLFAAnalyzer
        Analyzer for the current image; must have been processed by your pipeline
        (so that an.original_image is cropped and an._crop_bounds exists).
    info : dict
        Output from run_analysis(an, ...); must contain "top_runs" with at least
        one run (control band).
    mask_prefix : str
        Prefix used when saving the reference-positive test mask.
    alpha_test : float
        Alpha for the TEST overlay.
    alpha_ctrl : float
        Alpha for the CONTROL overlay.
    edge_margin_frac : float
        Passed to make_band_sampling_mask for the control ROI.
    row_radius : int
        Passed to make_band_sampling_mask for the control ROI.
    save_path : str or None
        If given, saves the figure to this path; otherwise shows it.
    """
    if an.original_image is None:
        raise ValueError("an.original_image is None; did you run the pipeline on `an`?")

    # --- 1) Get CONTROL run for this image ---
    top_runs = info.get("top_runs", [])
    if not top_runs:
        raise ValueError("No top_runs found in info; cannot determine control band.")
    ctrl_run = top_runs[0]  # (s, e)

    # Build control sampling mask (0/255) using your existing logic
    ctrl_mask = ip.make_band_sampling_mask(
        an,
        ctrl_run,
        edge_margin_frac=edge_margin_frac,
        row_radius=row_radius,
    )

    # --- 2) Reproject saved TEST mask into this image's cropped coordinates ---
    test_mask = _reproject_saved_test_mask_to_current(an, mask_prefix=mask_prefix)

    # --- 3) Prepare cropped original image in RGB ---
    img_rgb = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    if ctrl_mask.shape != (H, W):
        raise ValueError(
            f"Control mask shape {ctrl_mask.shape} != cropped image shape {(H, W)}."
        )
    if test_mask.shape != (H, W):
        raise ValueError(
            f"Test mask shape {test_mask.shape} != cropped image shape {(H, W)}."
        )

    ctrl_bool = ctrl_mask > 0
    test_bool = test_mask > 0

    # --- 4) Plot overlays ---
    plt.figure(figsize=(6, 4))
    plt.imshow(img_rgb)
    # Control in green
    plt.imshow(ctrl_bool, cmap="Greens", alpha=alpha_ctrl)
    # Test in red
    plt.imshow(test_bool, cmap="Reds", alpha=alpha_test)

    plt.title("Control (green) + Test (red) masks on cropped original")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved overlay visualization to {save_path}")
        plt.close()
    else:
        plt.show()
        