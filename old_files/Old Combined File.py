# lfa/image_processing.py
import numpy as np
import cv2


# =============================================================================
# PREPROCESS (moved from class.preprocess)
# =============================================================================
def preprocess(an):
    """Convert to grayscale and invert (so pink lines have HIGH values)"""
    # Grayscale
    an.gray_image = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2GRAY)

    # Invert: pink lines are darker than white background
    # After inversion: pink lines will be BRIGHT
    an.inverted_image = 255 - an.gray_image

    return an.inverted_image


# =============================================================================
# BACKGROUND SUBTRACTION (moved from class.subtract_background)
# =============================================================================
def subtract_background(an, method="morph", ksize=51, normalize=False, denoise=False):
    """
    Background subtraction on the inverted image.

    Parameters
    ----------
    method : str
        "morph" = morphological opening (recommended)
        "blur"  = large Gaussian blur background estimate
    ksize : int
        Kernel size for background estimation (odd is best).
        Larger = more aggressive background removal.

    Returns
    -------
    corrected : np.ndarray (uint8)
        Background-subtracted version of inverted image.
    """

    if an.inverted_image is None:
        raise ValueError("Run preprocess() first.")

    img = an.inverted_image.astype(np.uint8)

    if method == "tophat":
        # BETTER KERNEL FOR HORIZONTAL LINES
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            # (ksize * 3, ksize)   # wide horizontally
            (max(3, int(ksize * 3)), max(3, int(ksize)))
        )

        # Use white top-hat instead of manual subtract
        corrected = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

        # If you still want background saved:
        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    elif method == "morph":
        # Large kernel removes thin bright lines, keeps smooth background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(img, background)  # clips at 0 automatically for uint8

    elif method == "morph_ellips_mod":
        kh = ksize                  # height
        kw = int(ksize * 1.3)       # width multiplier (3–10 is typical)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(img, background)

    elif method == "morph_rect":
        # Horizontal-aware opening kernel
        kh = ksize        # height
        kw = ksize * 5    # width (tune 3–10×)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(img, background)

    elif method == "blur":
        # Smooth illumination background estimate
        if ksize % 2 == 0:
            ksize += 1  # GaussianBlur prefers odd
        background = cv2.GaussianBlur(img, (ksize, ksize), 0)
        corrected = cv2.subtract(img, background)  # clips at 0 automatically for uint8

    else:
        raise ValueError("method must be 'morph' or 'blur' or 'tophat'")

    # Post-processing AFTER corrected is finalized
    if normalize:
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    if denoise:
        corrected = cv2.medianBlur(corrected, 3)

    an.background_image = background
    an.corrected_image = corrected.astype(np.uint8)
    return an.corrected_image


# =============================================================================
# OTSU BINARIZATION (moved from class.otsu_binarize_corrected)
# =============================================================================
def otsu_binarize_corrected(an, blur_ksize=2, invert_mask=False, otsu_scale=0.75):
    """
    Apply Otsu thresholding to the background-subtracted (corrected) image
    to separate foreground (line) vs background/noise.

    Parameters
    ----------
    blur_ksize : int
        Optional Gaussian blur kernel size before Otsu (odd). Helps reduce speckle.
        Set to 0 or 1 to skip blurring.
    invert_mask : bool
        If True, invert the binary mask (swap foreground/background).

    Returns
    -------
    thresh : float
        Otsu threshold value.
    mask : np.ndarray (uint8)
        Binary mask, values in {0,255}.
    """
    if an.corrected_image is None:
        raise ValueError("Run preprocess() and subtract_background() first.")

    img = an.corrected_image.copy()

    # Optional denoise to make histogram bimodality cleaner
    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # Otsu threshold
    # Foreground in corrected image should be BRIGHT (since inverted), so THRESH_BINARY is appropriate
    t_otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Make threshold less aggressive
    t_used = max(0.0, float(otsu_scale) * float(t_otsu))

    # Apply threshold
    _, mask = cv2.threshold(img, t_used, 255, cv2.THRESH_BINARY)

    if invert_mask:
        mask = cv2.bitwise_not(mask)

    an.otsu_threshold = float(t_used)
    an.binary_mask = mask
    return an.otsu_threshold, an.binary_mask


# =============================================================================
# ROWWISE THRESHOLDING (moved from class.* rowwise helpers)
# =============================================================================
def _median_filter_1d(x, ksize):
    """
    1D median filter with reflect padding.
    x: (H,) float array
    ksize: odd int >= 3
    """
    k = int(ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    pad = k // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    # sliding window view -> median over last axis
    w = np.lib.stride_tricks.sliding_window_view(xp, window_shape=k)
    return np.median(w, axis=1)


def _keep_true_runs(hits, min_run=3):
    """
    Keep only contiguous True runs of length >= min_run.
    hits: (H,) bool
    """
    hits = hits.astype(bool)
    out = np.zeros_like(hits, dtype=bool)
    h = len(hits)

    i = 0
    while i < h:
        if not hits[i]:
            i += 1
            continue
        j = i
        while j < h and hits[j]:
            j += 1
        if (j - i) >= min_run:
            out[i:j] = True
        i = j
    return out


def _expand_rows(hits, radius=2):
    """
    Expand True rows by +/- radius using 1D dilation.
    """
    hits = hits.astype(np.uint8)
    k = 2 * int(radius) + 1
    kernel = np.ones(k, dtype=np.uint8)
    expanded = np.convolve(hits, kernel, mode="same") > 0
    return expanded


def detect_band_rows(
    img,
    stat="median",
    smooth_ksize=51,
    exclude_center_frac=0.20,
    k=4.0,
):
    """
    Build a per-row adaptive threshold from row statistics.

    Idea:
    row_score[i]   = mean/median intensity of row i
    baseline[i]    = smoothed(row_score)  (captures slow drift / illumination gradient)
    resid[i]       = row_score[i] - baseline[i]
    sigma          = robust scale estimate from resid in "background rows"
    T_row[i]       = baseline[i] + k*sigma

    Returns
    -------
    row_score : (H,) float
    baseline  : (H,) float
    T_row     : (H,) float
    row_hits  : (H,) bool   (rows whose score exceeds their threshold)
    peak_row  : int
    sigma     : float
    """
    img = img.astype(np.float32)

    # Row statistic
    if stat == "mean":
        row_score = np.mean(img, axis=1)
    elif stat == "median":
        row_score = np.median(img, axis=1)
    else:
        raise ValueError("stat must be 'mean' or 'median'")

    h = row_score.shape[0]

    # Smooth row_score to get a drift baseline
    baseline = _median_filter_1d(row_score, smooth_ksize)

    # Robust sigma from residuals on "background rows"
    resid = row_score - baseline

    center = h // 2
    band = int(exclude_center_frac * h)
    mask_rows = np.ones(h, dtype=bool)
    mask_rows[max(0, center - band // 2):min(h, center + band // 2)] = False

    bg_resid = resid[mask_rows]
    med = np.median(bg_resid)
    mad = np.median(np.abs(bg_resid - med)) + 1e-6
    sigma = 1.4826 * mad

    # Per-row threshold
    T_row = baseline + float(k) * float(sigma)

    row_hits = row_score > T_row
    peak_row = int(np.argmax(row_score))

    return row_score, baseline, T_row, row_hits, peak_row, float(sigma)


def rowwise_binarize_corrected(
    an,
    stat="median",
    smooth_ksize=51,
    exclude_center_frac=0.20,
    k=4.0,
    min_run=3,              # minimum consecutive rows to keep as a band
    expand=2,               # expand hits by +/- expand rows (thicken band)
    invert_mask=False
):
    """
    ROW-WISE thresholding: each row is either ON (255 across entire width) or OFF.

    mask[i, :] = 255 if row_score[i] > T_row[i], else 0

    Extra cleanup:
    - keep only runs of True rows with length >= min_run
    - expand bands vertically by +/- expand rows
    """
    if an.corrected_image is None:
        raise ValueError("Run preprocess() and subtract_background() first.")

    img = an.corrected_image.astype(np.float32)

    row_score, baseline, T_row, row_hits, peak_row, sigma = detect_band_rows(
        img,
        stat=stat,
        smooth_ksize=smooth_ksize,
        exclude_center_frac=exclude_center_frac,
        k=k,
    )

    # --- keep only long-enough contiguous runs (remove isolated hit rows) ---
    hits = row_hits.copy()
    if min_run and min_run > 1:
        hits = _keep_true_runs(hits, min_run=min_run)

    # --- expand vertically (optional) ---
    if expand and expand > 0:
        hits = _expand_rows(hits, radius=expand)

    # --- build ROW-WISE mask (entire row on/off) ---
    mask = (hits[:, None].astype(np.uint8) * 255)
    mask = np.repeat(mask, an.corrected_image.shape[1], axis=1)

    if invert_mask:
        mask = cv2.bitwise_not(mask)

    an.binary_mask = mask
    an.otsu_threshold = None
    an._rowwise_debug = {
        "row_score": row_score,
        "baseline": baseline,
        "T_row": T_row,
        "row_hits": row_hits,
        "row_hits_clean": hits,
        "peak_row": peak_row,
        "sigma": sigma,
        "stat": stat,
        "smooth_ksize": smooth_ksize,
        "k": k,
        "min_run": min_run,
        "expand": expand,
    }
    return T_row, mask


# =============================================================================
# BACKGROUND ESTIMATE (moved from class.estimate_background_percentile_mask)
# =============================================================================
def estimate_background_percentile_mask(
    an,
    which="inverted",          # use RAW inverted for quant
    bg_percentile=50.0,        # 50=median, 60–80 can be safer
    exclude_top=0.10,          # don't include very top edge if it has artifacts
    exclude_bottom=0.10,       # don't include very bottom edge (common gradient/edge)
    exclude_center_band=0.20,  # exclude a middle band where lines might exist
    edge_margin=0.05           # exclude left/right edges
):
    """
    Estimate background from a 'blank mask' using a percentile of selected pixels.

    Returns
    -------
    background_value : float
    mask : np.ndarray(bool)
    """

    # Pick image to estimate background on
    if which == "inverted":
        if an.inverted_image is None:
            raise ValueError("Run preprocess() first.")
        img = an.inverted_image.astype(np.float32)
    elif which == "gray":
        if an.gray_image is None:
            raise ValueError("Run preprocess() first.")
        img = an.gray_image.astype(np.float32)
    else:
        raise ValueError("which must be 'inverted' or 'gray'")

    h, w = img.shape
    mask = np.ones((h, w), dtype=bool)

    # Exclude top/bottom bands
    top_cut = int(exclude_top * h)
    bot_cut = int(exclude_bottom * h)
    if top_cut > 0:
        mask[:top_cut, :] = False
    if bot_cut > 0:
        mask[h - bot_cut:, :] = False

    # Exclude center band (where lines often live)
    center_half = int(0.5 * exclude_center_band * h)
    center = h // 2
    mask[max(0, center - center_half):min(h, center + center_half), :] = False

    # Exclude left/right edges
    edge = int(edge_margin * w)
    if edge > 0:
        mask[:, :edge] = False
        mask[:, w - edge:] = False

    # Pull masked pixels and compute percentile
    pixels = img[mask]
    if pixels.size == 0:
        raise ValueError("Blank mask removed all pixels; relax exclusions.")

    background_value = float(np.percentile(pixels, bg_percentile))
    return background_value, mask