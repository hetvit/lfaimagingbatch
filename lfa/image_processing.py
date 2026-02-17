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


def _runs_from_hits(hits: np.ndarray):
    """
    Convert boolean hits (H,) into list of (start, end) inclusive indices for True-runs.
    """
    hits = hits.astype(bool)
    runs = []
    h = len(hits)
    i = 0
    while i < h:
        if not hits[i]:
            i += 1
            continue
        j = i
        while j < h and hits[j]:
            j += 1
        # run is i .. (j-1)
        runs.append((i, j - 1))
        i = j
    return runs


# def _keep_top_k_runs(hits, resid, k_runs=2, score_mode="auc"):
#     """
#     Keep only top-k contiguous True runs in `hits`, based on a score computed from `resid`.

#     hits: (H,) bool
#     resid: (H,) float  (e.g., row_score - baseline)
#     score_mode:
#       - "auc": sum of positive residual inside the run (robust)
#       - "peak": max residual inside the run (sensitive)
#     """
#     runs = _runs_from_hits(hits)
#     if len(runs) <= k_runs:
#         return hits  # nothing to prune

#     scores = []
#     for (s, e) in runs:
#         r = resid[s:e+1]
#         if score_mode == "auc":
#             score = float(np.sum(np.maximum(r, 0.0)))
#         elif score_mode == "peak":
#             score = float(np.max(r))
#         else:
#             raise ValueError("score_mode must be 'auc' or 'peak'")
#         scores.append(score)

#     # pick indices of top-k scores
#     top_idx = np.argsort(scores)[::-1][:k_runs]

#     kept = np.zeros_like(hits, dtype=bool)
#     for idx in top_idx:
#         s, e = runs[int(idx)]
#         kept[s:e+1] = True

#     return kept

def keep_best_run_per_half(hits, resid, mid, score_mode="auc"):
    runs = _runs_from_hits(hits)
    if not runs:
        return hits

    def score_run(s, e):
        r = resid[s:e+1]
        if score_mode == "auc":
            return float(np.sum(np.maximum(r, 0.0)))
        elif score_mode == "peak":
            return float(np.max(r))
        else:
            raise ValueError("score_mode must be 'auc' or 'peak'")

    top = []
    bot = []
    for (s, e) in runs:
        center = 0.5 * (s + e)
        (top if center < mid else bot).append((s, e))

    kept = np.zeros_like(hits, dtype=bool)

    if top:
        best = max(top, key=lambda r: score_run(*r))
        kept[best[0]:best[1]+1] = True

    if bot:
        best = max(bot, key=lambda r: score_run(*r))
        kept[best[0]:best[1]+1] = True

    return kept


def detect_band_rows(
    img,
    stat="mean",
    smooth_ksize=51,
    exclude_center_frac=0.0, # remove this feature cuz it's wrong
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

    # Row statistic -> get a 1D intensity signal from 2D image, a single value for each row of the image -> vertical intensity profile
    if stat == "mean":
        row_score = np.mean(img, axis=1)
    elif stat == "median":
        row_score = np.median(img, axis=1)
    else:
        raise ValueError("stat must be 'mean' or 'median'")

    h = row_score.shape[0] # number of rows

    # Smooth row_score to get a drift baseline -> large-window median filter on row signal, removing narrow peaks but keeps slow trends, smooths out row_score
    baseline = _median_filter_1d(row_score, smooth_ksize)

    # Robust sigma from residuals on "background rows" -> bands have positive residuals, normalizing. Isolate extra brightness beyond background
    resid = row_score - baseline

    # Exclude center rows from noise estimation -> but this does it at the center... which might be wrong?
    center = h // 2
    band = int(exclude_center_frac * h)
    mask_rows = np.ones(h, dtype=bool)
    mask_rows[max(0, center - band // 2):min(h, center + band // 2)] = False

    # estimate noise level, using MAD (robust to outliers). sigma is the "noise floor scale" in row-score units
    bg_resid = resid[mask_rows]
    med = np.median(bg_resid)
    mad = np.median(np.abs(bg_resid - med)) + 1e-6
    sigma = 1.4826 * mad # estimate SD from MAD if noise is roughly normal

    # Per-row threshold -> threshold for each row -> z-score test -> 
    T_row = baseline + float(k) * float(sigma) # create threshold for each row -> band is a line row if > baseline + (k * sigma brighter than expected)
    row_hits = row_score > T_row # apply threshold
    peak_row = int(np.argmax(row_score)) # find the single strongest row peak

    return row_score, baseline, T_row, row_hits, peak_row, float(sigma)


def rowwise_binarize_corrected(
    an,
    stat="median",
    smooth_ksize=51,
    exclude_center_frac=0.00, # set to 0
    k=4.0,
    min_run=3,              # minimum consecutive rows to keep as a band
    expand=2,               # expand hits by +/- expand rows (thicken band)
    invert_mask=False,
    keep_top_bands=2,          # keep only top 2 brighest continguous peaks
    band_score_mode="auc",     # NEW: "auc" (recommended) or "peak"
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
    resid = row_score - baseline

    # --- keep only long-enough contiguous runs (remove isolated hit rows) ---
    hits = row_hits.copy()
    if min_run and min_run > 1:
        hits = _keep_true_runs(hits, min_run=min_run)

    # --- keep only top-N bands (top peaks) --- -> THIS IS BETTER
    if keep_top_bands is not None and keep_top_bands > 0:
        # Use helper from above (paste into your class or module)
        # hits = _keep_top_k_runs(hits, resid, k_runs=int(keep_top_bands), score_mode=band_score_mode)
        
        # NEW:
        H = len(hits)
        mid = H // 2

        # keep strongest run in top + strongest run in bottom
        hits = keep_best_run_per_half(hits, resid, mid, score_mode=band_score_mode)

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
        "keep_top_bands": keep_top_bands,
        "band_score_mode": band_score_mode,
    }
    return T_row, mask


### CLASSIFICATION

def classify_two_band_top_bottom(an, min_bands=1, max_bands=2):
    """
    Classification based on detected rowwise True runs (bands):
      - POSITIVE: exactly 2 bands total: one in top half, one in bottom half
      - NEGATIVE: exactly 1 band total and it is in top half (control-only)
      - INVALID: anything else (e.g., only bottom band, >2 bands, etc.)
    """
    if not hasattr(an, "_rowwise_debug"):
        return {"status": "INVALID", "reason": "No rowwise debug info. Run rowwise_binarize_corrected() first."}

    hits = an._rowwise_debug["row_hits_clean"]
    runs = _runs_from_hits(hits)

    H = len(hits)
    mid = H // 2

    def run_center(run):
        s, e = run
        return 0.5 * (s + e)

    top_runs = [r for r in runs if run_center(r) < mid]
    bot_runs = [r for r in runs if run_center(r) >= mid]

    # core logic
    if len(runs) == 2 and len(top_runs) == 1 and len(bot_runs) == 1:
        status = "POSITIVE"
    elif len(runs) == 1 and len(top_runs) == 1 and len(bot_runs) == 0:
        status = "NEGATIVE"
    else:
        status = "INVALID"

    return {
        "status": status,
        "runs": runs,
        "top_runs": top_runs,
        "bottom_runs": bot_runs,
        "mid_row": mid,
        "num_bands": len(runs),
    }
    
def band_mean_intensity_on_original(an, run, half=None, edge_margin_frac=0.05):
    """
    Compute mean intensity in the ORIGINAL image for rows in `run` (s,e),
    but only using pixels away from left/right edges to avoid edge artifacts.

    We use grayscale ORIGINAL (not inverted).
    Lower value = darker/pinker band.
    """
    if an.original_image is None:
        raise ValueError("original_image missing")

    gray = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    H, W = gray.shape
    s, e = run

    # clamp
    s = max(0, int(s)); e = min(H - 1, int(e))
    if e <= s:
        return float("nan")

    edge = int(edge_margin_frac * W)
    x0 = edge
    x1 = W - edge

    roi = gray[s:e+1, x0:x1]
    return float(np.mean(roi))


def compute_relative_intensity_from_runs(an, top_run, bottom_run):
    """
    Relative intensity = (control - test darkness?) depends on convention.

    Here we define "signal" as darkness relative to local background:
      signal = background_mean - band_mean   (since band is darker)
    ratio = test_signal / control_signal
    """
    # background estimate from ORIGINAL grayscale using your existing blank-mask method (or simple percentile)
    # We'll reuse your existing percentile mask on inverted/gray if you want, but simplest:
    gray = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = float(np.percentile(gray, 90))  # bright background

    ctrl_mean = band_mean_intensity_on_original(an, top_run)
    test_mean = band_mean_intensity_on_original(an, bottom_run)

    ctrl_signal = max(bg - ctrl_mean, 1e-6)
    test_signal = max(bg - test_mean, 0.0)

    return {
        "background_gray_p90": bg,
        "control_mean": ctrl_mean,
        "test_mean": test_mean,
        "control_signal": ctrl_signal,
        "test_signal": test_signal,
        "relative_intensity": float(test_signal / ctrl_signal),
    }


def classify_one_band_top_only(an):
    """
    Your requested rule:
      - POSITIVE if EXACTLY 1 band run detected AND it's in the TOP half
      - NEGATIVE otherwise (0 bands or >=2 bands)
      - INVALID if exactly 1 band run but it's in the BOTTOM half (problem)

    Returns dict with status + runs.
    """
    if not hasattr(an, "_rowwise_debug"):
        raise ValueError("Run rowwise_binarize_corrected() first (needs an._rowwise_debug).")

    hits = an._rowwise_debug["row_hits_clean"].astype(bool)
    H = hits.shape[0]
    mid = H // 2

    runs = _runs_from_hits(hits)
    n = len(runs)

    status = "NEGATIVE"
    problem = False
    bad_runs = []

    if n == 1:
        s, e = runs[0]
        center = 0.5 * (s + e)
        if center < mid:
            status = "POSITIVE"
        else:
            status = "INVALID"
            problem = True
            bad_runs = [runs[0]]
    else:
        # 0 bands or >=2 bands -> NEGATIVE per your rule
        status = "NEGATIVE"
        # optional: if any band is bottom-only you can still flag, but you didn't ask for that here

    return {
        "status": status,              # "POSITIVE" | "NEGATIVE" | "INVALID"
        "num_bands": n,
        "runs": runs,
        "mid_row": mid,
        "problem_bottom_half_band": problem,
        "bad_runs": bad_runs,
    }

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


def auto_crop_remove_dark_edges(an, dark_thresh=200, tol_frac=0.02, min_size=20):
    """
    Remove dark artifacts touching image edges.

    dark_thresh:
        Pixel value BELOW this is considered dark.
    tol_frac:
        Fraction of dark pixels required in a row/col to trim it.
        (0.02 = 2% dark pixels)
    min_size:
        Minimum remaining height/width allowed after cropping (safety).
    """
    img = an.original_image
    if img is None or img.size == 0:
        raise ValueError("original_image is empty (cv2.imread likely failed or prior crop went wrong).")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    top, bottom = 0, h - 1
    left, right = 0, w - 1

    # # ---- TOP ----
    # while top < bottom and (bottom - top + 1) > min_size:
    #     if np.mean(gray[top, :] < dark_thresh) >= tol_frac:
    #         top += 1
    #     else:
    #         break

    # # ---- BOTTOM ----
    # while bottom > top and (bottom - top + 1) > min_size:
    #     if np.mean(gray[bottom, :] < dark_thresh) >= tol_frac:
    #         bottom -= 1
    #     else:
    #         break

    # ---- LEFT ----
    while left < right and (right - left + 1) > min_size:
        if np.mean(gray[:, left] < dark_thresh) >= tol_frac:
            left += 1
        else:
            break

    # ---- RIGHT ----
    while right > left and (right - left + 1) > min_size:
        if np.mean(gray[:, right] < dark_thresh) >= tol_frac:
            right -= 1
        else:
            break

    cropped = img[top:bottom+1, left:right+1]
    if cropped.size == 0:
        # Safety fallback: do nothing
        return (0, h - 1, 0, w - 1)

    an.original_image = cropped
    return (top, bottom, left, right)