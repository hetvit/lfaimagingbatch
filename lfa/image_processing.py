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


def _prune_weak_runs(hits, resid, sigma, min_peak_sigma=2.0):
    """
    Remove contiguous True runs whose maximum residual is too small.

    hits  : (H,) bool
    resid : (H,) float, row_score - baseline
    sigma : float, robust noise scale from detect_band_rows
    min_peak_sigma : float, minimum peak height in units of sigma
                     (e.g. 2.0 → require peak >= 2σ)

    Returns
    -------
    pruned_hits : (H,) bool
    """
    hits = hits.astype(bool)
    H = len(hits)
    keep = np.zeros(H, dtype=bool)

    runs = _runs_from_hits(hits)
    if sigma <= 0 or not runs:
        return hits  # nothing to prune safely

    for (s, e) in runs:
        peak = float(np.max(resid[s:e+1]))
        if peak >= min_peak_sigma * sigma:
            keep[s:e+1] = True

    return keep

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
    ex. [False, True, True, True, False, False, True, True, False, ...] -> [(1, 3), (6, 7)] (continuous true intervals)
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

def keep_best_run_per_half(hits, resid, mid, score_mode="auc", edge_margin_rows=10):
    """
    Select the best run in the top half and the best run in the bottom half.

    We prefer runs that do NOT touch the top/bottom edges (within
    `edge_margin_rows`), but if a half only has edge runs, we still keep
    the best one there so we don't drop the band entirely.
    """
    runs = _runs_from_hits(hits) #converts hits, an array of booleans for each row, to an array with intervals of True
    if not runs:
        return hits
    
    H = len(hits)

    # Scoring function: AUC accounts for band width; peak is only highest score mattering
    def score_run(s, e):
        r = resid[s:e+1]
        if score_mode == "auc":
            return float(np.sum(np.maximum(r, 0.0)))
        elif score_mode == "peak":
            return float(np.max(r))
        else:
            raise ValueError("score_mode must be 'auc' or 'peak'")

    def is_edge_run(s, e):
        """True if the run touches top or bottom ~edge_margin_rows pixels."""
        # touches rows 0..edge_margin_rows-1
        if s <= edge_margin_rows - 1:
            return True
        # touches rows H-edge_margin_rows .. H-1
        if e >= H - edge_margin_rows:
            return True
        return False

    # split runs into top / bottom halves
    top_runs = []
    bot_runs = []
    for (s, e) in runs:
        center = 0.5 * (s + e) # center row
        if center < mid:
            top_runs.append((s, e))
        else:
            bot_runs.append((s, e))

    kept = np.zeros_like(hits, dtype=bool)

    def select_best(runs_half):
        """
        Given runs in one half:
          1) pick best non-edge run (by score),
          2) if none exist, pick best *edge* run instead.
        """
        if not runs_half:
            return None

        # score all runs
        scored = [(score_run(s, e), s, e) for (s, e) in runs_half]
        scored.sort(reverse=True, key=lambda t: t[0])  # high score first

        # only keep non-edge runs; ignore edge-only halves
        for score, s, e in scored:
            if not is_edge_run(s, e):
                return (s, e)

        # all runs in this half are edge runs -> treat as "no band"
        return None
        
        # THIS VERSION WILL CHECK IF EDGE, AND IF SO, KEEP THE HIGHEST AUC NON-EDGE BAND
        # # 1) Prefer non-edge runs
        # for score, s, e in scored:
        #     if not is_edge_run(s, e):
        #         return (s, e)

        # # 2) All runs here are edge runs → keep the best one
        # _, s_best, e_best = scored[0]
        # return (s_best, e_best)

    # keep strongest run in top half
    best_top = select_best(top_runs)
    if best_top is not None:
        s, e = best_top
        kept[s:e+1] = True

    # keep strongest run in bottom half
    best_bot = select_best(bot_runs)
    if best_bot is not None:
        s, e = best_bot
        kept[s:e+1] = True


    return kept


def detect_band_rows(
    img,
    stat="mean",
    smooth_ksize=51,
    exclude_center_frac_OLD=0.0, # remove this feature cuz it's wrong
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
    img = img.astype(np.float64)

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
    
    # print(list(baseline))

    # Robust sigma from residuals on "background rows" -> bands have positive residuals, normalizing. Isolate extra brightness beyond background
    resid = row_score - baseline

    # # Exclude center rows from noise estimation -> but this does it at the center... which might be wrong?
    center = h // 2
    band = int(exclude_center_frac_OLD * h)
    mask_rows = np.ones(h, dtype=bool)
    mask_rows[max(0, center - band // 2):min(h, center + band // 2)] = False

    # estimate noise level, using MAD (robust to outliers). sigma is the "noise floor scale" in row-score units
    bg_resid = resid[mask_rows]
    med = np.median(bg_resid)
    mad = np.median(np.abs(bg_resid - med)) + 1e-6
    sigma = 1.4826 * mad # estimate SD from MAD if noise is roughly normal
    
    # Per-row threshold -> threshold for each row -> z-score test
    T_row = baseline + float(k) * float(sigma) # create threshold for each row -> band is a line row if > baseline + (k * sigma brighter than expected)
    # print(list(T_row))
    row_hits = row_score > T_row # apply threshold
    # print(f"ROW HITS: {np.sum(row_hits)}")
    peak_row = int(np.argmax(row_score)) # find the single strongest row peak

    return row_score, baseline, T_row, row_hits, peak_row, float(sigma)


def rowwise_binarize_corrected(
    an,
    stat="mean",
    smooth_ksize=51,
    exclude_center_frac=0.00, # set to 0
    k=4.0,
    min_run=3,              # minimum consecutive rows to keep as a band
    expand=2,               # expand hits by +/- expand rows (thicken band)
    invert_mask=False,
    keep_top_bands=2,          # keep only top 2 brighest continguous peaks
    band_score_mode="auc",     # "auc" or "peak"
    min_peak_sigma=2.0,
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
        # exclude_center_frac_OLD=exclude_center_frac, #not using this feature!
        k=k,
    )
    resid = row_score - baseline

    # forbid any hits in the center fraction of rows (ignore center 10% as signal)
    if exclude_center_frac and exclude_center_frac > 0.0:
        H = row_hits.shape[0]
        band = int(exclude_center_frac * H)
        if band > 0:
            center = H // 2
            c0 = max(0, center - band // 2)
            c1 = min(H, center + band // 2)
            row_hits[c0:c1] = False   # center band cannot be "on"

    # --- keep only long-enough contiguous runs (remove isolated hit rows) ---
    hits = row_hits.copy()
    if min_run and min_run > 1:
        hits = _keep_true_runs(hits, min_run=min_run)

     # drop weak runs whose peak residual is too small ---
    if min_peak_sigma is not None and min_peak_sigma > 0 and sigma > 0:
        hits = _prune_weak_runs(hits, resid, sigma, min_peak_sigma=min_peak_sigma)

    # --- keep only top-N bands (top peaks) --- -> THIS IS BETTER
    if keep_top_bands is not None and keep_top_bands > 0:
        # hits = _keep_top_k_runs(hits, resid, k_runs=int(keep_top_bands), score_mode=band_score_mode)
        
        # NEW:
        H = len(hits)
        mid = H // 2

        # keep strongest run in top + strongest run in bottom
        hits = keep_best_run_per_half(hits, resid, mid, score_mode=band_score_mode)

    # --- expand vertically (optional) ---
    if expand and expand > 0:
        hits = _expand_rows(hits, radius=expand)

    # NEW: disallow any hits in the top/bottom edge_margin_rows
    edge_margin_rows = 5
    hits[:edge_margin_rows] = False
    hits[-edge_margin_rows:] = False

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

def classify_two_band_top_bottom(an, min_bands=1, max_bands=2, edge_margin_rows=5):
    """
    Classification based on detected rowwise True runs (bands):
      - POSITIVE: exactly 2 bands total: one in top half, one in bottom half
      - NEGATIVE: exactly 1 band total and it is in top half (control-only)
      - INVALID:
            * anything else (e.g., only bottom band, >2 bands, etc.), or
            * ANY band touches the top/bottom edge rows (within edge_margin_rows)
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

    def is_edge_run(run):
        """True if run touches the top or bottom edge within edge_margin_rows."""
        s, e = run
        # top region: rows [0 .. edge_margin_rows-1]
        if s <= edge_margin_rows - 1:
            return True
        # bottom region: rows [H-edge_margin_rows .. H-1]
        if e >= H - edge_margin_rows:
            return True
        return False

    # invalidate if any band is on the edge
    if any(is_edge_run(r) for r in runs):
        return {
            "status": "INVALID",
            "runs": runs,
            "top_runs": [],
            "bottom_runs": [],
            "mid_row": mid,
            "num_bands": len(runs),
            "reason": f"Band touches top/bottom {edge_margin_rows} rows.",
        }


    top_runs = [r for r in runs if run_center(r) < mid]
    bot_runs = [r for r in runs if run_center(r) >= mid]

    # core logic
    if len(runs) == 2 and len(top_runs) == 1 and len(bot_runs) == 1:
        status = "POSITIVE"
        reason = None
    elif len(runs) == 1 and len(top_runs) == 1 and len(bot_runs) == 0:
        status = "NEGATIVE"
        reason = None
    else:
        status = "INVALID"
        reason = "Band pattern not 1-top or 1-top+1-bottom."

    return {
        "status": status,
        "runs": runs,
        "top_runs": top_runs,
        "bottom_runs": bot_runs,
        "mid_row": mid,
        "num_bands": len(runs),
        "reason": reason,
    }
    
def band_mean_intensity_on_original(an, run, half=None, edge_margin_frac=0.01, row_radius=10):
    """
    Compute mean intensity in the ORIGINAL image for rows in `run` (s,e),
    but only using pixels away from left/right edges to avoid edge artifacts.

    We first compute the center row of the run, then average over a fixed
    vertical window: [center - row_radius, center + row_radius].
    
    If the band height (e - s + 1) is smaller than the standard window
    height (2*row_radius + 1), we use the entire band.

    This standardizes the band "height" used for intensity, so different
    mask thicknesses don't change the measurement.

    We use grayscale ORIGINAL (not inverted).
    Lower value = darker/pinker band.
    """
    if an.original_image is None:
        raise ValueError("original_image missing")

    gray = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    H, W = gray.shape
    s, e = run

    # clamp run bounds just in case
    s = max(0, int(s))
    e = min(H - 1, int(e))
    if e <= s:
        return float("nan")
    
    band_height = e - s + 1
    target_height = 2 * row_radius + 1

    # fixed-height window around center
    if band_height <= target_height:
        # Band is already smaller than our standard window → use entire band
        s_win = s
        e_win = e
    else:
        # Use fixed-height window around the band center
        center_row = 0.5 * (s + e)
        c = int(round(center_row))

        s_win = max(0, c - row_radius)
        e_win = min(H - 1, c + row_radius)

        # (optional but safe: keep window within the run as well)
        s_win = max(s_win, s)
        e_win = min(e_win, e)
    
    # horizontal cropping to avoid edges
    edge = int(edge_margin_frac * W)
    x0 = edge
    x1 = W - edge

    roi = gray[s_win:e_win + 1, x0:x1] # includes height from s_fixed to e_fixed and width from left edge to right edge
    if roi.size == 0:
        return float("nan")
    
    return float(np.mean(roi))


def make_band_sampling_mask(
    an,
    run,
    edge_margin_frac=0.01,
    row_radius=10,
):
    """
    Generate a binary mask (0/255) showing exactly which pixels are used
    for computing band_mean_intensity_on_original().

    This applies the same logic:
      - If band height <= window height => use full band
      - Else => use center ± row_radius
      - Also applies edge cropping
    """

    gray = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    s, e = run

    # Clamp bounds
    s = max(0, int(s))
    e = min(H - 1, int(e))
    if e <= s:
        mask = np.zeros((H, W), dtype=np.uint8)
        return mask

    band_height = e - s + 1
    target_height = 2 * row_radius + 1

    # --- choose window vertically ---
    if band_height <= target_height:
        # use entire band
        s_win = s
        e_win = e
    else:
        # fixed window around center
        center_row = 0.5 * (s + e)
        c = int(round(center_row))
        s_win = max(0, c - row_radius)
        e_win = min(H - 1, c + row_radius)

        # restrict within actual band (safe)
        s_win = max(s_win, s)
        e_win = min(e_win, e)

    # --- horizontal cropping ---
    edge = int(edge_margin_frac * W)
    x0 = max(0, edge)
    x1 = min(W, W - edge)

    # --- build mask ---
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[s_win:e_win+1, x0:x1] = 255

    return mask

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


def classify_one_band_top_only(an, edge_margin_rows=5):
    """
    Your requested rule:
      - POSITIVE if EXACTLY 1 band run detected AND it's in the TOP half
      - NEGATIVE otherwise (0 bands or >=2 bands)
      - INVALID if exactly 1 band run but it's in the BOTTOM half, OR if that band touches the top/bottom edge rows.

    Returns dict with status + runs.
    """
    if not hasattr(an, "_rowwise_debug"):
        raise ValueError("Run rowwise_binarize_corrected() first (needs an._rowwise_debug).")

    hits = an._rowwise_debug["row_hits_clean"].astype(bool)
    H = hits.shape[0]
    mid = H // 2

    runs = _runs_from_hits(hits)
    n = len(runs)
    
    def is_edge_run(run):
        s, e = run
        if s <= edge_margin_rows - 1:
            return True
        if e >= H - edge_margin_rows:
            return True
        return False

    # If any run touches edge, mark INVALID right away
    if any(is_edge_run(r) for r in runs):
        return {
            "status": "INVALID",
            "num_bands": n,
            "runs": runs,
            "mid_row": mid,
            "problem_bottom_half_band": False,
            "bad_runs": [r for r in runs if is_edge_run(r)],
        }

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


def auto_crop_remove_bright_edges(
    an,
    bright_delta=30,      # how much brighter than center to consider "too bright"
    center_frac=0.3,      # fraction of width/height used for center region
    tol_frac=0.01,        # fraction of pixels on an edge that must be too bright to crop
    min_size=20           # minimum allowed height/width
):
    """
    Remove bright artifacts touching image edges *after inversion*.

    Logic:
      - Work on an.inverted_image (bright bands, bright artifacts).
      - Compute a "center brightness" from a central window.
      - If any edge row/column has a fraction `tol_frac` of pixels that are
        > center_brightness + bright_delta, trim ONE row/column from that side.
      - Repeat until no edges are too bright or we hit min_size.

    Crops:
      - original_image
      - inverted_image
      - gray_image, corrected_image, background_image, binary_mask
        (if present and matching the same shape)
    """
    if an.inverted_image is None:
        raise ValueError("Run preprocess() first to set an.inverted_image.")

    inv = an.inverted_image
    if inv.ndim != 2:
        raise ValueError("an.inverted_image must be single-channel (grayscale).")

    H, W = inv.shape
    top, bottom = 0, H - 1
    left, right = 0, W - 1

    def current_subimage():
        return inv[top:bottom+1, left:right+1]

    def compute_center_brightness(sub):
        h, w = sub.shape
        ch = max(1, int(center_frac * h))
        cw = max(1, int(center_frac * w))

        r0 = (h - ch) // 2
        c0 = (w - cw) // 2
        r1 = r0 + ch
        c1 = c0 + cw

        center_region = sub[r0:r1, c0:c1]
        return float(np.median(center_region))

    def edge_too_bright(sub, center_val):
        h, w = sub.shape
        thr = center_val + bright_delta

        # Fraction of "too bright" pixels in each edge row/column
        top_frac = np.mean(sub[0, :] > thr)
        bot_frac = np.mean(sub[h-1, :] > thr)
        left_frac = np.mean(sub[:, 0] > thr)
        right_frac = np.mean(sub[:, w-1] > thr)

        return {
            "top": top_frac >= tol_frac,
            "bottom": bot_frac >= tol_frac,
            "left": left_frac >= tol_frac,
            "right": right_frac >= tol_frac,
        }

    changed = True
    while changed:
        changed = False

        # Safety: stop if we're about to go below min size
        h_cur = bottom - top + 1
        w_cur = right - left + 1
        if h_cur <= min_size or w_cur <= min_size:
            break

        sub = current_subimage()
        center_val = compute_center_brightness(sub)
        flags = edge_too_bright(sub, center_val)

        # Try to crop at most one row/col per edge per iteration,
        # respecting min_size.
        if flags["top"] and h_cur > min_size:
            top += 1
            changed = True

        # recompute size after each potential crop
        h_cur = bottom - top + 1
        if flags["bottom"] and h_cur > min_size:
            bottom -= 1
            changed = True

        w_cur = right - left + 1
        if flags["left"] and w_cur > min_size:
            left += 1
            changed = True

        w_cur = right - left + 1
        if flags["right"] and w_cur > min_size:
            right -= 1
            changed = True

    # Final safety: if something went wrong and crop is degenerate, bail out
    if top > bottom or left > right:
        return (0, H - 1, 0, W - 1)

    # Apply crop to inverted and original and any aligned derivatives
    def maybe_crop_attr(name):
        if hasattr(an, name):
            img = getattr(an, name)
            if img is not None and img.size > 0:
                # Only crop if shape matches original inverted shape
                if img.shape[:2] == (H, W):
                    setattr(an, name, img[top:bottom+1, left:right+1])

    maybe_crop_attr("original_image")
    maybe_crop_attr("inverted_image")
    maybe_crop_attr("gray_image")
    maybe_crop_attr("corrected_image")
    maybe_crop_attr("background_image")
    maybe_crop_attr("binary_mask")

    return (top, bottom, left, right)