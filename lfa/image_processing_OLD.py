# first step is to pre-process the image that's uploaded
# greyscale and then invert
# invert gray scale through 255 - gray (pink bands are darker than background --> now dark bands become bright peaks)
import numpy as np
import cv2

def preprocess(an):
    """
    Convert image to grayscale and invert it.
    After this, bright pixels = band signal, dark pixels = background.
    """
    an.gray_image = cv2.cvtColor(an.original_image, cv2.COLOR_BGR2GRAY)
    an.inverted_image = 255 - an.gray_image
    return an.inverted_image

# keeping the morph method with an elliptical kernel
# takes image, builds the elliptical kernel, morphological opening, subtracts background
def subtract_background(an, ksize=71, denoise=False):
    """
    Remove slow background variation using morphological opening.
    
    ksize should be comfortably larger than your widest expected band (in pixels).
    A good starting point is 61-81.
    
    denoise: apply a small median blur after subtraction to reduce speckle noise.
    """
    if an.inverted_image is None:
        raise ValueError("Run preprocess() first.")

    img = an.inverted_image.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(img, background)  # clips at 0 automatically for uint8

    if denoise:
        corrected = cv2.medianBlur(corrected, 3)

    an.background_image = background
    an.corrected_image = corrected.astype(np.uint8)
    return an.corrected_image

# takes the 2D corrected image and collapses to 1D signal using rowwise method
# row score for intensity, smooth version with median filter, residual, sigma, threshold, peak row

def detect_band_rows(img, stat="median", smooth_ksize=51, k=4.0):
    """
    Build a per-row adaptive threshold from row statistics.

    img         : corrected image (H x W float32)
    stat        : "median" is more robust to column artifacts than "mean"
    smooth_ksize: window for baseline median filter — should be larger than
                  your tallest band. 51 is a good default for ~40px bands.
    k           : how many sigma above baseline a row must be to count as a band.
                  Higher = stricter. 4.0 is a good starting point.

    Returns
    -------
    row_score : (H,) float  — raw per-row intensity
    baseline  : (H,) float  — smoothed background estimate
    T_row     : (H,) float  — adaptive threshold per row
    row_hits  : (H,) bool   — True where row_score exceeds threshold
    peak_row  : int          — index of single brightest row
    sigma     : float        — noise floor estimate
    """
    img = img.astype(np.float32)

    if stat == "median":
        row_score = np.median(img, axis=1)
    elif stat == "mean":
        row_score = np.mean(img, axis=1)
    else:
        raise ValueError("stat must be 'mean' or 'median'")

    # smooth row_score heavily to get background drift
    baseline = _median_filter_1d(row_score, smooth_ksize)

    # residual = signal above background
    resid = row_score - baseline

    # robust noise estimate using MAD
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-6
    sigma = 1.4826 * mad

    # per-row threshold
    T_row = baseline + float(k) * float(sigma)
    row_hits = row_score > T_row

    peak_row = int(np.argmax(row_score))

    return row_score, baseline, T_row, row_hits, peak_row, float(sigma)


# to reduce edge artifacts
def _median_filter_1d(x, ksize):
    """
    1D median filter with reflect padding.

    x     : (H,) float array — the row score signal
    ksize : int — window size, should be larger than your tallest band.
                  Odd numbers work best. 51 is a good default.

    Reflect padding means edges are mirrored rather than zero-padded,
    which avoids pulling the baseline down at the top/bottom of the strip.
    """
    k = int(ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1  # enforce odd

    pad = k // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")

    w = np.lib.stride_tricks.sliding_window_view(xp, window_shape=k)
    return np.median(w, axis=1)

# detect band rows called, finds peak rows in bottom and top half 

def rowwise_binarize_corrected(
    an,
    stat="median",
    smooth_ksize=51,
    k=4.0,
    min_run=3,
    expand=2,
    band_score_mode="auc",
):
    """
    Detect band rows and build a binary mask.

    stat         : row statistic, "median" recommended
    smooth_ksize : baseline filter window, should exceed tallest band in pixels
    k            : sigma multiplier for threshold strictness
    min_run      : minimum consecutive rows to count as a band (removes noise)
    expand       : expand detected bands by +/- this many rows (catches edges)
    band_score_mode : how to score competing runs — "auc" sums all signal in
                      the run (robust), "peak" uses only the brightest row
                      (more sensitive but less stable)
    """
    if an.corrected_image is None:
        raise ValueError("Run preprocess() and subtract_background() first.")

    img = an.corrected_image.astype(np.float32)

    row_score, baseline, T_row, row_hits, peak_row, sigma = detect_band_rows(
        img,
        stat=stat,
        smooth_ksize=smooth_ksize,
        k=k,
    )

    resid = row_score - baseline
    hits = row_hits.copy()

    # remove runs too short to be a real band
    if min_run and min_run > 1:
        hits = _keep_true_runs(hits, min_run=min_run)

    # keep strongest band in top half, strongest in bottom half
    H = len(hits)
    mid = H // 2
    hits = keep_best_run_per_half(hits, resid, mid, score_mode=band_score_mode)

    # thicken bands to catch edges that just missed threshold
    if expand and expand > 0:
        hits = _expand_rows(hits, radius=expand)

    # build 2D mask — entire row is ON or OFF
    mask = (hits[:, None].astype(np.uint8) * 255)
    mask = np.repeat(mask, an.corrected_image.shape[1], axis=1)

    an.binary_mask = mask
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
        "band_score_mode": band_score_mode,
    }

    return T_row, mask

def _keep_true_runs(hits, min_run=3):
    """
    Keep only contiguous True runs of length >= min_run.
    Anything shorter is treated as noise and set to False.
    """
    hits = hits.astype(bool)
    out = np.zeros_like(hits, dtype=bool)
    i = 0
    while i < len(hits):
        if not hits[i]:
            i += 1
            continue
        j = i
        while j < len(hits) and hits[j]:
            j += 1
        if (j - i) >= min_run:
            out[i:j] = True
        i = j
    return out
def _expand_rows(hits, radius=2):
    """
    Expand True rows by +/- radius using 1D convolution.
    radius=2 means each band grows by 2 rows on each side.
    """
    hits = hits.astype(np.uint8)
    k = 2 * int(radius) + 1
    kernel = np.ones(k, dtype=np.uint8)
    expanded = np.convolve(hits, kernel, mode="same") > 0
    return expanded

def keep_best_run_per_half(hits, resid, mid, score_mode="auc"):
    """
    Keep only the strongest band run in each half of the strip.

    hits      : (H,) bool
    resid     : (H,) float — row_score minus baseline
    mid       : int — row index dividing top and bottom half
    score_mode: "auc" sums all positive residual in the run (recommended)
                "peak" uses only the single brightest row in the run
    """
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

    top_runs = [(s, e) for (s, e) in runs if 0.5 * (s + e) < mid]
    bot_runs = [(s, e) for (s, e) in runs if 0.5 * (s + e) >= mid]

    kept = np.zeros_like(hits, dtype=bool)

    if top_runs:
        best = max(top_runs, key=lambda r: score_run(*r))
        kept[best[0]:best[1]+1] = True

    if bot_runs:
        best = max(bot_runs, key=lambda r: score_run(*r))
        kept[best[0]:best[1]+1] = True

    return kept

def _runs_from_hits(hits):
    """
    Convert boolean hits (H,) into list of (start, end) inclusive index
    pairs for each contiguous True run.
    """
    hits = hits.astype(bool)
    runs = []
    i = 0
    while i < len(hits):
        if not hits[i]:
            i += 1
            continue
        j = i
        while j < len(hits) and hits[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs

def classify_two_band_top_bottom(an):
    """
    Classify LFA result based on detected band runs.

    POSITIVE : 2 bands, one in each half (control + test line present)
    NEGATIVE : 1 band in top half only (control line only)
    INVALID  : anything else

    Requires rowwise_binarize_corrected() to have been run first.
    """
    if not hasattr(an, "_rowwise_debug"):
        raise ValueError("Run rowwise_binarize_corrected() first.")

    hits = an._rowwise_debug["row_hits_clean"].astype(bool)
    H = len(hits)
    mid = H // 2

    runs = _runs_from_hits(hits)

    def run_center(s, e):
        return 0.5 * (s + e)

    top_runs = [(s, e) for (s, e) in runs if run_center(s, e) < mid]
    bot_runs = [(s, e) for (s, e) in runs if run_center(s, e) >= mid]

    if len(top_runs) == 1 and len(bot_runs) == 1:
        status = "POSITIVE"
    elif len(top_runs) == 1 and len(bot_runs) == 0:
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