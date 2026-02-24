"""
Batch LFA analysis: runs all 9 images, collects relative intensity,
and plots intensity vs E. coli concentration.
"""

from lfa import SimpleLFAAnalyzer
from lfa.analysis import run_analysis
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

def four_pl(x, bottom, top, ec50, hill):
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hill)


# ── 1. Sample manifest ────────────────────────────────────────────────────────
SAMPLES = [
    {"file": "LFAIMAGES/original/image1neg_COPY.jpeg",  "conc": 0}, # -> not detected -> CROPPED POORLY
    {"file": "LFAIMAGES/original/50_fold_manual_1.jpeg",  "conc": 6e3}, # -> detected
    {"file": "LFAIMAGES/original/50_fold_manual_2.jpeg",  "conc": 6e3}, # -> detected
    {"file": "LFAIMAGES/original/75_fold_manual_1.jpeg",  "conc": 4e3}, # -> not detected
    {"file": "LFAIMAGES/original/image2-25folds_COPY.jpeg",  "conc": 1.2e4}, # -> detected
    {"file": "LFAIMAGES/original/image3-50fold_COPY.jpeg",  "conc": 6e3}, # -> detected
    {"file": "LFAIMAGES/original/image4-10fold_COPY.jpeg",  "conc": 3e4}, # -> detected
    {"file": "LFAIMAGES/original/image5-50fold2_COPY.jpeg",  "conc": 6e3}, # -> BARELY DETECTED
    {"file": "LFAIMAGES/original/image6-10fold2_COPY.jpeg",  "conc": 3e4}, # -> detected
    {"file": "LFAIMAGES/original/image7-75fold_COPY.jpeg",  "conc": 4e3}, # -> not detected -> DETECTED IF USING MEDIAN INSTEAD OF MEAN
    {"file": "LFAIMAGES/original/image8-25fold2_COPY.jpeg",  "conc": 1.2e4}, # -> detected
    {"file": "LFAIMAGES/original/image9-75fold2_COPY.jpeg",  "conc": 4e3}, # -> DETECTED
    
    
    # # 2-19 SP
    # {"file": "LFAIMAGES/SP 2-19/SP_1e5.png",  "conc": 1e5}, # -> not detected
    # {"file": "LFAIMAGES/SP 2-19/SP_3e5.png",  "conc": 3e5}, # -> detected
    # {"file": "LFAIMAGES/SP 2-19/SP_1e6.png",  "conc": 1e6}, # -> detected
    # {"file": "LFAIMAGES/SP 2-19/SP_3e6.png",  "conc": 3e6}, # -> detected
    # {"file": "LFAIMAGES/SP 2-19/SP_1e7.png",  "conc": 1e7}, # -> detected
    # {"file": "LFAIMAGES/SP 2-19/SP_neg.png",  "conc": 0}, # -> detected...
    
    
    # # 2-18 SP
    # {"file": "LFAIMAGES/SP-2-18/1e5_crop.jpg",  "conc": 1e5}, # -> not detected, rest are...
    # {"file": "LFAIMAGES/SP-2-18/1e6_crop.jpg",  "conc": 1e6},
    # {"file": "LFAIMAGES/SP-2-18/3e6_crop.jpg",  "conc": 3e6},
    # {"file": "LFAIMAGES/SP-2-18/1e7_crop.jpg",  "conc": 1e7},
    # {"file": "LFAIMAGES/SP-2-18/3e7_crop.jpg",  "conc": 3e7},
    # {"file": "LFAIMAGES/SP-2-18/1e8_crop.jpg",  "conc": 1e8},
    # {"file": "LFAIMAGES/SP-2-18/3e8_crop_manual.jpg",  "conc": 3e8},
    
    # # 2-19
    # {"file": "LFAIMAGES/2-19/1e6_crop_auto.JPG",  "conc": 1e6},
    # {"file": "LFAIMAGES/2-19/1e9_crop_auto.JPG",  "conc": 1e9}
    
    
    # # 2-20
    # {"file": "LFAIMAGES/standard 2-20/3e5 + atps 1.JPG",  "conc": 3e5},
    # {"file": "LFAIMAGES/standard 2-20/3e5 + atps 2.JPG",  "conc": 3e5},
    # {"file": "LFAIMAGES/standard 2-20/3e5 + no atps 1.JPG",  "conc": 3e5},
    # {"file": "LFAIMAGES/standard 2-20/3e5 + no atps 2.JPG",  "conc": 3e5},
    # {"file": "LFAIMAGES/standard 2-20/no atps 1 REAL.JPG",  "conc": 3e5},
]

# ── 2. Run analysis on every image ────────────────────────────────────────────
records = []

for s in SAMPLES:
    print(f"\nProcessing: {s['file']}")
    try:
        an = SimpleLFAAnalyzer(s["file"])
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
        
        # --- NEW: generate the big 3-panel debug image for this strip ---
        from lfa.visualization import plot_lfa_debug_panels
        plot_lfa_debug_panels(an, save_dir="lfa_strip_panels")
        # ---------------------------------------------------------------
        
        records.append({
            "file":               s["file"],
            "conc":               s["conc"],
            "status":             results.get("status", "UNKNOWN"),
            "relative_intensity": results.get("relative_intensity", None),
            "num_bands":          results.get("num_bands", None),
        })

        ri = results.get("relative_intensity", None)
        ri_str = f"{ri:.4f}" if ri is not None else "N/A"
        print(f"  → {results.get('status')}  |  rel. intensity = {ri_str}")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        records.append({
            "file":               s["file"],
            "conc":               s["conc"],
            "status":             "ERROR",
            "relative_intensity": None,
            "num_bands":          None,
        })

# ── 3. Separate into plottable and non-plottable ──────────────────────────────
plot_records   = [r for r in records if r["relative_intensity"] is not None]
no_ri_records  = [r for r in records if r["relative_intensity"] is None]

if no_ri_records:
    print("\nSamples with no relative intensity (NEGATIVE / ERROR / INVALID):")
    for r in no_ri_records:
        print(f"  {r['file']:40s}  conc={r['conc']:.2e}  status={r['status']}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

all_concs = np.array([r["conc"] for r in records], dtype=float)
all_ris   = np.array([
    r["relative_intensity"] if r["relative_intensity"] is not None else 0.0
    for r in records
], dtype=float)

ax.scatter(all_concs, all_ris, s=80, color="steelblue", zorder=3, edgecolors="white", linewidths=0.8)




# HIGHLIGHT_FILES = [
#     "LFAIMAGES/50_fold_manual_2.jpeg",
#     "LFAIMAGES/75_fold_manual_1.jpeg",
#     "LFAIMAGES/image2-25folds.jpeg",
#     "LFAIMAGES/image3-50fold.jpeg",
#     "LFAIMAGES/image4-10fold.jpeg",
#     "LFAIMAGES/image5-50fold2.jpeg",
#     "LFAIMAGES/image6-10fold2.jpeg",
#     "LFAIMAGES/image7-75fold.jpeg",
#     "LFAIMAGES/image8-25fold2.jpeg",
#     "LFAIMAGES/image9-75fold2.jpeg",
# ]

# # ── 4. Plot ───────────────────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(8, 5))

# all_concs = np.array([r["conc"] for r in records], dtype=float)
# all_ris   = np.array([
#     r["relative_intensity"] if r["relative_intensity"] is not None else 0.0
#     for r in records
# ], dtype=float)

# # Color only the manually selected files
# point_colors = [
#     "red" if r["file"] in HIGHLIGHT_FILES else "steelblue"
#     for r in records
# ]

# ax.scatter(
#     all_concs,
#     all_ris,
#     s=80,
#     c=point_colors,
#     zorder=3,
#     edgecolors="white",
#     linewidths=0.8,
# )

# # Optional: add a legend
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='red',
#            markeredgecolor='white', markersize=8, label='Highlighted samples'),
#     Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='steelblue',
#            markeredgecolor='white', markersize=8, label='Other samples'),
# ]
# ax.legend(handles=legend_elements, fontsize=10)


# ── Curve fit ────────────────────────────────────────────────────────────────
fit_records = [r for r in records if r["conc"] > 0 and r["relative_intensity"] is not None]
x_fit = np.array([r["conc"] for r in fit_records])
y_fit = np.array([r["relative_intensity"] for r in fit_records])

n_points = len(x_fit)

if n_points < 2:
    print("\n⚠ Not enough points for any curve fit (need at least 2).")
elif n_points == 2:
    print("\n⚠ Only 2 points available; skipping 4PL/3PL curve fit.")
    # Optionally you could fit a simple line in log-space here if you want:
    # def linear(x, m, b): return m * x + b
    # popt, pcov = curve_fit(linear, np.log10(x_fit), y_fit)
else:
    try:
        if n_points < 4:
            # Use a reduced-parameter logistic: fix Hill = 1.0 → 3 free params
            def three_pl(x, bottom, top, EC50):
                return four_pl(x, bottom, top, EC50, 1.0)

            p0_3 = [y_fit.min(), y_fit.max(), np.median(x_fit)]
            popt, pcov = curve_fit(three_pl, x_fit, y_fit, p0=p0_3, maxfev=10000)

            # R² check
            y_pred = three_pl(x_fit, *popt)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            x_curve = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 300)
            y_curve = three_pl(x_curve, *popt)

            ax.plot(x_curve, y_curve, color="tomato", linewidth=2,
                    label=f"3PL fit (Hill=1, R²={r2:.3f})", zorder=2)
            ax.legend(fontsize=10)

            print(f"\n3PL fit parameters (Hill fixed to 1.0):")
            print(f"  Bottom : {popt[0]:.4f}")
            print(f"  Top    : {popt[1]:.4f}")
            print(f"  EC50   : {popt[2]:.3e} CFU/mL")
            print(f"  Hill   : 1.0000 (fixed)")

        else:
            # Full 4-parameter logistic fit
            p0 = [y_fit.min(), y_fit.max(), np.median(x_fit), 1.0]
            popt, pcov = curve_fit(four_pl, x_fit, y_fit, p0=p0, maxfev=10000)

            # R² check
            y_pred = four_pl(x_fit, *popt)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            x_curve = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 300)
            y_curve = four_pl(x_curve, *popt)

            ax.plot(x_curve, y_curve, color="tomato", linewidth=2,
                    label=f"4PL fit (R²={r2:.3f})", zorder=2)
            ax.legend(fontsize=10)

            print(f"\n4PL fit parameters:")
            print(f"  Bottom : {popt[0]:.4f}")
            print(f"  Top    : {popt[1]:.4f}")
            print(f"  EC50   : {popt[2]:.3e} CFU/mL")
            print(f"  Hill   : {popt[3]:.4f}")

    except RuntimeError as e:
        print(f"\n⚠ Curve fit failed: {e}")

# X axis: handle zero separately so log scale still works
ax.set_xscale("symlog", linthresh=1e3)   # symlog shows 0 on a log-like axis
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: "0" if x == 0 else f"{x:.0e}"
))

ax.set_xlabel("E. coli Concentration (CFU/mL)", fontsize=12)
ax.set_ylabel("Relative Intensity (Test / Control)", fontsize=12)
ax.set_title("LFA Signal vs. E. coli Concentration", fontsize=13, fontweight="bold")

ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Plot fitted curve, depending on available points ---
n_points = len(x_fit)

if n_points >= 4:
    # 4PL line
    ax.plot(
        x_curve, y_curve,
        color="tomato", linewidth=2,
        label=f"4PL fit (R²={r2:.3f})",
        zorder=2,
    )
elif n_points == 3:
    # 3PL line (Hill=1)
    ax.plot(
        x_curve, y_curve,
        color="tomato", linewidth=2,
        label=f"3PL fit (Hill=1, R²={r2:.3f})",
        zorder=2,
    )
else:
    # No fit
    ax.text(
        0.5, 0.1,
        "Not enough data points for logistic fit",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, color="gray"
    )

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("lfa_intensity_vs_concentration.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: lfa_intensity_vs_concentration.png")
plt.show()

# ── 5. Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'File':<35} {'Conc (CFU/mL)':>14} {'Status':>10} {'Rel. Int.':>10}")
print("=" * 65)
for r in sorted(records, key=lambda x: x["conc"]):
    ri_str = f"{r['relative_intensity']:.4f}" if r["relative_intensity"] is not None else "N/A"
    print(f"{r['file']:<35} {r['conc']:>14.2e} {r['status']:>10} {ri_str:>10}")
print("=" * 65)
