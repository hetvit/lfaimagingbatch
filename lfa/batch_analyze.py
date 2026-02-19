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

def four_pl(x, bottom, top, ec50, hill):
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hill)

# ── 1. Sample manifest ────────────────────────────────────────────────────────
SAMPLES = [
    {"file": "LFAIMAGES/SP_1e5.png",  "conc": 1e5},
    {"file": "LFAIMAGES/SP_1e6.png",  "conc": 1e6},
    {"file": "LFAIMAGES/SP_1e7.png",   "conc": 1e7},
    {"file": "LFAIMAGES/SP_3e5.png",   "conc": 3e5},
    {"file": "LFAIMAGES/SP_3e6.png",  "conc": 3e6},
    {"file": "LFAIMAGES/SP_neg.png",  "conc": 0},
]

# ── 2. Run analysis on every image ────────────────────────────────────────────
records = []

for s in SAMPLES:
    print(f"\nProcessing: {s['file']}")
    try:
        an = SimpleLFAAnalyzer(s["file"])
        results = run_analysis(
            an,
            bg="morph",
            k=51,
            normalize=False,
            denoise=False,
            binarize_mode="rowwise",
            debug_plots=False,       # keep batch run quiet
        )

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

# ── Curve fit ────────────────────────────────────────────────────────────────
fit_records = [r for r in records if r["conc"] > 0 and r["relative_intensity"] is not None]
x_fit = np.array([r["conc"] for r in fit_records])
y_fit = np.array([r["relative_intensity"] for r in fit_records])

try:
    p0 = [y_fit.min(), y_fit.max(), np.median(x_fit), 1.0]
    popt, pcov = curve_fit(four_pl, x_fit, y_fit, p0=p0, maxfev=10000)
    popt, pcov = curve_fit(four_pl, x_fit, y_fit, p0=p0, maxfev=10000)

    # R² check
    y_pred = four_pl(x_fit, *popt)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    x_curve = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 300)
    y_curve = four_pl(x_curve, *popt)

    ax.plot(x_curve, y_curve, color="tomato", linewidth=2, label="4PL fit", zorder=2)
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
ax.plot(x_curve, y_curve, color="tomato", linewidth=2, label=f"4PL fit (R²={r2:.3f})", zorder=2)

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
