"""Aggregate per-image segmentation results into mean +/- std per set,
with paired significance tests (Wilcoxon + permutation) and bootstrap CIs
for the with/without Box-Cox comparison.

Reads results_ml.csv (long form) and writes:
  - summary_stats.csv : tidy table (set, method, metric, stats, p-values)
  - results_table.md  : human-readable table per set (percentages)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of src/)
RESULTS_DIR = os.path.join(ROOT, "results")
METRICS = ["IoU", "Dice", "Precision", "Recall", "Accuracy"]
SET_ORDER = ["Vertical", "Horizontal", "Left", "Big"]
METHOD_ORDER = ["SVM", "LightGBM", "KNN", "LDA", "QDA"]
N_BOOT = 10000
SEED = 42


def paired_perm_p(diff, n_perm=10000, rng=None):
    """Two-sided paired permutation test (sign-flip) on the mean difference."""
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return np.nan
    obs = np.mean(diff)
    rng = rng or np.random.default_rng(SEED)
    signs = rng.choice([-1, 1], size=(n_perm, len(diff)))
    perm_means = (signs * diff).mean(axis=1)
    return float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())


def boot_ci(x, n_boot=N_BOOT, rng=None, alpha=0.05):
    """Percentile bootstrap CI for the mean of x (resampling observations)."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def wilcoxon_p(bc, nobc):
    d = bc - nobc
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(bc, nobc, zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results_ml.csv"))
    rng = np.random.default_rng(SEED)
    rows = []
    for s in SET_ORDER:
        for m in METHOD_ORDER:
            sub = df[(df["set"] == s) & (df["method"] == m)]
            wide = sub.pivot_table(index="image", columns="condition", values=METRICS)
            for metric in METRICS:
                nobc = wide[(metric, "noBC")].to_numpy()
                bc = wide[(metric, "BC")].to_numpy()
                diff = bc - nobc
                lo_d, hi_d = boot_ci(diff, rng=rng)
                rows.append(dict(
                    set=s, method=m, metric=metric, n=len(nobc),
                    noBC_mean=np.nanmean(nobc) * 100, noBC_std=np.nanstd(nobc, ddof=1) * 100,
                    BC_mean=np.nanmean(bc) * 100, BC_std=np.nanstd(bc, ddof=1) * 100,
                    diff_mean=np.nanmean(diff) * 100,
                    diff_ci_low=lo_d * 100, diff_ci_high=hi_d * 100,
                    p_wilcoxon=wilcoxon_p(bc, nobc),
                    p_perm=paired_perm_p(diff, rng=rng),
                ))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RESULTS_DIR, "summary_stats.csv"), index=False)
    print("Saved summary_stats.csv")
    write_markdown(summ)


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def write_markdown(summ):
    lines = ["# Segmentación de grietas: Box-Cox vs sin Box-Cox",
             "",
             "Media ± desviación estándar (%) sobre 50 imágenes por set. "
             "`Δ` = media(BC) − media(sin BC) en puntos porcentuales. "
             "Significancia del test pareado de permutación: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001.",
             ""]
    for s in SET_ORDER:
        lines.append(f"## Set: {s}")
        lines.append("")
        header = "| Método | Métrica | Sin Box-Cox | Con Box-Cox | Δ (pp) | IC95% Δ | p (perm) | p (Wilcoxon) |"
        sep = "|---|---|---|---|---|---|---|---|"
        lines += [header, sep]
        for m in METHOD_ORDER:
            for metric in METRICS:
                r = summ[(summ["set"] == s) & (summ["method"] == m) &
                         (summ["metric"] == metric)].iloc[0]
                lines.append(
                    f"| {m} | {metric} | {r.noBC_mean:.1f} ± {r.noBC_std:.1f} | "
                    f"{r.BC_mean:.1f} ± {r.BC_std:.1f} | {r.diff_mean:+.1f}{stars(r.p_perm)} | "
                    f"[{r.diff_ci_low:+.1f}, {r.diff_ci_high:+.1f}] | "
                    f"{r.p_perm:.3f} | {r.p_wilcoxon:.3f} |"
                )
        lines.append("")
    with open(os.path.join(RESULTS_DIR, "results_table.md"), "w") as f:
        f.write("\n".join(lines))
    print("Saved results_table.md")


if __name__ == "__main__":
    main()
