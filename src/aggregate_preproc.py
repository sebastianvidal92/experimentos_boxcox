"""Aggregate the preprocessing comparison (raw vs Box-Cox vs HistEq vs CLAHE) for
LDA and QDA into mean +/- std per set, with paired significance tests (permutation
+ Wilcoxon) and bootstrap CIs for each transform measured against the raw baseline.

Reads results_preproc.csv (long form) and writes:
  - summary_preproc.csv : tidy table (set, method, metric, transform, stats, p-values)
  - preproc_table.md    : human-readable tables per set and classifier (percentages)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of src/)
RESULTS_DIR = os.path.join(ROOT, "results")
METRICS = ["IoU", "Dice", "Precision", "Recall"]  # Accuracy omitted (misleading, ~1% cracks)
SET_ORDER = ["Vertical", "Horizontal", "Left", "Big"]
METHOD_ORDER = ["LDA", "QDA"]
BASELINE = "raw"
TRANSFORMS = ["boxcox", "histeq", "clahe"]  # compared against BASELINE
COND_ORDER = [BASELINE] + TRANSFORMS
COND_LABEL = {"raw": "Raw", "boxcox": "Box-Cox", "histeq": "HistEq", "clahe": "CLAHE"}
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


def wilcoxon_p(tf, ref):
    d = tf - ref
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(tf, ref, zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


# Comparaciones directas entre transformaciones (a vs b), pareadas por imagen.
# diff = media(a) - media(b); positivo => 'a' mejor (todas las métricas: mayor=mejor).
PAIRS = [("boxcox", "histeq"), ("boxcox", "clahe")]


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results_preproc.csv"))
    rng = np.random.default_rng(SEED)
    rows = []
    pair_rows = []
    for s in SET_ORDER:
        for m in METHOD_ORDER:
            sub = df[(df["set"] == s) & (df["method"] == m)]
            wide = sub.pivot_table(index="image", columns="condition", values=METRICS)
            for metric in METRICS:
                ref = wide[(metric, BASELINE)].to_numpy()
                for tf in TRANSFORMS:
                    cur = wide[(metric, tf)].to_numpy()
                    diff = cur - ref
                    lo_d, hi_d = boot_ci(diff, rng=rng)
                    rows.append(dict(
                        set=s, method=m, metric=metric, transform=tf, n=len(ref),
                        raw_mean=np.nanmean(ref) * 100, raw_std=np.nanstd(ref, ddof=1) * 100,
                        tf_mean=np.nanmean(cur) * 100, tf_std=np.nanstd(cur, ddof=1) * 100,
                        diff_mean=np.nanmean(diff) * 100,
                        diff_ci_low=lo_d * 100, diff_ci_high=hi_d * 100,
                        p_perm=paired_perm_p(diff, rng=rng),
                        p_wilcoxon=wilcoxon_p(cur, ref),
                    ))
                for a, b in PAIRS:
                    va = wide[(metric, a)].to_numpy()
                    vb = wide[(metric, b)].to_numpy()
                    diff = va - vb
                    lo_d, hi_d = boot_ci(diff, rng=rng)
                    pair_rows.append(dict(
                        set=s, method=m, metric=metric, pair=f"{a}_vs_{b}",
                        a=a, b=b, n=len(va),
                        a_mean=np.nanmean(va) * 100, a_std=np.nanstd(va, ddof=1) * 100,
                        b_mean=np.nanmean(vb) * 100, b_std=np.nanstd(vb, ddof=1) * 100,
                        diff_mean=np.nanmean(diff) * 100,
                        diff_ci_low=lo_d * 100, diff_ci_high=hi_d * 100,
                        p_perm=paired_perm_p(diff, rng=rng),
                        p_wilcoxon=wilcoxon_p(va, vb),
                    ))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RESULTS_DIR, "summary_preproc.csv"), index=False)
    print("Saved summary_preproc.csv")
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(os.path.join(RESULTS_DIR, "pairwise_preproc.csv"), index=False)
    print("Saved pairwise_preproc.csv")
    write_markdown(summ, pairs)


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


def cond_means(summ, s, m, metric):
    """Dict condition -> (mean, std) for one (set, method, metric)."""
    out = {}
    block = summ[(summ["set"] == s) & (summ["method"] == m) & (summ["metric"] == metric)]
    out[BASELINE] = (block.iloc[0]["raw_mean"], block.iloc[0]["raw_std"])
    for _, r in block.iterrows():
        out[r["transform"]] = (r["tf_mean"], r["tf_std"])
    return out


PAIR_LABEL = {"boxcox_vs_histeq": "Box-Cox vs HistEq",
              "boxcox_vs_clahe": "Box-Cox vs CLAHE"}


def pairwise_markdown(pairs):
    """Tablas de comparación directa entre transformaciones (Δ = a − b, pp)."""
    lines = ["## Comparación directa entre transformaciones", "",
             "`Δ` = media(A) − media(B) en puntos porcentuales, **pareada por imagen** "
             "(positivo ⇒ A mejor). Significancia del test pareado de permutación: "
             "\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001. Las medias absolutas de cada "
             "preprocesamiento están en las tablas por set de arriba.", ""]
    for pair in pairs["pair"].unique():
        a_lab, b_lab = PAIR_LABEL[pair].split(" vs ")
        lines.append(f"### {PAIR_LABEL[pair]}  (Δ = {a_lab} − {b_lab})")
        lines.append("")
        lines += ["| Set | Clf | " + " | ".join(METRICS) + " |",
                  "|---|---|" + "|".join(["---"] * len(METRICS)) + "|"]
        for s in SET_ORDER:
            for m in METHOD_ORDER:
                cells = [s, m]
                for metric in METRICS:
                    r = pairs[(pairs["pair"] == pair) & (pairs["set"] == s) &
                              (pairs["method"] == m) & (pairs["metric"] == metric)].iloc[0]
                    cells.append(f"{r.diff_mean:+.1f}{stars(r.p_perm)}")
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def write_markdown(summ, pairs=None):
    lines = [
        "# Segmentación de grietas: comparación de preprocesamientos (LDA y QDA)",
        "",
        "Media ± desviación estándar (%) sobre 50 imágenes por set, para cada "
        "preprocesamiento: **Raw** (intensidad gris cruda, baseline), **Box-Cox**, "
        "**HistEq** (ecualización de histograma) y **CLAHE**. Entre paréntesis, "
        "`Δ` = media(transformación) − media(Raw) en puntos porcentuales. "
        "Significancia del test pareado de permutación frente a Raw: "
        "\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001. En **negrita**, el mejor "
        "valor de cada fila. Accuracy se omite (engañosa: la grieta es ~1 % de los píxeles).",
        "",
    ]
    win = {tf: 0 for tf in COND_ORDER}  # how often each condition is the row winner
    for s in SET_ORDER:
        lines.append(f"## Set: {s}")
        lines.append("")
        for m in METHOD_ORDER:
            lines.append(f"### {m}")
            lines.append("")
            header = "| Métrica | Raw | Box-Cox | HistEq | CLAHE |"
            lines += [header, "|---|---|---|---|---|"]
            for metric in METRICS:
                means = cond_means(summ, s, m, metric)
                best = max(COND_ORDER, key=lambda c: means[c][0])
                win[best] += 1
                block = summ[(summ["set"] == s) & (summ["method"] == m) &
                             (summ["metric"] == metric)].set_index("transform")
                cells = [metric]
                for c in COND_ORDER:
                    mean, std = means[c]
                    if c == BASELINE:
                        txt = f"{mean:.1f} ± {std:.1f}"
                    else:
                        r = block.loc[c]
                        txt = f"{mean:.1f} ± {std:.1f} ({r.diff_mean:+.1f}{stars(r.p_perm)})"
                    cells.append(f"**{txt}**" if c == best else txt)
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    lines.append("## Resumen: nº de veces que cada preprocesamiento es el mejor")
    lines.append("")
    lines.append("Conteo sobre " + str(len(SET_ORDER) * len(METHOD_ORDER) * len(METRICS)) +
                 " filas (4 sets × 2 clasificadores × 4 métricas).")
    lines.append("")
    lines += ["| Preprocesamiento | Veces #1 |", "|---|---|"]
    for c in COND_ORDER:
        lines.append(f"| {COND_LABEL[c]} | {win[c]} |")
    lines.append("")
    if pairs is not None:
        lines += pairwise_markdown(pairs)
    with open(os.path.join(RESULTS_DIR, "preproc_table.md"), "w") as f:
        f.write("\n".join(lines))
    print("Saved preproc_table.md")


if __name__ == "__main__":
    main()
