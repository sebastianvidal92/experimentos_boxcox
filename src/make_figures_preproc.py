# -*- coding: utf-8 -*-
"""Genera figuras de ejemplo (report/figures/preproc_*.png) comparando los cuatro
preprocesamientos: por cada tipo de grieta, panel
RGB | máscara | Raw | Box-Cox | HistEq | CLAHE (segmentación QDA)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from boxcox_seg import load_pair, make_feature_pp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RGB_DIR = os.path.join(ROOT, "data", "rgb")
MASK_DIR = os.path.join(ROOT, "data", "masks")
FIG_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TRAIN_CAP = 6000
SEED = 42

CONDITIONS = [("raw", "Raw"), ("boxcox", "Box-Cox"), ("histeq", "HistEq"), ("clahe", "CLAHE")]
# (set, fname, tag, metric_mode): tag = output filename preproc_<tag>.png; metric_mode
# "rp" muestra Recall/Precision (figuras del compromiso), "iou_dice" muestra IoU/Dice
# (calidad global, p.ej. el caso donde Box-Cox supera a CLAHE).
EXAMPLES = [
    ("Vertical", "595.jpg", "Vertical", "rp"),
    ("Horizontal", "500.jpg", "Horizontal", "rp"),
    ("Left", "011.jpg", "Left", "rp"),
    ("Big", "453.jpg", "Big", "rp"),
    ("Big", "253.jpg", "BoxCoxWin", "iou_dice"),
]
LABELS = {"Vertical": "Vertical", "Horizontal": "Horizontal",
          "Left": "Oblicua", "Big": "Grande"}


def all_metrics(y, p):
    yt, yp = y.astype(bool), p.astype(bool)
    tp = np.sum(yt & yp); fp = np.sum(~yt & yp); fn = np.sum(yt & ~yp)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return dict(Recall=rec, Prec=prec, IoU=iou, Dice=dice)


def segment(rgb, mask, method, seed):
    rng = np.random.default_rng(seed)
    y = mask.reshape(-1)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = TRAIN_CAP // 2
    idx = np.concatenate([rng.choice(i0, min(n, len(i0)), replace=False),
                          rng.choice(i1, min(n, len(i1)), replace=False)])
    feat, _ = make_feature_pp(rgb, method=method)
    X = feat.reshape(-1, 1)
    m = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X[idx], y[idx])
    pred = m.predict(X).reshape(mask.shape)
    return pred, all_metrics(y, pred.reshape(-1))


def panel(set_name, fname, tag, metric_mode="rp"):
    rgb, mask = load_pair(os.path.join(RGB_DIR, fname),
                          os.path.join(MASK_DIR, set_name, fname))
    seed = SEED + abs(hash(fname)) % 1000
    keys = ("Recall", "Prec") if metric_mode == "rp" else ("IoU", "Dice")
    n = 2 + len(CONDITIONS)
    fig, ax = plt.subplots(1, n, figsize=(2.45 * n, 3.5))
    ax[0].imshow(rgb); ax[0].set_title("Imagen RGB")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Máscara real")
    for k, (method, label) in enumerate(CONDITIONS):
        pred, mt = segment(rgb, mask, method, seed)
        a = ax[2 + k]
        a.imshow(pred, cmap="gray")
        a.set_title(f"{label}\n{keys[0]}={mt[keys[0]]:.2f}  {keys[1]}={mt[keys[1]]:.2f}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Grieta {LABELS[set_name].lower()} ({set_name}/{fname}, QDA)",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"preproc_{tag}.png")
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"{set_name}/{fname} -> {os.path.basename(out)}")
    return os.path.basename(out)


if __name__ == "__main__":
    paths = [panel(*ex) for ex in EXAMPLES]
    print("Figuras:", paths)
