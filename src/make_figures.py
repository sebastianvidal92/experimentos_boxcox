# -*- coding: utf-8 -*-
"""Genera figuras de ejemplo (report/figures/) mostrando imágenes donde Box-Cox
aumenta más el recall y la precision: panel RGB | máscara | sin BC | con BC."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from boxcox_seg import load_pair, make_feature

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RGB_DIR = os.path.join(ROOT, "data", "rgb")
MASK_DIR = os.path.join(ROOT, "data", "masks")
FIG_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TRAIN_CAP = 6000
SEED = 42

# Un ejemplo por tipo de grieta (clasificador QDA, donde el efecto es más claro)
EXAMPLES = [
    ("Vertical", "595.jpg"),
    ("Horizontal", "500.jpg"),
    ("Left", "011.jpg"),     # caso de rescate: sin BC no detecta nada
    ("Big", "249.jpg"),
]
LABELS = {"Vertical": "Vertical", "Horizontal": "Horizontal",
          "Left": "Oblicua", "Big": "Grande"}


def metrics(y, p):
    yt, yp = y.astype(bool), p.astype(bool)
    tp = np.sum(yt & yp); fp = np.sum(~yt & yp); fn = np.sum(yt & ~yp)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return rec, prec


def segment(rgb, mask, use_bc, seed):
    rng = np.random.default_rng(seed)
    y = mask.reshape(-1)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = TRAIN_CAP // 2
    idx = np.concatenate([rng.choice(i0, min(n, len(i0)), replace=False),
                          rng.choice(i1, min(n, len(i1)), replace=False)])
    feat, _ = make_feature(rgb, use_bc)
    X = feat.reshape(-1, 1)
    m = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X[idx], y[idx])
    pred = m.predict(X).reshape(mask.shape)
    rec, prec = metrics(y, pred.reshape(-1))
    return pred, rec, prec


def panel(set_name, fname):
    rgb, mask = load_pair(os.path.join(RGB_DIR, fname),
                          os.path.join(MASK_DIR, set_name, fname))
    seed = SEED + abs(hash(fname)) % 1000
    p0, r0, pr0 = segment(rgb, mask, False, seed)
    p1, r1, pr1 = segment(rgb, mask, True, seed)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    ax[0].imshow(rgb); ax[0].set_title("Imagen RGB")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Máscara real")
    ax[2].imshow(p0, cmap="gray")
    ax[2].set_title(f"Sin Box-Cox\nRecall={r0:.2f}  Precision={pr0:.2f}")
    ax[3].imshow(p1, cmap="gray")
    ax[3].set_title(f"Con Box-Cox\nRecall={r1:.2f}  Precision={pr1:.2f}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Grieta {LABELS[set_name].lower()} ({set_name}/{fname}, QDA)   "
                 f"$\\Delta$Recall={r1-r0:+.2f},  $\\Delta$Precision={pr1-pr0:+.2f}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"ej_{set_name}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"{set_name}/{fname}  R {r0:.2f}->{r1:.2f}  P {pr0:.2f}->{pr1:.2f}  -> {os.path.basename(out)}")
    return os.path.basename(out)


if __name__ == "__main__":
    paths = [panel(s, f) for s, f in EXAMPLES]
    print("Figuras:", paths)
