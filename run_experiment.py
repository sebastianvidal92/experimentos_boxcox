"""Run per-image ML segmentation over the 4 crack sets, with and without Box-Cox.

Outputs a long-form CSV: one row per (set, image, method, condition) with lambda
and segmentation metrics (IoU, Dice, Precision, Recall, Accuracy).
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import lightgbm as lgb

from boxcox_seg import load_pair, make_feature

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "Vertical": "Vertical",
    "Horizontal": "Horizontal",
    "Left": "Left",
    "Big": "Big",
}
RGB_DIR = os.path.join(ROOT, "rgb")
SIZE = 256
TRAIN_CAP = 6000  # max training pixels (stratified) per image, keeps SVM/KNN fast
SEED = 42


def build_models():
    return {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=500),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, num_leaves=31, verbose=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.01),
    }


def metrics(y_true, y_pred):
    """Segmentation metrics for the crack (positive=1) class."""
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = np.sum(yt & yp)
    fp = np.sum(~yt & yp)
    fn = np.sum(yt & ~yp)
    tn = np.sum(~yt & ~yp)
    union = tp + fp + fn
    iou = tp / union if union > 0 else np.nan
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(IoU=iou, Dice=dice, Precision=prec, Recall=rec, Accuracy=acc)


def stratified_train_idx(y, cap, rng):
    """Indices for a stratified training subsample of up to `cap` pixels."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n_each = cap // 2
    take0 = rng.choice(idx0, min(n_each, len(idx0)), replace=False)
    take1 = rng.choice(idx1, min(n_each, len(idx1)), replace=False)
    return np.concatenate([take0, take1])


def run(quick=False):
    rng = np.random.default_rng(SEED)
    rows = []
    models = build_models()
    for set_name, folder in SETS.items():
        mask_dir = os.path.join(ROOT, folder)
        files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".jpg"))
        if quick:
            files = files[:3]
        for fi, fname in enumerate(files):
            mask_path = os.path.join(mask_dir, fname)
            rgb_path = os.path.join(RGB_DIR, fname)
            if not os.path.exists(rgb_path):
                print(f"  [skip] no rgb for {set_name}/{fname}", flush=True)
                continue
            rgb, mask = load_pair(rgb_path, mask_path, size=SIZE)
            y = mask.reshape(-1)
            if y.sum() == 0:
                print(f"  [skip] empty mask {set_name}/{fname}", flush=True)
                continue
            tr_idx = stratified_train_idx(y, TRAIN_CAP, rng)
            for cond in ("noBC", "BC"):
                feat, lam = make_feature(rgb, use_boxcox=(cond == "BC"))
                X = feat.reshape(-1, 1)
                Xtr, ytr = X[tr_idx], y[tr_idx]
                for mname, model in models.items():
                    t0 = time.time()
                    try:
                        model.fit(Xtr, ytr)
                        yp = model.predict(X)
                        m = metrics(y, yp)
                    except Exception as e:
                        print(f"  [err] {set_name}/{fname} {cond} {mname}: {e}", flush=True)
                        m = dict(IoU=np.nan, Dice=np.nan, Precision=np.nan,
                                 Recall=np.nan, Accuracy=np.nan)
                    rows.append(dict(
                        set=set_name, image=fname, method=mname, condition=cond,
                        lam=lam, time_s=round(time.time() - t0, 3), **m,
                    ))
                # rebuild models so each fit is fresh
                models = build_models()
            print(f"[{set_name}] {fi+1}/{len(files)} {fname} done", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results_ml_quick.csv" if quick else "results_ml.csv")
    df.to_csv(out, index=False)
    print("Saved", out, "rows:", len(df))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 images/set for a smoke test")
    args = ap.parse_args()
    run(quick=args.quick)
