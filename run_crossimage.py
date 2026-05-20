"""Protocol 2: cross-image generalization for traditional ML segmentation.

For each set, 5-fold CV over the 50 images: train each classifier on a stratified
pixel sample drawn from the TRAINING images, predict each held-out TEST image fully.
Each image is a test image exactly once -> 50 per-image metrics per (set,method,cond).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
import lightgbm as lgb
import warnings

from boxcox_seg import load_pair, make_feature

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
RGB_DIR = os.path.join(ROOT, "rgb")
SETS = {"Vertical": "Vertical", "Horizontal": "Horizontal", "Left": "Left", "Big": "Big"}
METRICS = ["IoU", "Dice", "Precision", "Recall", "Accuracy"]
SEED = 42
N_SPLITS = 5
CAP_PER_CLASS_PER_IMG = 100  # pixels sampled per class per training image


def build_models():
    return {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=500),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, num_leaves=31, verbose=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.01),
    }


def metrics(y_true, y_pred):
    yt, yp = y_true.astype(bool), y_pred.astype(bool)
    tp = np.sum(yt & yp); fp = np.sum(~yt & yp); fn = np.sum(yt & ~yp); tn = np.sum(~yt & ~yp)
    union = tp + fp + fn
    return dict(
        IoU=tp / union if union > 0 else np.nan,
        Dice=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan,
        Precision=tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        Recall=tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        Accuracy=(tp + tn) / (tp + tn + fp + fn),
    )


def sample_train_pixels(files, folder, use_bc, rng):
    Xs, ys = [], []
    for f in files:
        rgb, mask = load_pair(os.path.join(RGB_DIR, f), os.path.join(ROOT, folder, f))
        y = mask.reshape(-1)
        feat, _ = make_feature(rgb, use_bc)
        x = feat.reshape(-1)
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        t0 = rng.choice(i0, min(CAP_PER_CLASS_PER_IMG, len(i0)), replace=False)
        t1 = rng.choice(i1, min(CAP_PER_CLASS_PER_IMG, len(i1)), replace=False)
        idx = np.concatenate([t0, t1])
        Xs.append(x[idx]); ys.append(y[idx])
    return np.concatenate(Xs).reshape(-1, 1), np.concatenate(ys)


def run(quick=False):
    rng = np.random.default_rng(SEED)
    rows = []
    for set_name, folder in SETS.items():
        files = sorted(f for f in os.listdir(os.path.join(ROOT, folder))
                       if f.lower().endswith(".jpg") and os.path.exists(os.path.join(RGB_DIR, f)))
        if quick:
            files = files[:10]
        files = np.array(files)
        kf = KFold(n_splits=2 if quick else N_SPLITS, shuffle=True, random_state=SEED)
        for fold, (tr_i, te_i) in enumerate(kf.split(files)):
            tr_files, te_files = files[tr_i], files[te_i]
            # cache transformed test images per condition
            for cond in ("noBC", "BC"):
                use_bc = cond == "BC"
                Xtr, ytr = sample_train_pixels(tr_files, folder, use_bc, rng)
                test_cache = {}
                for f in te_files:
                    rgb, mask = load_pair(os.path.join(RGB_DIR, f), os.path.join(ROOT, folder, f))
                    feat, lam = make_feature(rgb, use_bc)
                    test_cache[f] = (feat.reshape(-1, 1), mask.reshape(-1), lam)
                for mname, model in build_models().items():
                    try:
                        model.fit(Xtr, ytr)
                    except Exception:
                        continue
                    for f, (Xte, yte, lam) in test_cache.items():
                        yp = model.predict(Xte)
                        rows.append(dict(set=set_name, image=f, method=mname,
                                         condition=cond, lam=lam, fold=fold,
                                         **metrics(yte, yp)))
            print(f"[{set_name}] fold {fold+1} done", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results_ml_crossimg_quick.csv" if quick else "results_ml_crossimg.csv")
    df.to_csv(out, index=False)
    print("Saved", out, "rows:", len(df))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)
