# feature_maps_no_cluster.py
# Standalone script to generate FEATURE MAPS (49x49 feature-feature correlations)
# WITHOUT clustering / reordering.

from data_import import load_all_data
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, rankdata
from gpu_utils import require_accelerators, get_array_backend, to_numpy, log_runtime_gpu_status

# =========================
# CONFIG
# =========================
_CODE_DIR = Path(__file__).resolve().parent
_ROOT = _CODE_DIR.parent
OUT_MAP_DIR = _ROOT / "outputs" / "feature_maps"
OUT_MAP_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MAT_PATH = str(_ROOT / "data" / "training" / "train_data.mat")
TRAIN_IDS_PATH = str(_ROOT / "data" / "training" / "ids_fixed.mat")
TRAIN_XLSX_PATH = str(_ROOT / "data" / "training" / "data_train.xlsx")
TEST_DIR = str(_ROOT / "data" / "testing")

FEATURE_NAMES = [f"F{i+1:02d}" for i in range(48)]

FEATURE_FAMILY_MAP = {
    "pitch":         list(range(0, 10)),
    "formants":      list(range(10, 20)),
    "jitter":        [20, 21],
    "voicing":       [22, 23],
    "energy":        list(range(24, 32)),
    "zero_crossing": list(range(32, 38)),
    "spsl":          list(range(38, 46)),
    "duration":      [46, 47],
}

GENDER_IDX = 1
MODULE_IDX = 10
REDUCER = "mean"
DPI = 200
require_accelerators(require_gpu=False, require_npu=False)
XP, USING_GPU = get_array_backend()
log_runtime_gpu_status("feature_correlation.py")

# =========================
# HELPERS
# =========================
def is_male(a): return (a == "m") | (a == "M")
def is_female(a): return (a == "f") | (a == "F")

def module_is_4(arr):
    out = np.zeros(len(arr), bool)
    for i,v in enumerate(arr):
        try:
            out[i] = float(v) == 4.0
        except:
            out[i] = str(v).strip() in ("4","4.0")
    return out

def apply_module4_filter(X,y,flag):
    if not flag: return X,y
    keep = ~module_is_4(y[:,MODULE_IDX])
    return X[keep], y[keep]

def flatten_to_NWF(X):
    return X.reshape(X.shape[0], -1, X.shape[-1])

def collapse_subject(X):
    # X: (N, W, F) — drop samples where any feature is NaN, then reduce
    result = np.full((X.shape[0], X.shape[2]), np.nan)
    for i in range(X.shape[0]):
        s = X[i]  # (W, F)
        keep = np.isfinite(s).all(axis=1)
        s = s[keep]
        if s.shape[0] == 0:
            continue
        if USING_GPU:
            s_gpu = XP.asarray(s)
            if REDUCER == "mean":
                result[i] = to_numpy(XP.mean(s_gpu, axis=0))
            elif REDUCER == "median":
                result[i] = to_numpy(XP.median(s_gpu, axis=0))
        else:
            if REDUCER == "mean":
                result[i] = np.mean(s, axis=0)
            elif REDUCER == "median":
                result[i] = np.median(s, axis=0)
    return result

def corr_pair(x,y,method):
    m = np.isfinite(x)&np.isfinite(y)
    x=x[m]; y=y[m]
    if x.size<3 or np.all(x==x[0]) or np.all(y==y[0]):
        return np.nan,np.nan
    if method=="pearson":
        return pearsonr(x,y)
    if method=="spearman":
        return pearsonr(rankdata(x), rankdata(y))

def corr_matrix(X,method):
    F=X.shape[1]
    R=np.full((F,F),np.nan)
    P=np.full((F,F),np.nan)
    for i in range(F):
        R[i,i]=1; P[i,i]=0
        for j in range(i+1,F):
            r,p = corr_pair(X[:,i],X[:,j],method)
            R[i,j]=R[j,i]=r
            P[i,j]=P[j,i]=p
    return R,P

def save_bundle(X_nwf, tag):
    X_nf = collapse_subject(X_nwf)[:, :48]

    Rs, _ = corr_matrix(X_nf, "spearman")

    # Upper triangle only — mask lower triangle with NaN
    F = Rs.shape[0]
    M = Rs.copy().astype(float)
    M[np.tril_indices(F, k=-1)] = np.nan

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(M, aspect="auto", vmin=-1, vmax=1, cmap=cmap)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman correlation", fontsize=12)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=12)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=12)

    # Family boundary lines and labels
    boundaries = set()
    for indices in FEATURE_FAMILY_MAP.values():
        boundaries.add(min(indices))
        boundaries.add(max(indices) + 1)
    boundaries.discard(0)
    boundaries.discard(F)
    for b in sorted(boundaries):
        ax.axvline(b - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(b - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    labels_sorted = sorted(
        [(name, (min(idx) + max(idx)) / 2) for name, idx in FEATURE_FAMILY_MAP.items()],
        key=lambda t: t[1],
    )
    for i, (name, mid) in enumerate(labels_sorted):
        y = -1 if i % 2 == 0 else -2.5
        ax.text(mid, y, name, ha="center", va="bottom", fontsize=12,
                color="dimgray", style="italic", clip_on=False)
        ax.text(-4.0, mid, name, ha="right", va="center", fontsize=12,
                color="dimgray", style="italic", clip_on=False)

    fig.tight_layout()
    out = OUT_MAP_DIR / f"{tag}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved:", out)

# =========================
# LOAD
# =========================
Xtr,ytr,X1,y1,X2,y2 = load_all_data(
    train_mat_path=TRAIN_MAT_PATH,
    train_ids_mat_path=TRAIN_IDS_PATH,
    train_xlsx_path=TRAIN_XLSX_PATH,
    test_dir=TEST_DIR
)

Xtr=flatten_to_NWF(Xtr)
X1 =flatten_to_NWF(X1)
X2 =flatten_to_NWF(X2)

Xall=np.concatenate([Xtr,X1,X2])
yall=np.concatenate([ytr,y1,y2])

# =========================
# RUN
# =========================
save_bundle(Xall, "all_ages_combined")

m = is_male(yall[:, GENDER_IDX])
f = is_female(yall[:, GENDER_IDX])

if np.any(m): save_bundle(Xall[m], "all_ages_male")
if np.any(f): save_bundle(Xall[f], "all_ages_female")

print("Done.")
