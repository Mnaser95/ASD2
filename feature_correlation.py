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

FEATURE_NAMES = [f"F{i+1:02d}" for i in range(49)]

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
    if USING_GPU:
        X_gpu = XP.asarray(X)
        if REDUCER == "mean":
            return to_numpy(XP.nanmean(X_gpu, axis=1))
        if REDUCER == "median":
            return to_numpy(XP.nanmedian(X_gpu, axis=1))
    if REDUCER == "mean":
        return np.nanmean(X, axis=1)
    if REDUCER == "median":
        return np.nanmedian(X, axis=1)

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
    X_nf = collapse_subject(X_nwf)

    Rp, _ = corr_matrix(X_nf, "pearson")
    Rs, _ = corr_matrix(X_nf, "spearman")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, M, title in [
        (axes[0], Rp, f"Pearson  —  {tag}"),
        (axes[1], Rs, f"Spearman  —  {tag}"),
    ]:
        im = ax.imshow(M, aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(FEATURE_NAMES)))
        ax.set_yticks(range(len(FEATURE_NAMES)))
        ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(FEATURE_NAMES, fontsize=6)

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

