from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, rankdata
from data_import import load_all_data
from gpu_utils import require_accelerators, get_array_backend, to_numpy, log_runtime_gpu_status

# =========================
# CONFIG
# =========================
RUN_ALL_GENDERS = True      # compute correlations for combined genders
RUN_BY_GENDER   = True      # compute correlations separately for male / female

TARGET_COLS = [3, 4, 5]     # columns to correlate against: ADOS, SA, RRB
SCENARIOS = [               # (output_tag, exclude_module4_adults)
    ("all_ages", False),
]

GENDER_IDX = 1              # column index of gender in y
MODULE_IDX = 10             # column index of module in y
N_PERM     = 1000           # permutations for Pearson significance test
ALPHA      = 0.05           # significance threshold

Y_COL_NAMES = [             # label for each y column (0-based)
    "rec_id", "Gender", "Age",
    "ADOS", "SA", "RRB", "CSS", "SA_Rel", "RRB_Rel", "E2", "Module", "A1"
]

CODE_DIR = Path(__file__).resolve().parent
ROOT     = CODE_DIR.parent
OUT_DIR  = ROOT / "outputs" / "exploratory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# SETUP
# =========================
require_accelerators(require_gpu=False, require_npu=False)
XP, USING_GPU = get_array_backend()
log_runtime_gpu_status("exploratory.py")

# =========================
# LOAD DATA
# =========================
X_train, y_train, X_T1, y_T1, X_T2, y_T2 = load_all_data(
    train_mat_path=str(ROOT / "data" / "training" / "train_data.mat"),
    train_ids_mat_path=str(ROOT / "data" / "training" / "ids_fixed.mat"),
    train_xlsx_path=str(ROOT / "data" / "training" / "data_train.xlsx"),
    test_dir=str(ROOT / "data" / "testing")
)

assert len(Y_COL_NAMES) == y_train.shape[1], (
    f"Y_COL_NAMES has {len(Y_COL_NAMES)} names but y has {y_train.shape[1]} columns."
)

# Flatten windows and combine all splits
X_train_flat = X_train.reshape(X_train.shape[0], -1, X_train.shape[-1])
X_T1_flat    = X_T1.reshape(X_T1.shape[0],    -1, X_T1.shape[-1])
X_T2_flat    = X_T2.reshape(X_T2.shape[0],    -1, X_T2.shape[-1])

X_all_flat = np.concatenate([X_train_flat, X_T1_flat, X_T2_flat], axis=0)
y_all      = np.concatenate([y_train, y_T1, y_T2], axis=0)

# =========================
# FILTERS
# =========================
def is_male(arr):
    return (arr == "m") | (arr == "M")

def is_female(arr):
    return (arr == "f") | (arr == "F")

def module_is_4(arr):
    a = np.asarray(arr)
    out = np.zeros(a.shape[0], dtype=bool)
    for i, v in enumerate(a):
        if v is None:
            continue
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                pass
        try:
            out[i] = (float(v) == 4.0)
            continue
        except Exception:
            pass
        try:
            out[i] = str(v).strip() in ("4", "4.0")
        except Exception:
            out[i] = False
    return out

def apply_module4_filter(X, y, exclude: bool):
    if not exclude:
        return X, y
    keep = ~module_is_4(y[:, MODULE_IDX])
    return X[keep], y[keep]

# =========================
# STATISTICS
# =========================
def compute_stats(x_feat_1d, y_col_1d, seed=0, n_perm=N_PERM):
    x = np.asarray(x_feat_1d, dtype=float)
    y = np.asarray(y_col_1d,  dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, False, np.nan, np.nan, int(x.size)

    r_obs, _ = pearsonr(x, y)

    rng    = np.random.default_rng(seed)
    r_null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = rng.permutation(y)
        r_null[i] = 0.0 if np.all(yp == yp[0]) else pearsonr(x, yp)[0]

    p_emp = (np.sum(np.abs(r_null) >= np.abs(r_obs)) + 1) / (n_perm + 1)
    sig_pearson = np.abs(r_obs) > np.percentile(np.abs(r_null), 97.5)

    rho, p_spearman = pearsonr(rankdata(x), rankdata(y))
    return r_obs, p_emp, sig_pearson, rho, p_spearman, int(x.size)

def run_for_dataset(X, y, y_col_idx, seed_base=0):
    y_col = y[:, y_col_idx]
    rows  = []

    for j in range(X.shape[2]):
        x_feat = (to_numpy(XP.asarray(X[:, :, j]).mean(axis=1))
                  if USING_GPU else X[:, :, j].mean(axis=1))

        r, p_emp, sig_pearson, rho, p_spearman, n = compute_stats(
            x_feat, y_col, seed=seed_base + 2000 * y_col_idx + j
        )

        rows.append({
            "feature_idx":      j,
            "pearson_r":        r,
            "pearson_p_emp":    p_emp,
            "pearson_sig":      sig_pearson,
            "spearman_rho":     rho,
            "spearman_p":       p_spearman,
            "spearman_sig":     bool(p_spearman < ALPHA) if np.isfinite(p_spearman) else False,
            "n_used":           n,
        })

    return pd.DataFrame(rows)

# =========================
# PLOTTING
# =========================
GROUPS = [
    ("Combined", "steelblue"),
    ("Male",     "darkorange"),
    ("Female",   "mediumseagreen"),
]

FEATURE_FAMILY_MAP = {
    "pitch":        list(range(0, 10)),
    "formants":     list(range(10, 20)),
    "jitter":       [20, 21],
    "voicing":      [22, 23],
    "energy":       list(range(24, 32)),
    "zero_crossing":list(range(32, 38)),
    "spsl":         list(range(38, 46)),
    "duration":     [46, 47],
    "quantity":     [48],
}

def _add_family_annotations(ax, n_feats, family_map, y_min, y_max):
    """Draw vertical boundary lines and family name labels on ax."""
    boundaries = set()
    for indices in family_map.values():
        boundaries.add(min(indices))          # left edge of each family
        boundaries.add(max(indices) + 1)      # right edge
    boundaries.discard(0)
    boundaries.discard(n_feats)

    for b in sorted(boundaries):
        ax.axvline(b - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    y_label = y_max + (y_max - y_min) * 0.04
    for name, indices in family_map.items():
        mid = (min(indices) + max(indices)) / 2
        ax.text(mid, y_label, name, ha="center", va="bottom", fontsize=9,
                color="dimgray", style="italic")

def plot_grouped_bars(datasets, title, out_path):
    n_groups = len(datasets)
    n_feats  = len(datasets[0][1])
    feature_labels = [f"f{j+1}" for j in range(n_feats)]

    width = 0.8 / n_groups
    x = np.arange(n_feats)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(22, 10), sharex=True)

    for panel, (metric, p_col, ylabel) in enumerate([
        ("pearson_r",    "pearson_p_emp", "Pearson r"),
        ("spearman_rho", "spearman_p",    "Spearman ρ"),
    ]):
        ax = axes[panel]
        ax.axhline(0, linewidth=1, color="black")

        all_vals = []
        for g_idx, (label, df) in enumerate(datasets):
            df = df.sort_values("feature_idx").reset_index(drop=True)
            vals = df[metric].to_numpy(dtype=float)
            ps   = df[p_col].to_numpy(dtype=float)
            color = GROUPS[g_idx][1]
            offset = (g_idx - (n_groups - 1) / 2) * width

            ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.85)
            all_vals.append(vals)

            for i in np.where(np.isfinite(ps) & (ps < ALPHA))[0]:
                v = vals[i]
                stars = "***" if ps[i] < 0.0005 else "**" if ps[i] < 0.005 else "*"
                ax.text(x[i] + offset, v + (0.02 if v >= 0 else -0.02), stars,
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

        ax.set_ylim(-0.6, 0.6)
        _add_family_annotations(ax, n_feats, FEATURE_FAMILY_MAP, -0.6, 0.6)

        ax.set_ylabel(ylabel, fontsize=14)
        if panel == 0:
            ax.set_title(title, fontsize=16)
            from matplotlib.lines import Line2D
            handles, _ = ax.get_legend_handles_labels()
            handles += [
                Line2D([], [], linestyle="none", label="* p < 0.05"),
                Line2D([], [], linestyle="none", label="** p < 0.005"),
                Line2D([], [], linestyle="none", label="*** p < 0.0005"),
            ]
            ax.legend(handles=handles, fontsize=12)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_labels, rotation=90, ha="center", fontsize=9)
    axes[1].set_xlabel("Features", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# =========================
# RUN
# =========================
for MODULE_TAG, EXCLUDE_MODULE_4 in SCENARIOS:
    X_s, y_s = apply_module4_filter(X_all_flat, y_all, EXCLUDE_MODULE_4)

    mask_m = is_male(y_s[:, GENDER_IDX])
    mask_f = is_female(y_s[:, GENDER_IDX])

    for c in TARGET_COLS:
        col_name = Y_COL_NAMES[c]

        df_all = run_for_dataset(X_s,          y_s,          c, seed_base=0)
        df_m   = run_for_dataset(X_s[mask_m],  y_s[mask_m],  c, seed_base=10_000)
        df_f   = run_for_dataset(X_s[mask_f],  y_s[mask_f],  c, seed_base=20_000)

        datasets = [("combined", df_all), ("male", df_m), ("female", df_f)]
        out_path = OUT_DIR / f"{col_name}_{MODULE_TAG}_bars.png"

        plot_grouped_bars(
            datasets,
            title=f"{col_name}",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")
