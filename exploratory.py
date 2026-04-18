from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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

X_all_flat = np.concatenate([X_train_flat, X_T1_flat, X_T2_flat], axis=0)[:, :, :48]
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
def compute_stats(x_feat_1d, y_col_1d):
    x = np.asarray(x_feat_1d, dtype=float)
    y = np.asarray(y_col_1d,  dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, int(x.size)

    rho, p_spearman = spearmanr(x, y)
    return rho, p_spearman, int(x.size)

def run_for_dataset(X, y, y_col_idx):
    y_col = y[:, y_col_idx]
    rows  = []

    for j in range(X.shape[2]):
        x_feat = (to_numpy(XP.asarray(X[:, :, j]).mean(axis=1))
                  if USING_GPU else X[:, :, j].mean(axis=1))

        rho, p_spearman, n = compute_stats(x_feat, y_col)

        rows.append({
            "feature_idx":      j,
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

    y_base = y_max + (y_max - y_min) * 0.04

    labels = sorted(
        [(name, (min(idx) + max(idx)) / 2) for name, idx in family_map.items()],
        key=lambda t: t[1],
    )
    for name, mid in labels:
        ax.text(mid, y_base, name, ha="center", va="bottom", fontsize=18,
                color="dimgray", style="italic")

def plot_grouped_bars(datasets, title, out_path):
    n_groups = len(datasets)
    n_feats  = len(datasets[0][1])
    feature_labels = [f"f{j+1}" for j in range(n_feats)]

    width = 0.8 / n_groups
    x = np.arange(n_feats)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(22, 10))

    ax.axhline(0, linewidth=1, color="black")

    for g_idx, (label, df) in enumerate(datasets):
        df = df.sort_values("feature_idx").reset_index(drop=True)
        vals = df["spearman_rho"].to_numpy(dtype=float)
        ps   = df["spearman_p"].to_numpy(dtype=float)
        color = GROUPS[g_idx][1]
        offset = (g_idx - (n_groups - 1) / 2) * width

        ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.85)

        for i in np.where(np.isfinite(ps) & (ps < ALPHA))[0]:
            v = vals[i]
            stars = "***" if ps[i] < 0.0005 else "**" if ps[i] < 0.005 else "*"
            ax.text(x[i] + offset, v + (0.02 if v >= 0 else -0.02), stars,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    ax.set_ylim(-0.6, 0.6)
    _add_family_annotations(ax, n_feats, FEATURE_FAMILY_MAP, -0.6, 0.6)

    ax.tick_params(axis="y", labelsize=28)
    ax.set_ylabel("Spearman ρ", fontsize=24)
    ax.set_title(title, fontsize=24)
    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], linestyle="none", label="* p < 0.05"),
        Line2D([], [], linestyle="none", label="** p < 0.005"),
        Line2D([], [], linestyle="none", label="*** p < 0.0005"),
    ]
    ax.legend(handles=handles, fontsize=17)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=90, ha="center", fontsize=22)
    ax.set_xlabel("Features", fontsize=24)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# =========================
# GENDER BALANCE ANALYSIS
# =========================
N_BOOTSTRAP_REPS    = 1000  # repetitions for bootstrap CI
N_PERMUTATION_REPS  = 1000  # repetitions for permutation test
BOOTSTRAP_CI        = 95    # confidence interval percentage
RNG_SEED            = 42

BALANCE_DIR = OUT_DIR / "gender_balance"
BALANCE_DIR.mkdir(parents=True, exist_ok=True)


def run_bootstrap_ci(X, y, col_idx, n_reps, ci, seed, resample_size=None):
    """
    Bootstrap CI for Spearman rho per feature.
    resample_size: number of subjects drawn per bootstrap sample.
                   Pass min(N_male, N_female) for both groups to equalise CI width.
    Returns DataFrame with mean rho, lower and upper CI bounds.
    """
    n_feats  = X.shape[2]
    rng      = np.random.default_rng(seed)
    size     = resample_size if resample_size is not None else X.shape[0]
    all_rhos = np.full((n_reps, n_feats), np.nan)

    for r in range(n_reps):
        idx    = rng.choice(X.shape[0], size=size, replace=True)
        df_sub = run_for_dataset(X[idx], y[idx], col_idx)
        all_rhos[r] = df_sub.sort_values("feature_idx")["spearman_rho"].values

    alpha    = (100 - ci) / 2
    mean_rho = np.nanmean(all_rhos, axis=0)
    lo       = np.nanpercentile(all_rhos, alpha,       axis=0)
    hi       = np.nanpercentile(all_rhos, 100 - alpha, axis=0)
    rows = [{"feature_idx": j, "spearman_rho": mean_rho[j], "ci_lo": lo[j], "ci_hi": hi[j]}
            for j in range(n_feats)]
    return pd.DataFrame(rows)


def run_permutation_test(X_m, y_m, X_f, y_f, col_idx, n_reps, seed):
    """
    For each feature, test whether the observed difference in Spearman rho
    between males and females is significant by permuting gender labels.
    Returns DataFrame with observed delta_rho and p-value per feature.
    """
    n_feats = X_m.shape[2]
    rng     = np.random.default_rng(seed)

    # pool all subjects
    X_all = np.concatenate([X_m, X_f], axis=0)
    y_all = np.concatenate([y_m, y_f], axis=0)
    n_m   = X_m.shape[0]
    N     = X_all.shape[0]

    # observed rho difference
    df_m_obs = run_for_dataset(X_m, y_m, col_idx).sort_values("feature_idx")
    df_f_obs = run_for_dataset(X_f, y_f, col_idx).sort_values("feature_idx")
    obs_delta = df_m_obs["spearman_rho"].values - df_f_obs["spearman_rho"].values

    # permutation null distribution
    null_deltas = np.full((n_reps, n_feats), np.nan)
    for r in range(n_reps):
        perm   = rng.permutation(N)
        idx_m  = perm[:n_m]
        idx_f  = perm[n_m:]
        d_m    = run_for_dataset(X_all[idx_m], y_all[idx_m], col_idx).sort_values("feature_idx")
        d_f    = run_for_dataset(X_all[idx_f], y_all[idx_f], col_idx).sort_values("feature_idx")
        null_deltas[r] = d_m["spearman_rho"].values - d_f["spearman_rho"].values

    # two-sided p-value: proportion of permutations with |delta| >= |observed|
    p_vals = np.mean(np.abs(null_deltas) >= np.abs(obs_delta), axis=0)

    rows = [{"feature_idx": j, "delta_rho": obs_delta[j], "p_value": p_vals[j]}
            for j in range(n_feats)]
    return pd.DataFrame(rows)


def plot_balance(df_m_boot, df_f_boot, col_name, out_path, df_perm=None):
    """Bootstrap CI for male and female Spearman rho per feature.
    Permutation test p-values annotated as stars if df_perm is provided."""
    n_feats = len(df_m_boot)
    x       = np.arange(n_feats)
    feature_labels = [f"f{j+1}" for j in range(n_feats)]

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.axhline(0, linewidth=1, color="black")

    for df, label, color in [
        (df_m_boot, "Male",   "darkorange"),
        (df_f_boot, "Female", "mediumseagreen"),
    ]:
        df   = df.sort_values("feature_idx").reset_index(drop=True)
        vals = df["spearman_rho"].values
        ax.plot(x, vals, color=color, linewidth=1.5, label=label, marker="o", markersize=3)
        ax.fill_between(x, df["ci_lo"].values, df["ci_hi"].values, color=color, alpha=0.2)

    # annotate permutation test significance at top of plot
    if df_perm is not None:
        df_perm = df_perm.sort_values("feature_idx").reset_index(drop=True)
        y_star  = 0.72
        for j, p in enumerate(df_perm["p_value"].values):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                continue
            ax.text(j, y_star, stars, ha="center", va="bottom", fontsize=8, color="black")

    ax.set_ylim(-0.8, 0.8)
    _add_family_annotations(ax, n_feats, FEATURE_FAMILY_MAP, -0.8, 0.8)
    ax.tick_params(axis="y", labelsize=28)
    ax.set_ylabel("Spearman ρ", fontsize=24)
    ax.set_title(f"{col_name}", fontsize=24)
    ax.set_xlabel("Features", fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=90, ha="center", fontsize=22)
    ax.legend(fontsize=17)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =========================
# RUN
# =========================
for MODULE_TAG, EXCLUDE_MODULE_4 in SCENARIOS:
    X_s, y_s = apply_module4_filter(X_all_flat, y_all, EXCLUDE_MODULE_4)

    mask_m = is_male(y_s[:, GENDER_IDX])
    mask_f = is_female(y_s[:, GENDER_IDX])

    print(f"N_male={mask_m.sum()} | N_female={mask_f.sum()}")

    for c in TARGET_COLS:
        col_name = Y_COL_NAMES[c]

        df_all = run_for_dataset(X_s,         y_s,         c)
        df_m   = run_for_dataset(X_s[mask_m], y_s[mask_m], c)
        df_f   = run_for_dataset(X_s[mask_f], y_s[mask_f], c)

        datasets = [("combined", df_all), ("male", df_m), ("female", df_f)]
        out_path = OUT_DIR / f"{col_name}_{MODULE_TAG}_bars.png"

        plot_grouped_bars(
            datasets,
            title=f"{col_name}",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")

        # ---- gender balance analysis ----
        X_m, y_m = X_s[mask_m], y_s[mask_m]
        X_f, y_f = X_s[mask_f], y_s[mask_f]

        n_matched = min(X_m.shape[0], X_f.shape[0])
        df_m_boot = run_bootstrap_ci(X_m, y_m, c, N_BOOTSTRAP_REPS, BOOTSTRAP_CI, RNG_SEED, resample_size=n_matched)
        df_f_boot = run_bootstrap_ci(X_f, y_f, c, N_BOOTSTRAP_REPS, BOOTSTRAP_CI, RNG_SEED, resample_size=n_matched)

        plot_balance(
            df_m_boot, df_f_boot,
            col_name=col_name,
            out_path=BALANCE_DIR / f"{col_name}_{MODULE_TAG}_balance.png",
        )
