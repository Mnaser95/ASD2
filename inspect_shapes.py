"""
inspect_shapes.py — reports how many time steps (data points) are dropped
due to NaN removal for each feature-family ablation, across train/T1/T2.
"""

from data_import import load_all_data
from pathlib import Path
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ROOT     = CODE_DIR.parent
OUT_DIR  = ROOT / "outputs" / "modeling2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES_TOTAL = 49
N_SLICES         = 5

FEATURE_FAMILY_MAP = {
    "pitch":         list(range(0, 10)),
    "formants":      list(range(10, 20)),
    "jitter":        [20, 21],
    "voicing":       [22, 23],
    "energy":        list(range(24, 32)),
    "zero_crossing": list(range(32, 38)),
    "spsl":          list(range(38, 46)),
    "duration":      [46, 47],
    "quantity":      [48],
}

# ---------------------------------------------------------------------------

def resolve_drop_idx(family_map, families):
    exclude = []
    for fam in families:
        exclude.extend(family_map[fam])
    return sorted(set(int(i) for i in exclude))

def drop_feature_indices(X, drop_idx):
    if not drop_idx:
        return X
    keep = [i for i in range(X.shape[-1]) if i not in set(drop_idx)]
    return X[..., keep]

def extract_slice(X_raw, set_idx):
    return np.asarray(X_raw, dtype=np.float32)[:, :, set_idx, :]

def count_dropped(X):
    """X: (N, T, F) — returns total time steps dropped across all subjects."""
    total = X.shape[0] * X.shape[1]
    valid = int((~np.isnan(X).any(axis=2)).sum())
    return total, valid, total - valid

# ---------------------------------------------------------------------------

print("Loading data...")
X_train_raw, y_train_raw, X_T1_raw, y_T1_raw, X_T2_raw, y_T2_raw = load_all_data(
    train_mat_path=str(ROOT / "data" / "training" / "train_data.mat"),
    train_ids_mat_path=str(ROOT / "data" / "training" / "ids_fixed.mat"),
    train_xlsx_path=str(ROOT / "data" / "training" / "data_train.xlsx"),
    test_dir=str(ROOT / "data" / "testing"),
)

ablation_configs = [("none", [])] + [(fam, [fam]) for fam in FEATURE_FAMILY_MAP]

rows = []

for ablation_name, ablate_families in ablation_configs:
    drop_idx = resolve_drop_idx(FEATURE_FAMILY_MAP, ablate_families)

    # average dropped counts across slices
    slice_results = {s: [] for s in ("train", "T1", "T2")}

    for slice_idx in range(N_SLICES):
        for split_name, X_raw in [("train", X_train_raw), ("T1", X_T1_raw), ("T2", X_T2_raw)]:
            X = drop_feature_indices(extract_slice(X_raw, slice_idx), drop_idx)
            total, valid, dropped = count_dropped(X)
            slice_results[split_name].append((total, valid, dropped))

    row = {"ablation_family": ablation_name}
    for split_name in ("train", "T1", "T2"):
        totals   = [r[0] for r in slice_results[split_name]]
        valids   = [r[1] for r in slice_results[split_name]]
        droppeds = [r[2] for r in slice_results[split_name]]
        row[f"{split_name}_total"]   = int(np.mean(totals))
        row[f"{split_name}_kept"]    = int(np.mean(valids))
        row[f"{split_name}_dropped"] = int(np.mean(droppeds))
    rows.append(row)

summary = pd.DataFrame(rows).sort_values("ablation_family").reset_index(drop=True)

out_path = OUT_DIR / "shape_inspection.csv"
summary.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(summary.to_string(index=False))
