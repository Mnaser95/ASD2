"""
modeling2.py — subject-level models (CNN + XGBoost) using random-sample averaging.

Pipeline:
  1. Load raw data: X shape (N, 10, 100, 49)
  2. Combine dims 1+2 → (N, 1000, 49), drop last feature → (N, 1000, 48)
  3. Per subject, drop samples (rows) where any feature is NaN
  4. N_SLICES runs: randomly pick N_PICK valid samples per subject, average → (N, 48)
  5. CNN: input (N, 48, 1) — treats features as 1-D sequence
     XGBoost: input (N, 48) — flat feature vector
"""

from data_import import load_all_data
from pathlib import Path
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              cohen_kappa_score, confusion_matrix)
from gpu_utils import require_accelerators, configure_tensorflow_gpu, log_runtime_gpu_status

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

N_FEATURES = 48      # drop last feature (F49)
N_PICK     = 20       # samples to randomly pick and average per subject per slice
N_SLICES   = 20       # number of independent random draws

TARGET_COL_IDXS  = [3, 4, 5]
TARGET_COL_NAMES = {3: "ados", 4: "sa", 5: "rrb"}
FORCE_CLASSIFICATION = True
THRESHOLD_METHOD     = "median"
RUN_CNN         = True
RUN_XGBOOST_AVG = True
MODEL_FAMILIES  = (["cnn"] if RUN_CNN else []) + (["xgboost_avg"] if RUN_XGBOOST_AVG else [])

RUN_REGRESSION     = True
RUN_CLASSIFICATION = True

EXCLUDE_FAMILIES = []

BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 20

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

CODE_DIR   = Path(__file__).resolve().parent
ROOT       = CODE_DIR.parent
TRAIN_MAT  = ROOT / r"data\training\train_data.mat"
TRAIN_IDS  = ROOT / r"data\training\ids_fixed.mat"
TRAIN_XLSX = ROOT / r"data\training\data_train.xlsx"
TEST_DIR   = ROOT / r"data\testing"
OUT_DIR    = ROOT / "outputs" / "modeling2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

########################################################################################################################
# UTILITY

def to_float_1d(arr):
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=np.float32)

def reset_seed():
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.keras.utils.set_random_seed(SEED)
    except AttributeError:
        pass

def make_ds(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 5000), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ---------- feature exclusion ----------
def resolve_excluded_feature_indices(family_map, exclude_families, n_features_total, strict=True):
    exclude = []
    for fam in exclude_families:
        if fam not in family_map:
            msg = f"Unknown family '{fam}'. Known: {sorted(family_map.keys())}"
            if strict: raise ValueError(msg)
            print("WARNING:", msg); continue
        exclude.extend(list(family_map[fam]))
    exclude = sorted(set(int(i) for i in exclude))
    invalid = [i for i in exclude if i < 0 or i >= n_features_total]
    if invalid:
        msg = f"Invalid feature indices (out of range 0..{n_features_total-1}): {invalid}"
        if strict: raise ValueError(msg)
        print("WARNING:", msg)
        exclude = [i for i in exclude if 0 <= i < n_features_total]
    return exclude

# ---------- data preparation ----------
def flatten_and_clean(X_raw):
    """
    X_raw: (N, T, W, F)  e.g. (136, 10, 100, 49)
    Combines dims 1+2 → (N, 1000, F), drops last feature → (N, 1000, N_FEATURES).
    Returns list of N arrays, each (n_valid_i, N_FEATURES) — NaN samples dropped.
    """
    N = X_raw.shape[0]
    X = X_raw.reshape(N, -1, X_raw.shape[-1])[:, :, :N_FEATURES]  # (N, W_total, N_FEATURES)
    result = []
    for i in range(N):
        s     = X[i]
        valid = np.isfinite(s).all(axis=1)
        result.append(s[valid].astype(np.float32))
    return result

def sample_raw(X_clean, drop_idx, n_pick, rng):
    """
    Randomly picks n_pick samples per subject → (N, n_pick, F_kept).
    NaN slice if subject has no valid samples.
    """
    keep = [j for j in range(N_FEATURES) if j not in set(drop_idx)]
    N    = len(X_clean)
    out  = np.full((N, n_pick, len(keep)), np.nan, dtype=np.float32)
    for i, s in enumerate(X_clean):
        if s.shape[0] == 0:
            continue
        idx     = rng.choice(s.shape[0], size=n_pick, replace=s.shape[0] < n_pick)
        out[i]  = s[idx][:, keep]
    return out


# ---------- z-scoring ----------
def zscore_fit(X):
    """X: (N, F) — fit on rows with no NaN."""
    valid = X[np.isfinite(X).all(axis=1)]
    mean  = valid.mean(axis=0).astype(np.float32)
    std   = (valid.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std

def zscore_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

########################################################################################################################
# METRICS

def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    rmse   = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return float(rmse / ((np.max(y_true) - np.min(y_true)) + 1e-8))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred, np.float32) - np.asarray(y_true, np.float32))))

def get_thresholds(y_train):
    """Returns median threshold for binary classification."""
    y = np.asarray(y_train, dtype=np.float32)
    return float(np.percentile(y, 50)),

def to_2class(y, t1):
    """0 = low (below median), 1 = high (at or above median)."""
    y = np.asarray(y, dtype=np.float32)
    return (y >= t1).astype(np.int32)

def _cls_metrics(y_lbl, y_hat):
    acc      = float(np.mean(y_hat == y_lbl))
    f1_mac   = float(f1_score(y_lbl, y_hat, average="macro",    zero_division=0))
    f1_wt    = float(f1_score(y_lbl, y_hat, average="weighted", zero_division=0))
    prec     = float(precision_score(y_lbl, y_hat, average="macro",    zero_division=0))
    rec      = float(recall_score(y_lbl, y_hat, average="macro",       zero_division=0))
    kappa    = float(cohen_kappa_score(y_lbl, y_hat))
    cm       = confusion_matrix(y_lbl, y_hat).tolist()
    return {"acc": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt,
            "precision": prec, "recall": rec, "kappa": kappa, "cm": cm}

def evaluate_classification(model, X, y_lbl, name):
    y_lbl = np.asarray(y_lbl, dtype=np.int32).reshape(-1)
    t0    = time.perf_counter()
    y_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    infer_time = time.perf_counter() - t0
    y_hat = np.argmax(y_prob, axis=1).astype(np.int32)
    m = _cls_metrics(y_lbl, y_hat)
    m["infer_time_s"] = infer_time
    print(f"{name} -> ACC={m['acc']:.4f} | F1={m['f1_macro']:.4f} | κ={m['kappa']:.4f} | infer={infer_time:.3f}s")
    return m

def evaluate_regression(model, X, y, name):
    y      = np.asarray(y, np.float32).reshape(-1)
    t0     = time.perf_counter()
    y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0).reshape(-1).astype(np.float32)
    infer_time = time.perf_counter() - t0
    nrmse_val = nrmse(y, y_pred)
    mae_val   = mae(y, y_pred)
    print(f"{name} -> NRMSE={nrmse_val:.4f} | MAE={mae_val:.4f} | infer={infer_time:.3f}s")
    return {"nrmse": nrmse_val, "mae": mae_val, "infer_time_s": infer_time}

def evaluate_xgb_classification(model, X, y_lbl, name):
    y_lbl = np.asarray(y_lbl, dtype=np.int32).reshape(-1)
    t0    = time.perf_counter()
    y_hat = model.predict(X).astype(np.int32)
    infer_time = time.perf_counter() - t0
    m = _cls_metrics(y_lbl, y_hat)
    m["infer_time_s"] = infer_time
    print(f"{name} -> ACC={m['acc']:.4f} | F1={m['f1_macro']:.4f} | κ={m['kappa']:.4f} | infer={infer_time:.3f}s")
    return m

def evaluate_xgb_regression(model, X, y, name):
    y      = np.asarray(y, np.float32).reshape(-1)
    t0     = time.perf_counter()
    y_pred = model.predict(X).astype(np.float32)
    infer_time = time.perf_counter() - t0
    nrmse_val = nrmse(y, y_pred)
    mae_val   = mae(y, y_pred)
    print(f"{name} -> NRMSE={nrmse_val:.4f} | MAE={mae_val:.4f} | infer={infer_time:.3f}s")
    return {"nrmse": nrmse_val, "mae": mae_val, "infer_time_s": infer_time}

########################################################################################################################
# MODELS

def build_cnn(input_shape, lr=1e-3, task="regression"):
    """
    CNN for 1-D sequences. Input shape: (F, 1) when features are treated as the time axis.
    Conv1D(64) → MaxPool(2) → Conv1D(128) → Flatten → Dense(128) → Dense(64) → output.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.initializers import RandomUniform

    init = RandomUniform(seed=1)

    if task == "classification":
        n_out, out_activation = 2, "softmax"
    else:
        n_out, out_activation = 1, "linear"

    model = Sequential([
        Conv1D(64, 3, activation="relu", padding="same", input_shape=input_shape, kernel_initializer=init),
        MaxPool1D(pool_size=2),
        Conv1D(128, 3, activation="relu", padding="same", kernel_initializer=init),
        Flatten(),
        Dense(128, activation="relu", kernel_initializer=init),
        Dropout(0.3),
        Dense(64,  activation="relu", kernel_initializer=init),
        Dense(n_out, activation=out_activation, kernel_initializer=init),
    ])

    if task == "classification":
        model.compile(optimizer=RMSprop(learning_rate=lr), loss="sparse_categorical_crossentropy",
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    else:
        model.compile(optimizer=RMSprop(learning_rate=lr), loss="mse")
    return model

def build_xgboost(task):
    from xgboost import XGBClassifier, XGBRegressor
    params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=0,
        early_stopping_rounds=30,
    )
    if task == "classification":
        return XGBClassifier(**params, eval_metric="merror", num_class=2, objective="multi:softmax")
    else:
        return XGBRegressor(**params, eval_metric="rmse")

########################################################################################################################
# MAIN

def prepare_split(X_clean, y_raw, target_col_idx, drop_idx, n_pick, rng):
    """
    Sample raw picks, filter invalid labels.
    Returns X_seq (N_valid, n_pick, F), X_avg (N_valid, F), y (N_valid,).
    X_seq and X_avg are derived from the same random draws for fair comparison.
    """
    X_seq = sample_raw(X_clean, drop_idx, n_pick, rng)   # (N, n_pick, F)
    X_avg = np.nanmean(X_seq, axis=1)                     # (N, F)
    y     = to_float_1d(y_raw[:, target_col_idx])
    ok    = np.isfinite(y) & np.isfinite(X_avg).all(axis=1)
    return X_seq[ok], X_avg[ok], y[ok]


def main(task_override=None):
    require_accelerators(require_gpu=False, require_npu=False)
    tf_gpu_info = configure_tensorflow_gpu(verbose=True)
    log_runtime_gpu_status("modeling2.py")
    if int(tf_gpu_info.get("gpu_count", 0)) == 0:
        print("[GPU] modeling2.py: training will run on CPU.")

    X_train_raw, y_train_raw, X_T1_raw, y_T1_raw, X_T2_raw, y_T2_raw = load_all_data(
        train_mat_path=str(TRAIN_MAT), train_ids_mat_path=str(TRAIN_IDS),
        train_xlsx_path=str(TRAIN_XLSX), test_dir=str(TEST_DIR),
    )

    # Pre-process once: flatten dims 1+2, drop last feature, drop NaN samples per subject
    X_train_clean = flatten_and_clean(X_train_raw)
    X_T1_clean    = flatten_and_clean(X_T1_raw)
    X_T2_clean    = flatten_and_clean(X_T2_raw)

    task = str(task_override).strip().lower() if task_override else (
        "classification" if FORCE_CLASSIFICATION else "regression"
    )

    for target_col_idx in TARGET_COL_IDXS:
        print(f"\n==================== Target: {TARGET_COL_NAMES[target_col_idx]} ====================")

        ablation_configs = [("none", [])] + [(fam, [fam]) for fam in FEATURE_FAMILY_MAP.keys()]
        summary_rows = []

        for ablation_name, ablate_families in ablation_configs:
            print(f"\n==================== Ablation: {ablation_name} ====================")

            dropped_families = list(dict.fromkeys(list(EXCLUDE_FAMILIES) + list(ablate_families)))
            drop_idx = resolve_excluded_feature_indices(
                FEATURE_FAMILY_MAP, dropped_families, N_FEATURES, strict=False)
            if drop_idx:
                print(f"Dropping {len(drop_idx)} features: {drop_idx}")
            else:
                print("No features excluded.")

            for model_family in MODEL_FAMILIES:
                print(f"\n#################### Model: {model_family} ####################")

                slice_res_T1 = []
                slice_res_T2 = []

                for slice_idx in range(N_SLICES):
                    print(f"\n===== Slice {slice_idx + 1}/{N_SLICES} =====")
                    reset_seed()

                    # independent RNGs per split so picks don't interfere
                    rng_tr = np.random.default_rng(SEED + slice_idx * 3)
                    rng_v  = np.random.default_rng(SEED + slice_idx * 3 + 1)
                    rng_t  = np.random.default_rng(SEED + slice_idx * 3 + 2)

                    X_tr_seq, X_tr_avg, y_tr = prepare_split(X_train_clean, y_train_raw, target_col_idx, drop_idx, N_PICK, rng_tr)
                    X_v_seq,  X_v_avg,  y_v  = prepare_split(X_T1_clean,    y_T1_raw,    target_col_idx, drop_idx, N_PICK, rng_v)
                    X_t_seq,  X_t_avg,  y_t  = prepare_split(X_T2_clean,    y_T2_raw,    target_col_idx, drop_idx, N_PICK, rng_t)

                    if len(y_tr) == 0:
                        print("WARNING: no training samples — skipping.")
                        continue

                    # z-score using train avg statistics (applied to both avg and seq)
                    mean, std  = zscore_fit(X_tr_avg)
                    X_tr_avg   = zscore_apply(X_tr_avg, mean, std)
                    X_v_avg    = zscore_apply(X_v_avg,  mean, std)
                    X_t_avg    = zscore_apply(X_t_avg,  mean, std)
                    X_tr_seq   = zscore_apply(X_tr_seq, mean, std)
                    X_v_seq    = zscore_apply(X_v_seq,  mean, std)
                    X_t_seq    = zscore_apply(X_t_seq,  mean, std)

                    print(f"N_train={len(y_tr)} | N_T1={len(y_v)} | N_T2={len(y_t)} | F={X_tr_avg.shape[1]}")

                    if task == "classification":
                        (t1,) = get_thresholds(y_tr)
                        ytr_fit = to_2class(y_tr, t1)
                        yv_fit  = to_2class(y_v,  t1)
                        yt_lbl  = to_2class(y_t,  t1)
                    else:
                        ytr_fit = y_tr.astype(np.float32)
                        yv_fit  = y_v.astype(np.float32)

                    # ---- CNN ----
                    if model_family == "cnn":
                        input_shape = (X_tr_seq.shape[1], X_tr_seq.shape[2])
                        model    = build_cnn(input_shape, lr=LR, task=task)
                        ds_train = make_ds(X_tr_seq, ytr_fit, BATCH_SIZE, shuffle=True)
                        ds_val   = make_ds(X_v_seq,  yv_fit,  BATCH_SIZE, shuffle=False)

                        monitor = "val_acc" if task == "classification" else "val_loss"
                        mode    = "max"     if task == "classification" else "min"
                        t_train_start = time.perf_counter()
                        history = model.fit(
                            ds_train, validation_data=ds_val, epochs=EPOCHS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(
                                monitor=monitor, mode=mode, patience=PATIENCE,
                                restore_best_weights=True, verbose=1)],
                            verbose=1,
                        )
                        train_time = time.perf_counter() - t_train_start
                        epochs_run = len(history.history["loss"])
                        print(f"CNN train time: {train_time:.2f}s | epochs: {epochs_run}")

                        if task == "classification":
                            res_T1 = evaluate_classification(model, X_v_seq, yv_fit, f"T1 slice{slice_idx}")
                            res_T2 = evaluate_classification(model, X_t_seq, yt_lbl, f"T2 slice{slice_idx}")
                        else:
                            res_T1 = evaluate_regression(model, X_v_seq, y_v, f"T1 slice{slice_idx}")
                            res_T2 = evaluate_regression(model, X_t_seq, y_t, f"T2 slice{slice_idx}")

                    # ---- XGBoost ----
                    elif model_family == "xgboost_avg":
                        model = build_xgboost(task)
                        t_train_start = time.perf_counter()
                        model.fit(X_tr_avg, ytr_fit, eval_set=[(X_v_avg, yv_fit)], verbose=False)
                        train_time = time.perf_counter() - t_train_start
                        epochs_run = int(model.best_iteration) + 1 if hasattr(model, "best_iteration") else None
                        print(f"XGBoost train time: {train_time:.2f}s | best_iteration: {epochs_run}")

                        if task == "classification":
                            res_T1 = evaluate_xgb_classification(model, X_v_avg, yv_fit, f"T1 slice{slice_idx}")
                            res_T2 = evaluate_xgb_classification(model, X_t_avg, yt_lbl, f"T2 slice{slice_idx}")
                        else:
                            res_T1 = evaluate_xgb_regression(model, X_v_avg, y_v, f"T1 slice{slice_idx}")
                            res_T2 = evaluate_xgb_regression(model, X_t_avg, y_t, f"T2 slice{slice_idx}")

                    res_T2["train_time_s"] = train_time
                    res_T2["epochs_run"]   = epochs_run
                    slice_res_T1.append(res_T1)
                    slice_res_T2.append(res_T2)

                if not slice_res_T2:
                    print("No valid slices — skipping.")
                    continue

                # ---- average across slices ----
                def _avg(key): return float(np.mean([r[key] for r in slice_res_T2 if key in r]))
                def _std(key): return float(np.std( [r[key] for r in slice_res_T2 if key in r]))

                train_time_mean = _avg("train_time_s")
                train_time_std  = _std("train_time_s")
                infer_time_mean = _avg("infer_time_s")
                epochs_mean     = float(np.mean([r["epochs_run"] for r in slice_res_T2 if r.get("epochs_run") is not None]))

                if task == "classification":
                    t2_acc_mean  = _avg("acc");  t2_acc_std  = _std("acc")
                    f1_mac_mean  = _avg("f1_macro");  f1_mac_std  = _std("f1_macro")
                    f1_wt_mean   = _avg("f1_weighted")
                    prec_mean    = _avg("precision")
                    rec_mean     = _avg("recall")
                    kappa_mean   = _avg("kappa");  kappa_std  = _std("kappa")
                    print(f"[AVG {N_SLICES} slices] {ablation_name} | {model_family} | "
                          f"ACC={t2_acc_mean:.4f}±{t2_acc_std:.4f} | F1={f1_mac_mean:.4f}±{f1_mac_std:.4f} | "
                          f"κ={kappa_mean:.4f}±{kappa_std:.4f} | train={train_time_mean:.2f}s")
                    summary_rows.append({
                        "target_col_idx":   target_col_idx,
                        "ablation_family":  ablation_name,
                        "model_family":     model_family,
                        "t2_acc_mean":      t2_acc_mean,      "t2_acc_std":      t2_acc_std,
                        "t2_f1_macro_mean": f1_mac_mean,      "t2_f1_macro_std": f1_mac_std,
                        "t2_f1_weighted":   f1_wt_mean,
                        "t2_precision":     prec_mean,
                        "t2_recall":        rec_mean,
                        "t2_kappa_mean":    kappa_mean,       "t2_kappa_std":    kappa_std,
                        "train_time_mean":  train_time_mean,  "train_time_std":  train_time_std,
                        "infer_time_mean":  infer_time_mean,
                        "epochs_mean":      epochs_mean,
                    })
                else:
                    t2_nrmse_mean = _avg("nrmse"); t2_nrmse_std = _std("nrmse")
                    t2_mae_mean   = _avg("mae");   t2_mae_std   = _std("mae")
                    print(f"[AVG {N_SLICES} slices] {ablation_name} | {model_family} | "
                          f"NRMSE={t2_nrmse_mean:.4f}±{t2_nrmse_std:.4f} | MAE={t2_mae_mean:.4f}±{t2_mae_std:.4f} | "
                          f"train={train_time_mean:.2f}s")
                    summary_rows.append({
                        "target_col_idx":  target_col_idx,
                        "ablation_family": ablation_name,
                        "model_family":    model_family,
                        "t2_nrmse_mean":   t2_nrmse_mean,    "t2_nrmse_std":   t2_nrmse_std,
                        "t2_mae_mean":     t2_mae_mean,      "t2_mae_std":     t2_mae_std,
                        "train_time_mean": train_time_mean,  "train_time_std": train_time_std,
                        "infer_time_mean": infer_time_mean,
                        "epochs_mean":     epochs_mean,
                    })

        # ---- save summary CSV (merge with existing results from other models) ----
        if summary_rows:
            new_df = pd.DataFrame(summary_rows)
            path   = OUT_DIR / f"summary_{TARGET_COL_NAMES[target_col_idx]}_{task}.csv"

            if path.exists():
                existing = pd.read_csv(path)
                # drop rows for models we just re-ran to avoid duplicates
                existing = existing[~existing["model_family"].isin(new_df["model_family"].unique())]
                summary_df = pd.concat([existing, new_df], ignore_index=True)
            else:
                summary_df = new_df

            # recompute deltas across all models now in the combined df
            baseline_df = summary_df[summary_df["ablation_family"] == "none"].copy()

            if task == "classification":
                summary_df = summary_df.merge(
                    baseline_df[["model_family", "t2_acc_mean", "t2_f1_macro_mean", "t2_kappa_mean"]],
                    on="model_family", how="left", suffixes=("", "_base"))
                summary_df["delta_acc"]      = summary_df["t2_acc_mean"]      - summary_df["t2_acc_mean_base"]
                summary_df["delta_f1_macro"] = summary_df["t2_f1_macro_mean"] - summary_df["t2_f1_macro_mean_base"]
                summary_df["delta_kappa"]    = summary_df["t2_kappa_mean"]    - summary_df["t2_kappa_mean_base"]
                summary_df["rank"]           = summary_df["t2_acc_mean"].rank(ascending=False, method="min").astype(int)
            else:
                summary_df = summary_df.merge(
                    baseline_df[["model_family", "t2_nrmse_mean", "t2_mae_mean"]],
                    on="model_family", how="left", suffixes=("", "_base"))
                summary_df["delta_nrmse"] = summary_df["t2_nrmse_mean"] - summary_df["t2_nrmse_mean_base"]
                summary_df["delta_mae"]   = summary_df["t2_mae_mean"]   - summary_df["t2_mae_mean_base"]
                summary_df["rank"]        = summary_df["t2_nrmse_mean"].rank(ascending=True, method="min").astype(int)

            summary_df.to_csv(path, index=False)
            print(f"Saved: {path}")

########################################################################################################################


if __name__ == "__main__":
    for _task, _run in [("regression", RUN_REGRESSION), ("classification", RUN_CLASSIFICATION)]:
        if not _run:
            continue
        print(f"\n\n==================== RUN TASK: {_task.upper()} ====================")
        main(task_override=_task)
    print("All done.")
