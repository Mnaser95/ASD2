from data_import import load_all_data
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from gpu_utils import require_accelerators, configure_tensorflow_gpu, log_runtime_gpu_status

SEEDS = [42]
SEED = SEEDS[0]
tf.random.set_seed(SEED)
np.random.seed(SEED)

N_FEATURES_TOTAL = 49
TARGET_COL_IDX = 3  # default (kept for compatibility)
TARGET_COL_IDXS = [3, 4, 5]
TARGET_COL_NAMES = {3: "ados", 4: "sa", 5: "rrb"}
FORCE_CLASSIFICATION = True
THRESHOLD_METHOD = "median" # this is used to convert the values into a classification problem.
MODEL_FAMILY = "transformer"  # kept for backward compatibility
MODEL_FAMILIES = ["cnn", "transformer"]
MODEL_SHORT = {
    "cnn": "cnn",
    "transformer": "tr",
}

# this is on top of the loop that excludes one family at a time
EXCLUDE_FAMILIES = [
    # "pitch",
    # "formants",
]

BATCH_SIZE = 64
TRANSFORMER_BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 2
PATIENCE = 10

FEATURE_FAMILY_MAP = {
    "pitch": list(range(0, 10)),            # 0..9
    "formants": list(range(10, 20)),        # 10..19
    "jitter": [20, 21],                     # 20..21
    "voicing": [22, 23],                    # 22..23
    "energy": list(range(24, 32)),          # 24..31
    "zero_crossing": list(range(32, 38)),   # 32..37
    "spsl": list(range(38, 46)),            # 38..45
    "duration": [46, 47],                   # 46..47
    "quantity": [48],                       # 48
}

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
TRAIN_MAT = ROOT / r"data\training\train_data.mat"
TRAIN_IDS = ROOT / r"data\training\ids_fixed.mat"
TRAIN_XLSX = ROOT / r"data\training\data_train.xlsx"
TEST_DIR = ROOT / r"data\testing"
OUT_DIR = ROOT / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
########################################################################################################################

def to_float_1d(arr):
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=np.float32)

def set_run_seed(seed):
    global SEED
    SEED = int(seed)
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
            if strict:
                raise ValueError(msg)
            print("WARNING:", msg)
            continue
        exclude.extend(list(family_map[fam]))

    exclude = sorted(set(int(i) for i in exclude))

    invalid = [i for i in exclude if i < 0 or i >= n_features_total]
    if invalid:
        msg = f"Invalid feature indices (out of range 0..{n_features_total-1}): {invalid}"
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
        exclude = [i for i in exclude if 0 <= i < n_features_total]

    return exclude

def drop_feature_indices(X, drop_idx):
    # X: (N,T,F)
    if not drop_idx:
        return X
    drop_set = set(drop_idx)
    keep_idx = [i for i in range(X.shape[-1]) if i not in drop_set]
    if len(keep_idx) == 0:
        raise ValueError("All features were dropped. Keep at least one feature.")
    return X[..., keep_idx]

# ---------- NaN row/time-point removal (NO IMPUTATION) ----------
def remove_nan_time_rows_per_sample(X):
    """
    X: (N,T,F)
    For each sample i, remove rows t where any NaN exists in X[i,t,:].
    Returns list of (Ti,F) arrays + lengths Ti.
    """
    X = np.asarray(X)
    cleaned = []
    lengths = np.zeros(X.shape[0], dtype=np.int32)

    for i in range(X.shape[0]):
        Xi = X[i]  # (T,F)
        good = ~np.isnan(Xi).any(axis=1)
        Xi_clean = Xi[good].astype(np.float32)
        cleaned.append(Xi_clean)
        lengths[i] = Xi_clean.shape[0]

    return cleaned, lengths

def pad_to_length(list_X, lengths, T_target):
    """
    Pads each (Ti,F) to (T_target,F) with zeros.
    Returns X_pad (N,T_target,F) and mask (N,T_target) where 1=real,0=pad.
    """
    N = len(list_X)
    if N == 0:
        raise ValueError("No samples to pad.")
    F = list_X[0].shape[1]

    X_pad = np.zeros((N, T_target, F), dtype=np.float32)
    mask = np.zeros((N, T_target), dtype=np.float32)

    for i, Xi in enumerate(list_X):
        Ti = int(lengths[i])
        Ti_use = min(Ti, T_target)
        if Ti_use > 0:
            X_pad[i, :Ti_use, :] = Xi[:Ti_use, :]
            mask[i, :Ti_use] = 1.0

    return X_pad, mask

def zscore_fit_masked(X, mask):
    """
    Compute mean/std per feature using only non-padded rows (mask==1).
    """
    X = X.astype(np.float32)
    mask = mask.astype(np.float32)

    N, T, F = X.shape
    Xf = X.reshape(N * T, F)
    mf = mask.reshape(N * T)

    valid = mf > 0.5
    Xv = Xf[valid]
    if Xv.shape[0] == 0:
        raise ValueError("All rows are padded after NaN-row removal.")

    mean = Xv.mean(axis=0)
    std = Xv.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)

def zscore_apply_masked(X, mean, std, mask):
    Xn = ((X - mean) / std).astype(np.float32)
    # Keep padded rows neutral.
    Xn *= mask[..., None].astype(np.float32)
    return Xn

# ---------- regression evaluation ----------
def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = (np.max(y_true) - np.min(y_true)) + 1e-8
    return float(rmse / denom)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_pred - y_true)))

def evaluate_regression(model, X, y, name, batch_size=BATCH_SIZE):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_pred = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1).astype(np.float32)

    nrmse_val = float(nrmse(y, y_pred))
    mae_val = float(mae(y, y_pred))
    print(f"{name} -> NRMSE={nrmse_val:.4f} | MAE={mae_val:.4f}")
    return {"nrmse": nrmse_val, "mae": mae_val}

# ---------- classification preparation + evaluation ----------
def get_threshold(y_train, method="median"):
    y_train = np.asarray(y_train, dtype=np.float32)
    if method == "median":
        return float(np.median(y_train))
    raise ValueError("THRESHOLD_METHOD must be 'median'")

def to_binary(y, thr):
    y = np.asarray(y, dtype=np.float32)
    return (y >= float(thr)).astype(np.float32)

def evaluate_classification(model, X, y_true01, name, batch_size=BATCH_SIZE):
    y_true01 = np.asarray(y_true01, dtype=np.float32).reshape(-1)
    y_prob = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1).astype(np.float32)
    y_hat = (y_prob >= 0.5).astype(np.float32)

    acc = float(np.mean(y_hat == y_true01))
    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true01, y_prob)
    auc = float(auc_metric.result().numpy())

    print(f"{name} -> ACC={acc:.4f} | AUC={auc:.4f}")
    return {"acc": acc, "auc": auc}

########################################################################################################################
############################################### CNN
def build_cnn(input_shape, lr=1e-3, task="regression"):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.initializers import RandomUniform

    init = RandomUniform(seed=1)
    out_activation = "sigmoid" if task == "classification" else "linear"

    model = Sequential([
        Conv1D(256, 3, activation="relu", input_shape=input_shape, kernel_initializer=init),
        MaxPool1D(pool_size=3),
        Conv1D(256, 3, activation="relu", kernel_initializer=init),
        Dense(1024, activation="relu", kernel_initializer=init),
        Dropout(0.5),
        Dense(512, activation="relu", kernel_initializer=init),
        Dropout(0.5),
        Dense(256, activation="relu", kernel_initializer=init),
        Dense(128, activation="relu", kernel_initializer=init),
        Flatten(),
        Dense(1, activation=out_activation, kernel_initializer=init),
    ])

    if task == "classification":
        model.compile(
            optimizer=RMSprop(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")]
        )
    else:
        model.compile(
            optimizer=RMSprop(learning_rate=lr),
            loss="mse",
            metrics=[]
        )
    return model

############################################### Transformer
@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_ff, activation="relu"),
            tf.keras.layers.Dense(self.d_model),
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.do1 = tf.keras.layers.Dropout(self.dropout)
        self.do2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.ln1(x + self.do1(attn, training=training))
        ffn = self.ffn(x, training=training)
        x = self.ln2(x + self.do2(ffn, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        })
        return config

def build_transformer(input_shape, lr=1e-3, task="regression", d_model=128, num_heads=4, d_ff=256, n_blocks=3):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inp)

    for _ in range(n_blocks):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    if task == "classification":
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        out = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        )
    return model

########################################################################################################################
def plot_loss_curves(history, tag):
    epochs = np.arange(1, len(history.history.get("loss", [])) + 1)

    plt.figure()
    if "loss" in history.history:
        plt.plot(epochs, history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], label="val loss")

    plt.legend(loc="best")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss curves — {tag}")
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f"losscurves_{tag}.png"))

def plot_performance(metrics_T1, metrics_T2, tag, task):
    plt.figure()
    splits = ["T1", "T2"]

    if task == "classification":
        acc = [metrics_T1["acc"], metrics_T2["acc"]]
        plt.plot(splits, acc, marker="o", label="ACC")
        plt.ylabel("score")
        plt.ylim(0.0, 1.0)
    else:
        nrmse_v = [metrics_T1["nrmse"], metrics_T2["nrmse"]]
        mae_v = [metrics_T1["mae"], metrics_T2["mae"]]
        plt.plot(splits, nrmse_v, marker="o", label="NRMSE (range)")
        plt.plot(splits, mae_v, marker="s", label="MAE")
        plt.ylabel("error")

    plt.xlabel("split")
    plt.title(f"Performance — {tag}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f"performance_{tag}.png"))

########################################################################################################################
def main(task_override=None):
    require_accelerators(require_gpu=False, require_npu=False)
    tf_gpu_info = configure_tensorflow_gpu(verbose=True)
    log_runtime_gpu_status("modeling.py")
    if int(tf_gpu_info.get("gpu_count", 0)) == 0:
        print("[GPU] modeling.py: training will run on CPU because TensorFlow did not detect a GPU.")
    log_runtime_gpu_status("modeling.py")

    X_train_raw, y_train_raw, X_T1_raw, y_T1_raw, X_T2_raw, y_T2_raw = load_all_data(
        train_mat_path=str(TRAIN_MAT),
        train_ids_mat_path=str(TRAIN_IDS),
        train_xlsx_path=str(TRAIN_XLSX),
        test_dir=str(TEST_DIR),
    )

    if task_override is None:
        task = "classification" if FORCE_CLASSIFICATION else "regression"
    else:
        task = str(task_override).strip().lower()

    for target_col_idx in TARGET_COL_IDXS:
        print(f"\n==================== Target Column: {target_col_idx} ====================")

        # Baseline + leave-one-family-out ablations.
        ablation_configs = [("none", [])] + [(fam, [fam]) for fam in FEATURE_FAMILY_MAP.keys()]

        summary_rows = []

        for ablation_name, ablate_families in ablation_configs:
            print(f"\n==================== Ablation: {ablation_name} ====================")

            # (1) RESTRUCTURE FIRST: reshape to (N,T,F)
            X_train = np.asarray(X_train_raw).reshape(X_train_raw.shape[0], -1, X_train_raw.shape[-1]).astype(np.float32)
            X_T1 = np.asarray(X_T1_raw).reshape(X_T1_raw.shape[0], -1, X_T1_raw.shape[-1]).astype(np.float32)
            X_T2 = np.asarray(X_T2_raw).reshape(X_T2_raw.shape[0], -1, X_T2_raw.shape[-1]).astype(np.float32)
            y_train = np.asarray(y_train_raw)
            y_T1 = np.asarray(y_T1_raw)
            y_T2 = np.asarray(y_T2_raw)

            dropped_families = list(dict.fromkeys(list(EXCLUDE_FAMILIES) + list(ablate_families)))
            drop_idx = resolve_excluded_feature_indices(
                FEATURE_FAMILY_MAP,
                dropped_families,
                n_features_total=N_FEATURES_TOTAL,
                strict=False,
            )

            if drop_idx:
                print(f"Dropping {len(drop_idx)} features for ablation '{ablation_name}': {drop_idx}")
                X_train = drop_feature_indices(X_train, drop_idx)
                X_T1 = drop_feature_indices(X_T1, drop_idx)
                X_T2 = drop_feature_indices(X_T2, drop_idx)
            else:
                print(f"No features excluded for ablation '{ablation_name}'.")

            print("Shape after reshape + feature drop (N,T,F):", X_train.shape)

            # (2) DROP NaNs (remove NaN rows per sample; no imputation) + pad to fixed T
            train_list, train_len = remove_nan_time_rows_per_sample(X_train)
            t1_list, t1_len = remove_nan_time_rows_per_sample(X_T1)
            t2_list, t2_len = remove_nan_time_rows_per_sample(X_T2)

            T_global = int(max(train_len.max(initial=0), t1_len.max(initial=0), t2_len.max(initial=0)))
            if T_global == 0:
                raise ValueError("After removing NaN rows, all samples have zero valid rows.")

            X_train, m_train = pad_to_length(train_list, train_len, T_global)
            X_T1, m_T1 = pad_to_length(t1_list, t1_len, T_global)
            X_T2, m_T2 = pad_to_length(t2_list, t2_len, T_global)

            # Target extraction
            ytr = to_float_1d(y_train[:, target_col_idx])
            yv = to_float_1d(y_T1[:, target_col_idx])
            yt = to_float_1d(y_T2[:, target_col_idx])

            # Drop samples where target is not finite (align X + mask)
            tr_ok = np.isfinite(ytr)
            v_ok = np.isfinite(yv)
            t_ok = np.isfinite(yt)

            X_train, m_train, ytr = X_train[tr_ok], m_train[tr_ok], ytr[tr_ok]
            X_T1, m_T1, yv = X_T1[v_ok], m_T1[v_ok], yv[v_ok]
            X_T2, m_T2, yt = X_T2[t_ok], m_T2[t_ok], yt[t_ok]

            # Normalize using train split only (exclude padded rows).
            mean, std = zscore_fit_masked(X_train, m_train)
            X_train = zscore_apply_masked(X_train, mean, std, m_train)
            X_T1 = zscore_apply_masked(X_T1, mean, std, m_T1)
            X_T2 = zscore_apply_masked(X_T2, mean, std, m_T2)

            input_shape = (X_train.shape[1], X_train.shape[2])
            print("Input shape (T,F):", input_shape)

            for model_family in MODEL_FAMILIES:
                print(f"\n#################### Model family: {model_family} ####################")

                task_results = []
                first_history = None
                first_res_T1 = None
                first_res_T2 = None
                first_tag = None

                model_tag = MODEL_SHORT.get(model_family, model_family)
                task_tag = "cls" if task == "classification" else "reg"
                tag_base = f"{TARGET_COL_NAMES.get(target_col_idx, str(target_col_idx))}_{task_tag}_{model_tag}_ab-{ablation_name}"

                for run_idx, seed in enumerate(SEEDS):
                    print(f"\n========== {model_family} | Ablation {ablation_name} | Seed {seed} ({run_idx + 1}/{len(SEEDS)}) ==========")
                    set_run_seed(seed)
                    run_batch_size = TRANSFORMER_BATCH_SIZE if model_family == "transformer" else BATCH_SIZE

                    thr = None

                    if task == "classification":
                        thr = get_threshold(ytr, method=THRESHOLD_METHOD)
                        ytr_fit = to_binary(ytr, thr)
                        yv_fit = to_binary(yv, thr)
                        ds_train = make_ds(X_train, ytr_fit, run_batch_size, shuffle=True)
                        ds_val = make_ds(X_T1, yv_fit, run_batch_size, shuffle=False)
                    else:
                        ytr_fit = ytr.astype(np.float32)
                        yv_fit = yv.astype(np.float32)
                        ds_train = make_ds(X_train, ytr_fit, run_batch_size, shuffle=True)
                        ds_val = make_ds(X_T1, yv_fit, run_batch_size, shuffle=False)

                    if model_family == "cnn":
                        model = build_cnn(input_shape, lr=LR, task=task)
                    elif model_family == "transformer":
                        model = build_transformer(input_shape, lr=LR, task=task)
                    else:
                        raise ValueError(f"Unknown model family: {model_family}")

                    tag = f"{tag_base}_s{seed}"

                    monitor = "val_acc" if task == "classification" else "val_loss"
                    mode = "max" if task == "classification" else "min"

                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(
                            monitor=monitor, mode=mode, patience=PATIENCE, restore_best_weights=True, verbose=1
                        ),
                    ]

                    history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=1)

                    if task == "classification":
                        yv_bin = to_binary(yv, thr)
                        yt_bin = to_binary(yt, thr)
                        res_T1 = evaluate_classification(
                            model, X_T1, yv_bin,
                            f"T1 (eval) {model_family} ablation={ablation_name} seed={seed}",
                            batch_size=run_batch_size
                        )
                        res_T2 = evaluate_classification(
                            model, X_T2, yt_bin,
                            f"T2 (test) {model_family} ablation={ablation_name} seed={seed}",
                            batch_size=run_batch_size
                        )
                        print(f"Threshold used (train-only median): {thr:.6f}")

                        row = {
                            "target_col_idx": target_col_idx,
                            "ablation_family": ablation_name,
                            "model_family": model_family,
                            "seed": seed,
                            "t2_acc": res_T2["acc"],
                            "t2_auc": res_T2["auc"],
                        }
                    else:
                        res_T1 = evaluate_regression(
                            model, X_T1, yv,
                            f"T1 (eval) {model_family} ablation={ablation_name} seed={seed}",
                            batch_size=run_batch_size
                        )
                        res_T2 = evaluate_regression(
                            model, X_T2, yt,
                            f"T2 (test) {model_family} ablation={ablation_name} seed={seed}",
                            batch_size=run_batch_size
                        )

                        row = {
                            "target_col_idx": target_col_idx,
                            "ablation_family": ablation_name,
                            "model_family": model_family,
                            "seed": seed,
                            "t1_nrmse": res_T1["nrmse"],
                            "t1_mae": res_T1["mae"],
                            "t2_nrmse": res_T2["nrmse"],
                            "t2_mae": res_T2["mae"],
                        }

                    task_results.append(row)

                    if run_idx == 0:
                        first_history = history
                        first_res_T1 = res_T1
                        first_res_T2 = res_T2
                        first_tag = tag

                if task == "classification":
                    t2_acc_mean = float(np.mean([r["t2_acc"] for r in task_results]))
                    t2_acc_std  = float(np.std([r["t2_acc"]  for r in task_results]))
                    t2_auc_mean = float(np.mean([r["t2_auc"] for r in task_results]))
                    t2_auc_std  = float(np.std([r["t2_auc"]  for r in task_results]))
                    print(f"Summary | ablation={ablation_name} | model={model_family} | T2 ACC={t2_acc_mean:.4f}±{t2_acc_std:.4f} | T2 AUC={t2_auc_mean:.4f}±{t2_auc_std:.4f}")
                    summary_rows.append({
                        "target_col_idx": target_col_idx,
                        "ablation_family": ablation_name,
                        "model_family":    model_family,
                        "t2_acc_mean":     t2_acc_mean,
                        "t2_acc_std":      t2_acc_std,
                        "t2_auc_mean":     t2_auc_mean,
                        "t2_auc_std":      t2_auc_std,
                    })
                else:
                    t2_nrmse_mean = float(np.mean([r["t2_nrmse"] for r in task_results]))
                    t2_nrmse_std  = float(np.std([r["t2_nrmse"]  for r in task_results]))
                    t2_mae_mean   = float(np.mean([r["t2_mae"]   for r in task_results]))
                    t2_mae_std    = float(np.std([r["t2_mae"]    for r in task_results]))
                    print(f"Summary | ablation={ablation_name} | model={model_family} | T2 NRMSE={t2_nrmse_mean:.4f}±{t2_nrmse_std:.4f} | T2 MAE={t2_mae_mean:.4f}±{t2_mae_std:.4f}")

                    summary_rows.append({
                        "target_col_idx": target_col_idx,
                        "ablation_family": ablation_name,
                        "model_family":    model_family,
                        "t2_nrmse_mean":   t2_nrmse_mean,
                        "t2_nrmse_std":    t2_nrmse_std,
                        "t2_mae_mean":     t2_mae_mean,
                        "t2_mae_std":      t2_mae_std,
                    })

                # Plot loss curves and performance only for baseline (no families dropped).
                if ablation_name == "none":
                    if first_history is not None and first_tag is not None:
                        plot_loss_curves(first_history, first_tag)
                    if first_res_T1 is not None and first_res_T2 is not None and first_tag is not None:
                        plot_performance(first_res_T1, first_res_T2, first_tag, task)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            baseline_df = summary_df[summary_df["ablation_family"] == "none"].copy()

            if task == "classification":
                summary_df = summary_df.merge(
                    baseline_df[["model_family", "t2_acc_mean", "t2_auc_mean"]],
                    on="model_family",
                    how="left",
                    suffixes=("", "_base"),
                )
                summary_df["delta_acc"] = summary_df["t2_acc_mean"] - summary_df["t2_acc_mean_base"]
                summary_df["delta_auc"] = summary_df["t2_auc_mean"] - summary_df["t2_auc_mean_base"]
                summary_df["rank"] = summary_df["t2_auc_mean"].rank(ascending=False, method="min").astype(int)
            else:
                summary_df = summary_df.merge(
                    baseline_df[["model_family", "t2_nrmse_mean", "t2_mae_mean"]],
                    on="model_family",
                    how="left",
                    suffixes=("", "_base"),
                )
                summary_df["delta_nrmse"] = summary_df["t2_nrmse_mean"] - summary_df["t2_nrmse_mean_base"]
                summary_df["delta_mae"] = summary_df["t2_mae_mean"] - summary_df["t2_mae_mean_base"]
                summary_df["rank"] = summary_df["t2_nrmse_mean"].rank(ascending=True, method="min").astype(int)

            summary_path = OUT_DIR / f"summary_{TARGET_COL_NAMES.get(target_col_idx, str(target_col_idx))}_{task}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved consolidated summary: {summary_path}")
########################################################################################################################
VIS_DIR       = OUT_DIR / "visualization"
VIS_PLOTS_DIR = VIS_DIR / "plots"
VIS_CLASS_PLOTS_DIR = VIS_PLOTS_DIR / "classification"

TARGET_LABEL = {v: v.upper() for v in TARGET_COL_NAMES.values()}

def _get_ordered_families(families):
    families = list(pd.unique(families))
    return ["none"] + sorted([f for f in families if f != "none"])

def _plot_heatmap_bar_combo(pivot_df, title, ylabel, out_png_path, bar_ylabel,
                             add_zero_line=False, value_format=".3f", figsize=(15, 5.5)):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax_h = axes[0]
    im = ax_h.imshow(pivot_df.values, aspect="auto")
    ax_h.set_title(f"{title} - Heatmap")
    ax_h.set_xlabel("Model Family")
    ax_h.set_ylabel(ylabel)
    ax_h.set_xticks(range(len(pivot_df.columns)))
    ax_h.set_xticklabels(pivot_df.columns, rotation=30, ha="right")
    ax_h.set_yticks(range(len(pivot_df.index)))
    ax_h.set_yticklabels(pivot_df.index)
    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            ax_h.text(j, i, format(pivot_df.iloc[i, j], value_format),
                      ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

    ax_b = axes[1]
    pivot_df.plot(kind="bar", ax=ax_b)
    ax_b.set_title(f"{title} - Bar")
    ax_b.set_xlabel("Ablation Family")
    ax_b.set_ylabel(bar_ylabel)
    ax_b.tick_params(axis="x", rotation=45)
    for tick in ax_b.get_xticklabels():
        tick.set_ha("right")
    if add_zero_line:
        ax_b.axhline(0, color="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png_path}")

def generate_visualizations():
    VIS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_CLASS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    reg_metrics = [
        ("t2_nrmse_mean", "NRMSE",   False, "raw_nrmse"),
        ("t2_mae_mean",   "MAE",     False, "raw_mae"),
        ("delta_nrmse",   "Δ NRMSE", True,  "delta_nrmse"),
        ("delta_mae",     "Δ MAE",   True,  "delta_mae"),
    ]
    cls_metrics = [
        ("t2_auc_mean", "AUC",   False, "raw_auc"),
        ("t2_acc_mean", "ACC",   False, "raw_acc"),
        ("delta_auc",   "Δ AUC", True,  "delta_auc"),
        ("delta_acc",   "Δ ACC", True,  "delta_acc"),
    ]

    for target_name in TARGET_COL_NAMES.values():
        for task, metrics, plots_dir in [
            ("regression",     reg_metrics, VIS_PLOTS_DIR),
            ("classification", cls_metrics, VIS_CLASS_PLOTS_DIR),
        ]:
            path = OUT_DIR / f"summary_{target_name}_{task}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            ordered = _get_ordered_families(df["ablation_family"])

            for metric, bar_ylabel, add_zero, suffix in metrics:
                if metric not in df.columns:
                    continue
                pivot = (
                    df.pivot(index="ablation_family", columns="model_family", values=metric)
                    .reindex(ordered)
                )
                _plot_heatmap_bar_combo(
                    pivot,
                    title=f"{target_name.upper()} - {bar_ylabel}",
                    ylabel="Ablation Family",
                    out_png_path=plots_dir / f"{target_name}_{suffix}.png",
                    bar_ylabel=bar_ylabel,
                    add_zero_line=add_zero,
                )

if __name__ == "__main__":
    for _task in ("regression", "classification"):
        print(f"\n\n==================== RUN TASK: {_task.upper()} ====================")
        main(task_override=_task)

    print("\n\n==================== GENERATING VISUALIZATIONS ====================")
    generate_visualizations()
    print("All done.")




