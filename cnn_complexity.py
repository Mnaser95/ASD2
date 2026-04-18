"""
cnn_complexity.py — CNN capacity scaling experiment.

Trains three CNN variants of increasing complexity on the same data/slices
as modeling2.py, to show performance plateaus with larger models.

CNN variants (all share the same input shape: (N_PICK, N_FEATURES)):
  small  : Conv1D(64)  → MaxPool → Conv1D(128) → Dense(128) → Drop(0.3) → Dense(64)
  medium : Conv1D(128) → MaxPool → Conv1D(256) → Dense(256) → Drop(0.3) → Dense(128)
  large  : Conv1D(128) → MaxPool → Conv1D(256) → MaxPool → Conv1D(256)
           → Dense(512) → Drop(0.4) → Dense(256) → Drop(0.3) → Dense(128)

Outputs: <ROOT>/outputs/cnn_complexity/
  summary.csv      — per-variant metrics averaged across slices and targets
  complexity.png   — ACC + param count vs variant
"""

from data_import import load_all_data
from pathlib import Path
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score
from gpu_utils import require_accelerators, configure_tensorflow_gpu, log_runtime_gpu_status

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

N_FEATURES = 48
N_PICK     = 20
N_SLICES   = 20

TARGET_COL_IDXS  = [4, 5]
TARGET_COL_NAMES = {4: "sa", 5: "rrb"}

BATCH_SIZE = 64
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 20

CODE_DIR = Path(__file__).resolve().parent
ROOT     = CODE_DIR.parent
TRAIN_MAT  = ROOT / r"data\training\train_data.mat"
TRAIN_IDS  = ROOT / r"data\training\ids_fixed.mat"
TRAIN_XLSX = ROOT / r"data\training\data_train.xlsx"
TEST_DIR   = ROOT / r"data\testing"
OUT_DIR    = ROOT / "outputs" / "cnn_complexity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

########################################################################################################################
# DATA PIPELINE (identical to modeling2.py)

def flatten_and_clean(X_raw):
    N = X_raw.shape[0]
    X = X_raw.reshape(N, -1, X_raw.shape[-1])[:, :, :N_FEATURES]
    result = []
    for i in range(N):
        s = X[i]
        result.append(s[np.isfinite(s).all(axis=1)].astype(np.float32))
    return result

def sample_raw(X_clean, n_pick, rng):
    N   = len(X_clean)
    out = np.full((N, n_pick, N_FEATURES), np.nan, dtype=np.float32)
    for i, s in enumerate(X_clean):
        if s.shape[0] == 0:
            continue
        idx    = rng.choice(s.shape[0], size=n_pick, replace=s.shape[0] < n_pick)
        out[i] = s[idx]
    return out

def prepare_split(X_clean, y_raw, target_col_idx, rng):
    X_seq = sample_raw(X_clean, N_PICK, rng)
    X_avg = np.nanmean(X_seq, axis=1)
    y     = pd.to_numeric(pd.Series(y_raw[:, target_col_idx]), errors="coerce").to_numpy(np.float32)
    ok    = np.isfinite(y) & np.isfinite(X_avg).all(axis=1)
    return X_seq[ok], y[ok]

def zscore_fit(X):
    v = X[np.isfinite(X).all(axis=1)]
    return v.mean(axis=0).astype(np.float32), (v.std(axis=0) + 1e-8).astype(np.float32)

def zscore_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def get_thresholds(y):
    return float(np.percentile(y, 50)),

def to_2class(y, t1):
    return (np.asarray(y, np.float32) >= t1).astype(np.int32)

def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 5000), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def reset_seed():
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.keras.utils.set_random_seed(SEED)
    except AttributeError:
        pass

########################################################################################################################
# CNN VARIANTS

CNN_VARIANTS = {
    "small": {
        "conv_layers": [(64, 3), (128, 3)],
        "pool_after":  [0],           # pool after layer index 0
        "dense_layers": [128, 64],
        "dropouts":    {0: 0.3},      # dropout after dense index 0
    },
    "medium": {
        "conv_layers": [(128, 3), (256, 3)],
        "pool_after":  [0],
        "dense_layers": [256, 128],
        "dropouts":    {0: 0.3},
    },
    "large": {
        "conv_layers": [(128, 3), (256, 3), (256, 3)],
        "pool_after":  [0, 1],
        "dense_layers": [512, 256, 128],
        "dropouts":    {0: 0.4, 1: 0.3},
    },
}

def build_variant(name, input_shape, task="classification"):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.initializers import RandomUniform

    cfg  = CNN_VARIANTS[name]
    init = RandomUniform(seed=1)
    n_out, out_act = (2, "softmax") if task == "classification" else (1, "linear")

    layers = [tf.keras.layers.InputLayer(input_shape=input_shape)]
    for idx, (filters, kernel) in enumerate(cfg["conv_layers"]):
        layers.append(Conv1D(filters, kernel, activation="relu", padding="same", kernel_initializer=init))
        if idx in cfg["pool_after"]:
            layers.append(MaxPool1D(pool_size=2))
    layers.append(Flatten())
    for idx, units in enumerate(cfg["dense_layers"]):
        layers.append(Dense(units, activation="relu", kernel_initializer=init))
        if idx in cfg["dropouts"]:
            layers.append(Dropout(cfg["dropouts"][idx]))
    layers.append(Dense(n_out, activation=out_act, kernel_initializer=init))

    model = Sequential(layers)
    if task == "classification":
        model.compile(optimizer=RMSprop(LR), loss="sparse_categorical_crossentropy",
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    else:
        model.compile(optimizer=RMSprop(LR), loss="mse")
    return model

def count_params(name, input_shape, task="classification"):
    m = build_variant(name, input_shape, task)
    n = int(m.count_params())
    tf.keras.backend.clear_session()
    return n

########################################################################################################################
# MAIN

def main():
    require_accelerators(require_gpu=False, require_npu=False)
    configure_tensorflow_gpu(verbose=True)
    log_runtime_gpu_status("cnn_complexity.py")

    X_train_raw, y_train_raw, X_T1_raw, y_T1_raw, X_T2_raw, y_T2_raw = load_all_data(
        train_mat_path=str(TRAIN_MAT), train_ids_mat_path=str(TRAIN_IDS),
        train_xlsx_path=str(TRAIN_XLSX), test_dir=str(TEST_DIR),
    )

    X_train_clean = flatten_and_clean(X_train_raw)
    X_T1_clean    = flatten_and_clean(X_T1_raw)
    X_T2_clean    = flatten_and_clean(X_T2_raw)

    input_shape = (N_PICK, N_FEATURES)
    param_counts = {v: count_params(v, input_shape) for v in CNN_VARIANTS}
    print("Parameter counts:", param_counts)

    summary_rows = []

    for target_col_idx in TARGET_COL_IDXS:
        target_name = TARGET_COL_NAMES[target_col_idx]
        print(f"\n==================== Target: {target_name} ====================")

        for variant_name in CNN_VARIANTS:
            print(f"\n---------- Variant: {variant_name} ----------")

            slice_accs   = []
            slice_f1s    = []
            slice_kappas = []
            slice_trains = []

            for slice_idx in range(N_SLICES):
                reset_seed()
                rng_tr = np.random.default_rng(SEED + slice_idx * 3)
                rng_v  = np.random.default_rng(SEED + slice_idx * 3 + 1)
                rng_t  = np.random.default_rng(SEED + slice_idx * 3 + 2)

                X_tr, y_tr = prepare_split(X_train_clean, y_train_raw, target_col_idx, rng_tr)
                X_v,  y_v  = prepare_split(X_T1_clean,    y_T1_raw,    target_col_idx, rng_v)
                X_t,  y_t  = prepare_split(X_T2_clean,    y_T2_raw,    target_col_idx, rng_t)

                if len(y_tr) == 0:
                    continue

                mean, std = zscore_fit(np.nanmean(X_tr, axis=1))
                X_tr = zscore_apply(X_tr, mean, std)
                X_v  = zscore_apply(X_v,  mean, std)
                X_t  = zscore_apply(X_t,  mean, std)

                (t1,)   = get_thresholds(y_tr)
                ytr_lbl = to_2class(y_tr, t1)
                yv_lbl  = to_2class(y_v,  t1)
                yt_lbl  = to_2class(y_t,  t1)

                model    = build_variant(variant_name, input_shape)
                ds_train = make_ds(X_tr, ytr_lbl, shuffle=True)
                ds_val   = make_ds(X_v,  yv_lbl,  shuffle=False)

                t0 = time.perf_counter()
                model.fit(
                    ds_train, validation_data=ds_val, epochs=EPOCHS,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor="val_acc", mode="max", patience=PATIENCE,
                        restore_best_weights=True, verbose=0)],
                    verbose=0,
                )
                train_time = time.perf_counter() - t0

                y_prob = model.predict(X_t, batch_size=BATCH_SIZE, verbose=0)
                y_hat  = np.argmax(y_prob, axis=1).astype(np.int32)
                acc    = float(np.mean(y_hat == yt_lbl))
                f1     = float(f1_score(yt_lbl, y_hat, average="macro", zero_division=0))
                kappa  = float(cohen_kappa_score(yt_lbl, y_hat))

                print(f"  Slice {slice_idx+1}: ACC={acc:.4f} F1={f1:.4f} κ={kappa:.4f} t={train_time:.1f}s")
                slice_accs.append(acc)
                slice_f1s.append(f1)
                slice_kappas.append(kappa)
                slice_trains.append(train_time)

            if not slice_accs:
                continue

            summary_rows.append({
                "target":        target_name,
                "variant":       variant_name,
                "n_params":      param_counts[variant_name],
                "acc_mean":      float(np.mean(slice_accs)),
                "acc_std":       float(np.std(slice_accs)),
                "f1_macro_mean": float(np.mean(slice_f1s)),
                "f1_macro_std":  float(np.std(slice_f1s)),
                "kappa_mean":    float(np.mean(slice_kappas)),
                "kappa_std":     float(np.std(slice_kappas)),
                "train_time_mean": float(np.mean(slice_trains)),
                "train_time_std":  float(np.std(slice_trains)),
            })
            print(f"  [{variant_name}] ACC={summary_rows[-1]['acc_mean']:.4f}±{summary_rows[-1]['acc_std']:.4f} "
                  f"F1={summary_rows[-1]['f1_macro_mean']:.4f} κ={summary_rows[-1]['kappa_mean']:.4f}")

    # ---- save CSV ----
    df = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ---- plot ----
    variant_order = list(CNN_VARIANTS.keys())
    colors        = ["steelblue", "darkorange", "seagreen"]
    fig, axes     = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, ylabel) in zip(axes, [
        ("acc_mean",      "Accuracy"),
        ("f1_macro_mean", "F1 (macro)"),
        ("kappa_mean",    "Cohen's κ"),
    ]):
        for target_name in TARGET_COL_NAMES.values():
            sub    = df[df["target"] == target_name]
            sub    = sub.set_index("variant").reindex(variant_order)
            means  = sub[metric].values
            ax.plot(variant_order, means, marker="o", label=target_name)

        # annotate param counts on x-axis
        ax.set_xticks(range(len(variant_order)))
        ax.set_xticklabels(
            [f"{v}\n({param_counts[v]:,} params)" for v in variant_order],
            fontsize=9)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle("CNN Capacity Scaling — Classification (2-class)", fontsize=13)
    fig.tight_layout()
    plot_path = OUT_DIR / "complexity.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")


########################################################################################################################
if __name__ == "__main__":
    main()
    print("All done.")
