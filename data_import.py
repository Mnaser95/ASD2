import os, glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from gpu_utils import require_accelerators, log_runtime_gpu_status

def load_all_data(
    train_mat_path,
    train_ids_mat_path,
    train_xlsx_path,
    test_dir,
    feats_take=49,
    num_mats=10
):
    require_accelerators(require_gpu=False, require_npu=False)
    log_runtime_gpu_status("data_import.load_all_data")

    # ---------- helpers ----------
    def clean_ids(s):
        return pd.Series(s).astype(str).str.strip().str.replace(".mat", "", regex=False).tolist()

    def cell10_to_arr(cell):
        return np.stack([np.array(cell[i, 0], dtype=np.float32) for i in range(10)], axis=0)

    # =========================
    # TRAINING
    # =========================
    data = loadmat(train_mat_path, squeeze_me=True, struct_as_record=False)
    ids  = loadmat(train_ids_mat_path, squeeze_me=True, struct_as_record=False)

    rec_ids_train = clean_ids(ids["rec_id"])

    df_train = pd.read_excel(train_xlsx_path)
    df_train["rec_id"] = clean_ids(df_train["rec_id"])
    df_train = df_train[df_train["rec_id"].isin(rec_ids_train)]
    df_train["rec_id"] = pd.Categorical(df_train["rec_id"], rec_ids_train, ordered=True)
    df_train = df_train.sort_values("rec_id").reset_index(drop=True)

    feats_all = data["features"]
    feats = np.asarray([feats_all[k][:, :feats_take] for k in range(len(feats_all))], dtype=np.float32)
    X_all = feats.reshape(-1, num_mats, feats.shape[1], feats.shape[2])

    idx_map = {rid: i for i, rid in enumerate(rec_ids_train)}
    X_train = X_all[[idx_map[rid] for rid in df_train["rec_id"]]]
    y_train = df_train.to_numpy()

    # =========================
    # TESTING (T1 / T2)
    # =========================
    mats_dir = os.path.join(test_dir, "data")
    mat_map = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(mats_dir, "*.mat"))}

    def load_test_split(xlsx_path):
        df = pd.read_excel(xlsx_path)
        df["rec_id"] = clean_ids(df["rec_id"])
        df = df[df["rec_id"].isin(mat_map.keys())].reset_index(drop=True)

        X = np.stack([
            cell10_to_arr(
                loadmat(mat_map[rid], squeeze_me=False, struct_as_record=False)[
                    [k for k, v in loadmat(mat_map[rid], squeeze_me=False, struct_as_record=False).items()
                     if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype == object][0]
                ]
            )
            for rid in df["rec_id"]
        ], axis=0)

        y = df.to_numpy()
        return X, y

    X_T1, y_T1 = load_test_split(os.path.join(test_dir, "data_T1.xlsx"))
    X_T2, y_T2 = load_test_split(os.path.join(test_dir, "data_T2.xlsx"))

    return X_train, y_train, X_T1, y_T1, X_T2, y_T2

