"""
visualize_modeling2.py — generate plots from the CSV summaries produced by modeling2.py.

Expected CSVs in <ROOT>/outputs/modeling2/:
    summary_{target}_{task}.csv   (target: ados/sa/rrb, task: regression/classification)

Outputs go to <ROOT>/outputs/modeling2/visualization/
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TARGET_COL_NAMES = {3: "ados", 4: "sa", 5: "rrb"}

CODE_DIR = Path(__file__).resolve().parent
ROOT     = CODE_DIR.parent
OUT_DIR  = ROOT / "outputs" / "modeling2"
VIS_DIR  = OUT_DIR / "visualization"

########################################################################################################################

def _get_ordered_families(families):
    families = list(pd.unique(families))
    return ["none"] + sorted([f for f in families if f != "none"])


def _draw_metric_on_ax(ax, df, metric, title, ylabel):
    """Draw a metric line plot with delta annotations onto an existing ax."""
    ordered   = _get_ordered_families(df["ablation_family"])
    pivot_raw = df.pivot(index="ablation_family", columns="model_family",
                         values=f"t2_{metric}_mean").reindex(ordered)
    pivot_dlt = df.pivot(index="ablation_family", columns="model_family",
                         values=f"delta_{metric}").reindex(ordered)
    models = list(pivot_raw.columns)
    colors = ["steelblue", "seagreen", "darkorange", "crimson"][:len(models)]
    x      = list(range(len(ordered)))

    for i, model in enumerate(models):
        vals   = pivot_raw[model].values
        deltas = pivot_dlt[model].values
        ax.plot(x, vals, marker="o", markersize=16, color=colors[i], linewidth=8, label=model)
        baseline_val = vals[0]
        if baseline_val == baseline_val:  # not NaN
            ax.axhline(baseline_val, color=colors[i], linewidth=1.5,
                       linestyle="--", alpha=0.4)
        for j, (v, d) in enumerate(zip(vals, deltas)):
            if j == 0:
                continue
            if not (v != v) and not (d != d):
                ax.annotate(f"{d:+.3f}", xy=(j, v), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=28, color=colors[i])

    ax.set_ylabel(ylabel, fontsize=40)
    ax.set_xlabel("Ablation Family", fontsize=40)
    ax.tick_params(axis="both", labelsize=32)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=0, ha="right", fontsize=32)
    ax.legend(title="Model", loc="best", fontsize=32, title_fontsize=32)
    ax.set_title(f"{title}", fontsize=40)


def _plot_timing_combined(df_reg, df_cls, target_name, out_png_path):
    """
    Single plot: regression and classification timing bars side by side,
    with a vertical separator between the two groups.
    """
    def _extract(df):
        if df is None:
            return [], [], [], []
        base = df[df["ablation_family"] == "none"].copy()
        if base.empty:
            return [], [], [], []
        models      = list(base["model_family"])
        train_means = base["train_time_mean"].values
        train_stds  = base["train_time_std"].values if "train_time_std" in base.columns else [0] * len(base)
        infer_means = base["infer_time_mean"].values
        return models, train_means, train_stds, infer_means

    reg_models, reg_tr, reg_std, reg_inf = _extract(df_reg)
    cls_models, cls_tr, cls_std, cls_inf = _extract(df_cls)

    if not reg_models and not cls_models:
        return

    colors  = ["steelblue", "seagreen", "darkorange", "crimson"]
    bar_w   = 0.15
    gap     = 1.0   # gap between regression and classification groups

    # build x positions: reg group at 0..n_reg-1, cls group at n_reg+gap..
    n_reg = len(reg_models)
    n_cls = len(cls_models)
    reg_x = list(range(n_reg))
    cls_x = [n_reg + gap + i for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(37.5, 18.75))

    def _annotate_bars(ax, x, tr, std, inf, c):
        ax.annotate(f"{tr:.2f}s", xy=(x - bar_w/2, tr + std),
                    xytext=(-16, 16), textcoords="offset points",
                    ha="right", va="bottom", fontsize=62, color=c)
        ax.annotate(f"{inf:.3f}s", xy=(x + bar_w/2, inf),
                    xytext=(16, 16), textcoords="offset points",
                    ha="left", va="bottom", fontsize=62, color=c)

    for i, (model, c) in enumerate(zip(reg_models, colors)):
        ax.bar(reg_x[i] - bar_w/2, reg_tr[i], width=bar_w, color=c,
               alpha=0.85, yerr=reg_std[i], capsize=16, label=f"{model} train")
        ax.bar(reg_x[i] + bar_w/2, reg_inf[i], width=bar_w, color=c,
               alpha=0.4, label=f"{model} infer")
        _annotate_bars(ax, reg_x[i], reg_tr[i], reg_std[i], reg_inf[i], c)

    for i, (model, c) in enumerate(zip(cls_models, colors)):
        ax.bar(cls_x[i] - bar_w/2, cls_tr[i], width=bar_w, color=c,
               alpha=0.85, yerr=cls_std[i], capsize=16)
        ax.bar(cls_x[i] + bar_w/2, cls_inf[i], width=bar_w, color=c, alpha=0.4)
        _annotate_bars(ax, cls_x[i], cls_tr[i], cls_std[i], cls_inf[i], c)

    # separator line between groups
    if reg_models and cls_models:
        sep_x = (reg_x[-1] + cls_x[0]) / 2
        ax.axvline(sep_x, color="gray", linewidth=4, linestyle="--", alpha=0.7)

    all_x   = reg_x + cls_x
    all_lbl = reg_models + cls_models
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_lbl, rotation=0, ha="right", fontsize=62)

    ax.tick_params(axis="both", labelsize=62, width=2, length=6)
    ax.set_ylabel("Time (s)", fontsize=78)
    ax.legend(fontsize=62)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    fig.tight_layout()

    # group labels drawn after tight_layout so positions are stable
    if reg_models:
        mid = sum(reg_x) / len(reg_x)
        ax.annotate("Regression", xy=(mid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -234), textcoords="offset points",
                    ha="center", fontsize=62, color="dimgray", style="italic",
                    annotation_clip=False)
    if cls_models:
        mid = sum(cls_x) / len(cls_x)
        ax.annotate("Classification", xy=(mid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -234), textcoords="offset points",
                    ha="center", fontsize=62, color="dimgray", style="italic",
                    annotation_clip=False)
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png_path}")


def generate_visualizations():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    for target_name in TARGET_COL_NAMES.values():
        path_reg = OUT_DIR / f"summary_{target_name}_regression.csv"
        path_cls = OUT_DIR / f"summary_{target_name}_classification.csv"

        df_reg = pd.read_csv(path_reg) if path_reg.exists() else None
        df_cls = pd.read_csv(path_cls) if path_cls.exists() else None

        has_reg = df_reg is not None and "t2_nrmse_mean" in df_reg.columns and "delta_nrmse" in df_reg.columns
        has_cls = df_cls is not None and "t2_acc_mean"   in df_cls.columns and "delta_acc"   in df_cls.columns

        if has_reg or has_cls:
            n_rows = int(has_reg) + int(has_cls)
            fig, axes = plt.subplots(n_rows, 1, figsize=(28, 12 * n_rows))
            if n_rows == 1:
                axes = [axes]
            ax_idx = 0
            if has_reg:
                _draw_metric_on_ax(axes[ax_idx], df_reg, "nrmse",
                                   target_name.upper(), "NRMSE")
                ax_idx += 1
            if has_cls:
                _draw_metric_on_ax(axes[ax_idx], df_cls, "acc",
                                   target_name.upper(), "ACC")
            fig.tight_layout()
            out_path = VIS_DIR / f"{target_name}_metrics.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

        has_reg_timing = df_reg is not None and "train_time_mean" in df_reg.columns

        if has_reg_timing:
            _plot_timing_combined(
                df_reg,
                None,
                target_name,
                VIS_DIR / f"{target_name}_timing.png",
            )


########################################################################################################################
if __name__ == "__main__":
    generate_visualizations()
    print("All done.")
