import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 0. Paths / configuration
# ============================

# Root directory that contains all experiment subfolders
# Example structure:
# D:/These/Graphics-LPIPS/out/TMQ/New_Render/Original/TMQ_NR_4VP_org_kfolds/4VP/correlation_summary_kfolds.csv
BASE_RESULTS_DIR = r"D:\These\Graphics-LPIPS\out"

# Name of the recap file written by correlation_VP.py
SUMMARY_FILENAME = "correlation_summary_kfolds.csv"

# Column to use for the heatmaps (PLCC = Pearson)
METRIC_COL = "pearson_mean"


# ============================
# 1. Helpers to read summaries
# ============================

def load_correlation_summaries(root_dir, summary_filename=SUMMARY_FILENAME):
    """Recursively scan root_dir and concatenate all summary CSVs."""
    frames = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if summary_filename in filenames:
            full_path = os.path.join(dirpath, summary_filename)
            df = pd.read_csv(full_path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No '{summary_filename}' found under {root_dir}")
    summary = pd.concat(frames, ignore_index=True)
    return summary


def training_alias_from_model(model):
    """Map model name (training config) to a short alias used in the heatmaps."""
    m = str(model).upper()
    if "TMQ_OR_1VP" in m:
        return "TMQ-O-1V"
    if "TMQ_OR_4VP" in m:
        return "TMQ-O-4V"
    if "TMQ_NR_1VP" in m:
        return "TMQ-N-1V"
    if "TMQ_NR_4VP" in m:
        return "TMQ-N-4V"
    if "TSMD_NR_1VP" in m:
        return "TSMD-1V"
    if "TSMD_NR_4VP" in m:
        return "TSMD-4V"
    # Baseline Yana / Graphics-LPIPS original network
    if "GRAPHICSLPIPS" in m or "YANA" in m:
        return "Yana-Original"
    return None


def test_alias_from_row(row):
    """Map (database, render_method, view_method, testing_views) to heatmap alias."""
    db = str(row.get("database", "")).upper()
    rm = str(row.get("render_method", "")).upper()
    vm = str(row.get("view_method", "")).upper()
    v = int(row.get("testing_views", 0))

    # TMQ - Old render
    if db == "TMQ" and rm == "OLD_RENDER" and vm == "ORIGINAL":
        return f"TMQ-O-{v}V"

    # TMQ - New render
    if db == "TMQ" and rm == "NEW_RENDER" and vm == "ORIGINAL":
        return f"TMQ-N-{v}V"

    # TSMD - New render + Y_fixed_0.3
    if db == "TSMD" and rm == "NEW_RENDER" and vm.startswith("Y_FIXED"):
        return f"TSMD-{v}V"

    return None


def build_matrix(agg, train_labels, test_labels):
    """Build a matrix [train x test] from a MultiIndex Series agg[(train_alias, test_alias)]."""
    mat = np.full((len(train_labels), len(test_labels)), np.nan, dtype=float)
    for i, tr in enumerate(train_labels):
        for j, te in enumerate(test_labels):
            key = (tr, te)
            if key in agg.index:
                mat[i, j] = agg.loc[key]
    return mat


# ============================
# 2. Load all results
# ============================

summary = load_correlation_summaries(BASE_RESULTS_DIR, SUMMARY_FILENAME)

# Derive aliases
summary["train_alias"] = summary["model"].apply(training_alias_from_model)
summary["test_alias"] = summary.apply(test_alias_from_row, axis=1)

# Keep only rows we can map to aliases
summary_valid = summary.dropna(subset=["train_alias", "test_alias"])

# Aggregate (mean in case multiple runs exist for same (train, test))
agg = summary_valid.groupby(["train_alias", "test_alias"])[METRIC_COL].mean()

# ============================
# 3. Intra-TMQ heatmap (PLCC)
# ============================

configs_tmq = [
    "TMQ-O-1V",
    "TMQ-O-4V",
    "TMQ-N-1V",
    "TMQ-N-4V",
]

train_labels_tmq = ["Yana-Original"] + configs_tmq
test_labels_tmq = configs_tmq

data_tmq = build_matrix(agg, train_labels_tmq, test_labels_tmq)
results_tmq = pd.DataFrame(data_tmq, index=train_labels_tmq, columns=test_labels_tmq)

sns.set_theme(style="white", font_scale=1.0)
plt.figure(figsize=(6, 5))

ax = sns.heatmap(
    results_tmq,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    vmin=np.nanmin(results_tmq.values),
    vmax=np.nanmax(results_tmq.values),
)
ax.set_title("Intra-TMQ - PLCC", pad=12)
ax.set_xlabel("Test")
ax.set_ylabel("Entraînement")

plt.tight_layout()
plt.show()
# plt.savefig("heatmap_intra_TMQ_PLCC.png", dpi=300, bbox_inches="tight")


# ============================
# 4. Intra-TSMD heatmap (PLCC)
# ============================

configs_tsmd = ["TSMD-1V", "TSMD-4V"]

train_labels_tsmd = configs_tsmd
test_labels_tsmd = configs_tsmd

data_tsmd = build_matrix(agg, train_labels_tsmd, test_labels_tsmd)
results_tsmd = pd.DataFrame(data_tsmd, index=train_labels_tsmd, columns=test_labels_tsmd)

sns.set_theme(style="white", font_scale=1.0)
plt.figure(figsize=(4, 3.5))

ax = sns.heatmap(
    results_tsmd,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    vmin=np.nanmin(results_tsmd.values),
    vmax=np.nanmax(results_tsmd.values),
)
ax.set_title("Intra-TSMD (New) - PLCC", pad=12)
ax.set_xlabel("Test")
ax.set_ylabel("Entraînement")

plt.tight_layout()
plt.show()
# plt.savefig("heatmap_intra_TSMD_PLCC.png", dpi=300, bbox_inches="tight")


# ============================
# 5. Cross-base New Render heatmap (PLCC)
# ============================

configs_new = [
    "TMQ-N-1V",
    "TMQ-N-4V",
    "TSMD-1V",
    "TSMD-4V",
]

train_labels_cross = configs_new
test_labels_cross = configs_new

data_cross = build_matrix(agg, train_labels_cross, test_labels_cross)
results_cross = pd.DataFrame(data_cross, index=train_labels_cross, columns=test_labels_cross)

sns.set_theme(style="white", font_scale=1.0)
plt.figure(figsize=(6, 5))

ax = sns.heatmap(
    results_cross,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    vmin=np.nanmin(results_cross.values),
    vmax=np.nanmax(results_cross.values),
)
ax.set_title("Cross-base (New Render) - PLCC", pad=12)
ax.set_xlabel("Test")
ax.set_ylabel("Entraînement")

plt.tight_layout()
plt.show()
# plt.savefig("heatmap_crossbase_New_PLCC.png", dpi=300, bbox_inches="tight")
