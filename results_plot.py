import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 0. Paths / configuration
# ============================

BASE_RESULTS_DIR = r"D:\These\Graphics-LPIPS\out"
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


# ============================
# 2. Alias helpers
# ============================

def training_alias_from_model(model):
    m = str(model).upper()

    if "GRAPHICSLPIPS" in m or "YANA" in m:
        return "Yana-Original"

    naa = ("_NAA" in m) or ("-NAA" in m) or ("NAA_" in m) or m.endswith("NAA") or ("_XAA" in m)

    # Longest first to avoid matching TMQ before TMQ-SJTU
    pattern = r"(TMQ(?:-|_)SJTU|SJTU-TMQA|TMQ|TSMD)_(OR|NR)_(\d+)VP_([A-Z0-9]+)"
    match = re.search(pattern, m)
    if not match:
        return None

    db, render_tag, nviews, view_tag = match.groups()

    # Normalize
    db = db.replace("_", "-")

    render_short = "O" if render_tag == "OR" else "N" if render_tag == "NR" else render_tag

    if view_tag.startswith("ORG"):
        view_short = "Org"
    elif view_tag.startswith("YF"):
        view_short = "YF"
    elif view_tag.startswith("FIB"):
        view_short = "Fib"
    else:
        view_short = view_tag

    if naa:
        view_short += "-NAA"

    return f"{db}-{render_short}-{view_short}-{int(nviews)}V"


def test_alias_from_row(row):
    db = str(row.get("database", "")).upper()
    rm = str(row.get("render_method", "")).upper()
    vm = str(row.get("view_method", "")).upper()
    v = int(row.get("testing_views", 0))

    if db not in {"TMQ", "TSMD", "SJTU-TMQA", "TMQ-SJTU"}:
        return None

    if rm == "OLD_RENDER":
        render_short = "O"
    elif rm == "NEW_RENDER":
        render_short = "N"
    else:
        return None

    if vm == "ORIGINAL":
        view_short = "Org"
    elif vm.startswith("Y_FIXED"):
        view_short = "YF"
    elif vm.startswith("FIBONACCI"):
        view_short = "Fib"
    else:
        return None

    naa = ("NAA" in vm) or ("_XAA" in vm)
    if naa:
        view_short += "-NAA"

    return f"{db}-{render_short}-{view_short}-{v}V"


def build_matrix(agg, train_labels, test_labels):
    """Build a matrix [train x test] from a MultiIndex Series agg[(train_alias, test_alias)]."""
    mat = np.full((len(train_labels), len(test_labels)), np.nan, dtype=float)
    for i, tr in enumerate(train_labels):
        for j, te in enumerate(test_labels):
            key = (tr, te)
            if key in agg.index:
                mat[i, j] = agg.loc[key]
    return mat


def plot_heatmap(data, train_labels, test_labels, title, figsize=(6, 5), vmin=None, vmax=None):
    """Helper to plot a single heatmap."""
    df = pd.DataFrame(data, index=train_labels, columns=test_labels)
    sns.set_theme(style="white", font_scale=1.0)
    plt.figure(figsize=figsize)

    # Auto vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanmin(df.values)
    if vmax is None:
        vmax = np.nanmax(df.values)

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Test")
    ax.set_ylabel("Training")

    plt.tight_layout()
    plt.show()
    # You can replace by plt.savefig(...) if needed


# ============================
# 3. Load all results
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
# 4. Intra-TMQ - Orginal
# ============================

# Training configs: Yana baseline + TMQ Old/New with Orginal viewpoint (1 and 4 views)
train_labels_tmq_Org = [
    "Yana-Original",
    "TMQ-O-Org-1V",
    "TMQ-O-Org-4V",
    "TMQ-N-Org-1V",
    "TMQ-N-Org-4V",
]

test_labels_tmq_Org = [
    "TMQ-O-Org-1V",
    "TMQ-O-Org-4V",
    "TMQ-N-Org-1V",
    "TMQ-N-Org-4V",
]

data_tmq_Org = build_matrix(agg, train_labels_tmq_Org, test_labels_tmq_Org)
plot_heatmap(
    data_tmq_Org,
    train_labels_tmq_Org,
    test_labels_tmq_Org,
    title="Intra-TMQ - Orginal viewpoints - PLCC",
    figsize=(6, 5),
)


# ============================
# 5. Intra-TMQ - Y_fixed_0.3
# ============================

# Adapte cette liste selon les runs que tu as réellement:
# Par exemple, tu peux avoir 4, 8, 16 vues pour TMQ New Render + Y_fixed_0.3
train_labels_tmq_yf = [
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",
    "TMQ-N-YF-16V",
]

test_labels_tmq_yf = train_labels_tmq_yf

data_tmq_yf = build_matrix(agg, train_labels_tmq_yf, test_labels_tmq_yf)
plot_heatmap(
    data_tmq_yf,
    train_labels_tmq_yf,
    test_labels_tmq_yf,
    title="Intra-TMQ - Y_fixed_0.3 - PLCC",
    figsize=(5, 4),
)


# ============================
# 6. Intra-TMQ - Fibonacci
# ============================

# Idem, adapte cette liste aux configs existantes
train_labels_tmq_fib = [
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-8V",
    "TMQ-N-Fib-16V",
]

test_labels_tmq_fib = train_labels_tmq_fib

data_tmq_fib = build_matrix(agg, train_labels_tmq_fib, test_labels_tmq_fib)
plot_heatmap(
    data_tmq_fib,
    train_labels_tmq_fib,
    test_labels_tmq_fib,
    title="Intra-TMQ - Fibonacci - PLCC",
    figsize=(5, 4),
)


# ============================
# 7. Intra-TSMD - Y_fixed_0.3
# ============================

# TSMD: New render + Y_fixed_0.3 uniquement
# Ajoute ici 8V/16V si tu as ces runs
train_labels_tsmd_yf = [
    "TSMD-N-YF-1V",
    "TSMD-N-YF-4V",
    "TSMD-N-YF-8V",
    "TSMD-N-YF-16V",
]

test_labels_tsmd_yf = train_labels_tsmd_yf

data_tsmd_yf = build_matrix(agg, train_labels_tsmd_yf, test_labels_tsmd_yf)
plot_heatmap(
    data_tsmd_yf,
    train_labels_tsmd_yf,
    test_labels_tsmd_yf,
    title="Intra-TSMD - Y_fixed_0.3 - PLCC",
    figsize=(5, 4),
)


# ============================
# 8. Cross-base New Render - Y_fixed_0.3
# ============================

# Cross-base entre TMQ et TSMD, New render, Y_fixed, avec plusieurs nombres de vues
train_labels_cross_yf = [
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",
    "TMQ-N-YF-16V",
    "TSMD-N-YF-1V",
    "TSMD-N-YF-4V",
    "TSMD-N-YF-8V",
    "TSMD-N-YF-16V",
]

test_labels_cross_yf = train_labels_cross_yf

data_cross_yf = build_matrix(agg, train_labels_cross_yf, test_labels_cross_yf)
plot_heatmap(
    data_cross_yf,
    train_labels_cross_yf,
    test_labels_cross_yf,
    title="Cross-base - New Render - Y_fixed_0.3 - PLCC",
    figsize=(7, 5),
)


# ============================
# 9. Cross-base New Render - Fibonacci (optionnel)
# ============================

train_labels_cross_fib = [
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-8V",
    "TMQ-N-Fib-16V",
    # Ajoute ici des TSMD-N-Fib-xV si tu en as
]

test_labels_cross_fib = train_labels_cross_fib

data_cross_fib = build_matrix(agg, train_labels_cross_fib, test_labels_cross_fib)
plot_heatmap(
    data_cross_fib,
    train_labels_cross_fib,
    test_labels_cross_fib,
    title="Cross-base - New Render - Fibonacci - PLCC",
    figsize=(6, 5),
)


# SJTU-TMQA intra-base results
train_labels_sjtu_tmqa = [
    "SJTU-TMQA-N-Fib-4V",
    "SJTU-TMQA-N-Fib-8V",
    "SJTU-TMQA-N-Fib-16V",
    "SJTU-TMQA-N-YF-4V",
    "SJTU-TMQA-N-YF-8V",
    "SJTU-TMQA-N-YF-16V",    
]
test_labels_sjtu_tmqa = train_labels_sjtu_tmqa
data_sjtu_tmqa = build_matrix(agg, train_labels_sjtu_tmqa, test_labels_sjtu_tmqa)
plot_heatmap(
    data_sjtu_tmqa,
    train_labels_sjtu_tmqa,
    test_labels_sjtu_tmqa,
    title="Intra-SJTU-TMQA - PLCC",
    figsize=(7, 5),
)

# Tests cross-base TMQ <-> SJTU-TMQA
train_labels_cross_sjtu_tmqa = [
    "Yana-Original",
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-8V",
    "TMQ-N-Fib-16V",
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",
    "TMQ-N-YF-16V",    
    "SJTU-TMQA-N-Fib-4V",
    "SJTU-TMQA-N-Fib-8V",
    "SJTU-TMQA-N-Fib-16V",
    "SJTU-TMQA-N-YF-4V",
    "SJTU-TMQA-N-YF-8V",
    "SJTU-TMQA-N-YF-16V",    
]
test_labels_cross_sjtu_tmqa = [
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-8V",
    "TMQ-N-Fib-16V",
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",
    "TMQ-N-YF-16V",    
    "SJTU-TMQA-N-Fib-4V",
    "SJTU-TMQA-N-Fib-8V",
    "SJTU-TMQA-N-Fib-16V",
    "SJTU-TMQA-N-YF-4V",
    "SJTU-TMQA-N-YF-8V",
    "SJTU-TMQA-N-YF-16V",    
]   
data_cross_sjtu_tmqa = build_matrix(agg, train_labels_cross_sjtu_tmqa, test_labels_cross_sjtu_tmqa)
plot_heatmap(
    data_cross_sjtu_tmqa,
    train_labels_cross_sjtu_tmqa,
    test_labels_cross_sjtu_tmqa,
    title="Cross-base TMQ <-> SJTU-TMQA - PLCC",
    figsize=(8, 6),
)
# Tests cross-base YF TMQ <-> SJTU-TMQA witrhout anti-aliasing
train_labels_cross_sjtu_tmqa_YFNAA = [
    # "Yana-Original",
    "TMQ-N-Org-1V",
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",    
    "TMQ-SJTU-N-YF-4V",
    "TMQ-SJTU-N-YF-8V",
    "TMQ-SJTU-N-YF-16V",
    "SJTU-TMQA-N-YF-NAA-4V",
    "SJTU-TMQA-N-YF-NAA-8V",
    "SJTU-TMQA-N-YF-NAA-16V",    
]
test_labels_cross_sjtu_tmqa_YFNAA = [
    "TMQ-N-Org-1V",
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",    
    "SJTU-TMQA-N-YF-NAA-4V",
    "SJTU-TMQA-N-YF-NAA-8V",
    "SJTU-TMQA-N-YF-NAA-16V",   
]   
data_cross_sjtu_tmqa_YFNAA = build_matrix(agg, train_labels_cross_sjtu_tmqa_YFNAA, test_labels_cross_sjtu_tmqa_YFNAA)

print(summary[summary["model"].str.contains("TMQ[-_]SJTU", case=False, na=False)][["model", "train_alias"]]
      .drop_duplicates().head(50))
plot_heatmap(
    data_cross_sjtu_tmqa_YFNAA,
    train_labels_cross_sjtu_tmqa_YFNAA,
    test_labels_cross_sjtu_tmqa_YFNAA,
    title="Cross-base TMQ <-> SJTU-TMQA - NAA - PLCC",
    figsize=(8, 6),
)