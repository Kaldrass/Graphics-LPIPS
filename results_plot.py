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

    # Matches:
    # TMQ_NR_4VP_YF03_...
    # TMQ_OR_1VP_ORG_...
    # TMQ_SJTU_NR_8VP_FIB_...
    # SJTU-TMQA_NR_16VP_YF03_...
    # BASICS_SP_8VP_YF03_...
    pattern = r"(TMQ(?:-|_)SJTU|TMQ-SJTU|SJTU-TMQA|TMQ|TSMD|BASICS)_(OR|NR|SP|VX)_(\d+)VP_([A-Z0-9]+)"
    match = re.search(pattern, m)
    if not match:
        return None

    db, render_tag, nviews, view_tag = match.groups()

    db = db.replace("_", "-")  # TMQ_SJTU -> TMQ-SJTU
    # Render tag
    if render_tag == "OR":
        render_short = "O"
    elif render_tag == "NR":
        render_short = "N"
    else:
        render_short = render_tag  # SP or VX

    # View tag
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
    db_raw = str(row.get("database", "")).upper()
    rm = str(row.get("render_method", "")).upper()
    vm = str(row.get("view_method", "")).upper()

    # Defensive cast
    try:
        v = int(row.get("testing_views", 0))
    except Exception:
        v = 0

    # Normalize DB formatting
    db_norm = db_raw.replace("_", "-")

    # Map DB
    if db_norm in {"TMQ", "TSMD", "SJTU-TMQA", "TMQ-SJTU"}:
        db = db_norm
    elif db_raw == "BASICS(PC)_DB" or db_norm == "BASICS(PC)-DB":
        db = "BASICS"
    else:
        return None

    # Render method
    if rm in {"OLD_RENDER", "OLD"}:
        render_short = "O"
    elif rm in {"NEW_RENDER", "NEW"}:
        render_short = "N"
    elif rm in {"SP", "VX"}:
        render_short = rm
    else:
        return None

    # View method
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
# 10. Intra-BASICS(PC)_DB - PLCC
# ============================

summary = load_correlation_summaries(BASE_RESULTS_DIR, SUMMARY_FILENAME)

# Derive aliases
summary["train_alias"] = summary["model"].apply(training_alias_from_model)
summary["test_alias"] = summary.apply(test_alias_from_row, axis=1)

# Keep only rows we can map to aliases
summary_valid = summary.dropna(subset=["train_alias", "test_alias"])

# Aggregate (mean in case multiple runs exist for same (train, test))
agg = summary_valid.groupby(["train_alias", "test_alias"])[METRIC_COL].mean()

agg = summary_valid.groupby(["train_alias", "test_alias"])[METRIC_COL].mean()
# Collect all aliases that correspond to BASICS in train and test



ALIAS_RE = re.compile(r"^(?P<db>.+)-(?P<render>O|N|SP|VX)-(?P<view>.+)-(?P<nviews>\d+)V$")

def _parse_alias(alias: str) -> dict:
    m = ALIAS_RE.match(str(alias))
    if not m:
        return {"db": None, "render": None, "view": None, "nviews": None}
    d = m.groupdict()
    d["nviews"] = int(d["nviews"])
    return d


def make_pairs_df(agg_series: pd.Series) -> pd.DataFrame:
    """
    Convert agg (MultiIndex train_alias,test_alias -> metric) to a DataFrame with parsed columns.
    """
    pairs = agg_series.rename("metric").reset_index()  # columns: train_alias, test_alias, metric

    train_parsed = pd.json_normalize(pairs["train_alias"].apply(_parse_alias))
    train_parsed = train_parsed.add_prefix("train_")

    test_parsed = pd.json_normalize(pairs["test_alias"].apply(_parse_alias))
    test_parsed = test_parsed.add_prefix("test_")

    out = pd.concat([pairs, train_parsed, test_parsed], axis=1)
    return out


def plot_corr_vs_train_views(
    pairs_df: pd.DataFrame,
    *,
    db: str,
    render: str,
    view_prefix: str,
    title: str,
    metric_label: str = "Correlation",
    use_diagonal: bool = True,
    fixed_test_views: int | None = None,
    fixed_test_same_method: bool = True,
):
    """
    Plot correlation vs training number of views for a fixed viewpoint method.

    Default behavior (use_diagonal=True):
      - Uses only rows where train_alias == test_alias (intra-config diagonal)

    Alternative behavior:
      - fixed_test_views = 4 (for example) tests every training config against test configs with 4 views
      - If fixed_test_same_method=True, test method must match view_prefix
    """
    df = pairs_df.copy()

    # Filter train side
    df = df[
        (df["train_db"] == db) &
        (df["train_render"] == render) &
        (df["train_view"].fillna("").str.startswith(view_prefix))
    ]

    if use_diagonal:
        df = df[df["train_alias"] == df["test_alias"]]
    else:
        # Filter test side
        df = df[df["test_db"] == db]
        df = df[df["test_render"] == render]

        if fixed_test_same_method:
            df = df[df["test_view"].fillna("").str.startswith(view_prefix)]

        if fixed_test_views is not None:
            df = df[df["test_nviews"] == int(fixed_test_views)]

    if df.empty:
        print(f"[plot] No data for db={db}, render={render}, view={view_prefix}, diagonal={use_diagonal}, fixed_test_views={fixed_test_views}")
        return

    # Aggregate in case multiple rows exist per train_nviews (should be rare)
    curve = (
        df.groupby("train_nviews")["metric"]
        .mean()
        .sort_index()
        .reset_index()
        .rename(columns={"train_nviews": "nviews"})
    )

    x = curve["nviews"].to_list()
    y = curve["metric"].to_list()

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Number of training views")
    plt.ylabel(metric_label)
    plt.tight_layout()
    plt.show()


pairs_df = make_pairs_df(agg)



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

# Par exemple, tu peux avoir 4, 8, 16 vues pour TMQ New Render + Y_fixed_0.3
train_labels_tmq_yf = [
    "TMQ-N-YF-1V",
    "TMQ-N-YF-2V",
    "TMQ-N-YF-3V",
    "TMQ-N-YF-4V",
    "TMQ-N-YF-5V",
    "TMQ-N-YF-6V",
    "TMQ-N-YF-7V",
    "TMQ-N-YF-8V",
    "TMQ-N-YF-9V",
    "TMQ-N-YF-10V",
    # "TMQ-N-YF-16V",
]

test_labels_tmq_yf = train_labels_tmq_yf

# Idem, adapte cette liste aux configs existantes
train_labels_tmq_fib = [
    "TMQ-N-Fib-1V",
    "TMQ-N-Fib-2V",
    "TMQ-N-Fib-3V",
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-5V",
    "TMQ-N-Fib-6V",
    "TMQ-N-Fib-7V",
    "TMQ-N-Fib-8V",
    "TMQ-N-Fib-9V",
    "TMQ-N-Fib-10V",
    # "TMQ-N-Fib-16V",
]

test_labels_tmq_fib = train_labels_tmq_fib

# Ajoute ici 8V/16V si tu as ces runs
train_labels_tsmd_yf = [
    "TSMD-N-YF-1V",
    "TSMD-N-YF-4V",
    "TSMD-N-YF-8V",
    # "TSMD-N-YF-16V",
]

test_labels_tsmd_yf = train_labels_tsmd_yf

# Cross-base entre TMQ et TSMD, New render, Y_fixed, avec plusieurs nombres de vues
train_labels_cross_yf = [
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",
    # "TMQ-N-YF-16V",
    "TSMD-N-YF-1V",
    "TSMD-N-YF-4V",
    "TSMD-N-YF-8V",
    "TSMD-N-YF-16V",
]

test_labels_cross_yf = train_labels_cross_yf

train_labels_cross_fib = [
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-8V",
    "TMQ-N-Fib-16V",
    # Ajoute ici des TSMD-N-Fib-xV si tu en as
]

test_labels_cross_fib = train_labels_cross_fib

train_labels_sjtu_tmqa = [
    "SJTU-TMQA-N-Fib-4V",
    "SJTU-TMQA-N-Fib-8V",
    "SJTU-TMQA-N-Fib-16V",
    "SJTU-TMQA-N-YF-4V",
    "SJTU-TMQA-N-YF-8V",
    "SJTU-TMQA-N-YF-16V",    
]
test_labels_sjtu_tmqa = train_labels_sjtu_tmqa

train_labels_cross_sjtu_tmqa = [
    "Yana-Original",
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-8V",
    # "TMQ-N-Fib-16V",
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",
    # "TMQ-N-YF-16V",    
    "SJTU-TMQA-N-Fib-4V",
    "SJTU-TMQA-N-Fib-8V",
    # "SJTU-TMQA-N-Fib-16V",
    "SJTU-TMQA-N-YF-4V",
    "SJTU-TMQA-N-YF-8V",
    # "SJTU-TMQA-N-YF-16V",    
]
test_labels_cross_sjtu_tmqa = [
    "TMQ-N-Fib-4V",
    "TMQ-N-Fib-8V",
    # "TMQ-N-Fib-16V",
    "TMQ-N-YF-4V",
    "TMQ-N-YF-8V",
    # "TMQ-N-YF-16V",    
    "SJTU-TMQA-N-Fib-4V",
    "SJTU-TMQA-N-Fib-8V",
    # "SJTU-TMQA-N-Fib-16V",
    "SJTU-TMQA-N-YF-4V",
    "SJTU-TMQA-N-YF-8V",
    # "SJTU-TMQA-N-YF-16V",    
]   
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

def plot_two_methods_overlay(pairs_df, *, db: str, render: str, methods: list[str], title: str, diagonal: bool = True):
    plt.figure(figsize=(6, 4))
    baseline = 0.59
    plt.axhline(baseline, linestyle="--", linewidth=1.5)
    for view_prefix in methods:
        df = pairs_df[
            (pairs_df["train_db"] == db) &
            (pairs_df["train_render"] == render) &
            (pairs_df["train_view"].fillna("").str.startswith(view_prefix))
        ]
        if diagonal:
            df = df[df["train_alias"] == df["test_alias"]]
        if df.empty:
            continue

        curve = df.groupby("train_nviews")["metric"].mean().sort_index()
        plt.plot(curve.index.to_list(), curve.values.tolist(), marker="o", label=view_prefix)
    plt.text(
        0.99, baseline,
        "YUV_PSNR (0.59)",
        transform=plt.gca().get_yaxis_transform(),  # x en coords axes [0..1], y en data
        ha="right",
        va="bottom"
    )
    ax= plt.gca()
    # ax.set_xticks(curve.index.to_list())
    ax.set_ylim(0.55, 0.70)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Number of training views")
    plt.ylabel(METRIC_COL)
    plt.legend()
    plt.tight_layout()
    plt.show()



# plot_corr_vs_train_views(
#     pairs_df,
#     db="TMQ",
#     render="N",
#     view_prefix="Org",
#     title=f"TMQ (New render) - Org - {METRIC_COL} vs number of views (diagonal)",
#     metric_label=METRIC_COL,
#     use_diagonal=True,
# )
# plot_corr_vs_train_views(
#     pairs_df,
#     db="TMQ",
#     render="N",
#     view_prefix="Fib",
#     title=f"TMQ (New render) - Fibonacci - {METRIC_COL} vs number of views (diagonal)",
#     metric_label=METRIC_COL,
#     use_diagonal=True,
# )
# plot_two_methods_overlay(
#     pairs_df,
#     db="TMQ",
#     render="N",
#     methods=["YF", "Fib"],
#     title=f"TMQ (New render) - {METRIC_COL} vs number of views (No diagonal)",
#     diagonal=False,
# )

plot_two_methods_overlay(
    pairs_df,
    db="TMQ",
    render="N",
    methods=["YF", "Fib"],
    title=f"TMQ (New render) - {METRIC_COL} vs number of views",
    diagonal=True,
)
plot_two_methods_overlay(
    pairs_df,
    db="TSMD",
    render="N",
    methods=["YF", "Fib"],
    title=f"TSMD (New render) - {METRIC_COL} vs number of views",
    diagonal=True,
)
plot_two_methods_overlay(
    pairs_df,
    db="SJTU-TMQA",
    render="N",
    methods=["YF", "Fib"],
    title=f"SJTU-TMQA (New render) - {METRIC_COL} vs number of views",
    diagonal=True,
)
plot_two_methods_overlay(
    pairs_df,
    db="BASICS",
    render="SP",
    methods=["YF", "Fib"],
    title=f"BASICS (New render) - {METRIC_COL} vs number of views",
    diagonal=True,
)
# plot_corr_vs_train_views(
#     pairs_df,
#     db="TMQ",
#     render="N",
#     view_prefix="YF",
#     title=f"TMQ (New render) - YF - {METRIC_COL} vs train views",
#     metric_label=METRIC_COL,
#     use_diagonal=False,
#     # fixed_test_views=4,
#     fixed_test_same_method=False,
# )
def plot_all_heatmaps():
    # Intra-BASICS(PC)_DB - PLCC
    all_train_aliases = agg.index.get_level_values(0).unique()
    all_test_aliases = agg.index.get_level_values(1).unique()
    tmq_tests = ["TMQ-N-YF-1V","TMQ-N-YF-4V"]

    basics_train = sorted(a for a in all_train_aliases if a.startswith("BASICS-"))
    basics_train = tmq_tests + basics_train
    basics_test = sorted(a for a in all_test_aliases if a.startswith("BASICS-"))

    if basics_train and basics_test:
        data_basics = build_matrix(agg, basics_train, basics_test)
        plot_heatmap(
            data_basics,
            basics_train,
            basics_test,
            title="Intra-BASICS(PC)_DB - PLCC",
            figsize=(max(4, 0.8 * len(basics_test)), max(4, 0.8 * len(basics_train))),
        )
    else:
        print("No BASICS(PC)_DB entries found (no train/test alias starting with 'BASICS-').")

    # data_tmq_Org = build_matrix(agg, train_labels_tmq_Org, test_labels_tmq_Org)
    # plot_heatmap(
    #     data_tmq_Org,
    #     train_labels_tmq_Org,
    #     test_labels_tmq_Org,
    #     title="Intra-TMQ - Orginal viewpoints - PLCC",
    #     figsize=(6, 5),
    # )
   
    # data_tmq_yf = build_matrix(agg, train_labels_tmq_yf, test_labels_tmq_yf)
    # plot_heatmap(
    #     data_tmq_yf,
    #     train_labels_tmq_yf,
    #     test_labels_tmq_yf,
    #     title="Intra-TMQ - Y_fixed_0.3 - PLCC",
    #     figsize=(5, 4),
    # )
    # data_tmq_fib = build_matrix(agg, train_labels_tmq_fib, test_labels_tmq_fib)
    # plot_heatmap(
    #     data_tmq_fib,
    #     train_labels_tmq_fib,
    #     test_labels_tmq_fib,
    #     title="Intra-TMQ - Fibonacci - PLCC",
    #     figsize=(5, 4),
    # )
    # data_tsmd_yf = build_matrix(agg, train_labels_tsmd_yf, test_labels_tsmd_yf)
    # plot_heatmap(
    #     data_tsmd_yf,
    #     train_labels_tsmd_yf,
    #     test_labels_tsmd_yf,
    #     title="Intra-TSMD - Y_fixed_0.3 - PLCC",
    #     figsize=(5, 4),
    # )
    # data_cross_yf = build_matrix(agg, train_labels_cross_yf, test_labels_cross_yf)
    # plot_heatmap(
    #     data_cross_yf,
    #     train_labels_cross_yf,
    #     test_labels_cross_yf,
    #     title="Cross-base - New Render - Y_fixed_0.3 - PLCC",
    #     figsize=(7, 5),
    # )
    # data_cross_fib = build_matrix(agg, train_labels_cross_fib, test_labels_cross_fib)
    # plot_heatmap(
    #     data_cross_fib,
    #     train_labels_cross_fib,
    #     test_labels_cross_fib,
    #     title="Cross-base - New Render - Fibonacci - PLCC",
    #     figsize=(6, 5),
    # )
    # data_sjtu_tmqa = build_matrix(agg, train_labels_sjtu_tmqa, test_labels_sjtu_tmqa)
    # plot_heatmap(
    #     data_sjtu_tmqa,
    #     train_labels_sjtu_tmqa,
    #     test_labels_sjtu_tmqa,
    #     title="Intra-SJTU-TMQA - PLCC",
    #     figsize=(7, 5),
    # )
    # data_cross_sjtu_tmqa = build_matrix(agg, train_labels_cross_sjtu_tmqa, test_labels_cross_sjtu_tmqa)
    # plot_heatmap(
    #     data_cross_sjtu_tmqa,
    #     train_labels_cross_sjtu_tmqa,
    #     test_labels_cross_sjtu_tmqa,
    #     title="Cross-base TMQ <-> SJTU-TMQA - PLCC",
    #     figsize=(8, 6),
    # )
    # data_cross_sjtu_tmqa_YFNAA = build_matrix(agg, train_labels_cross_sjtu_tmqa_YFNAA, test_labels_cross_sjtu_tmqa_YFNAA)
    # plot_heatmap(
    #     data_cross_sjtu_tmqa_YFNAA,
    #     train_labels_cross_sjtu_tmqa_YFNAA,
    #     test_labels_cross_sjtu_tmqa_YFNAA,
    #     title="Cross-base TMQ <-> SJTU-TMQA - NAA - PLCC",
    #     figsize=(8, 6),
    # )
plot_all_heatmaps()