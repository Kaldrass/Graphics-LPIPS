import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Configs TMQ
# -------------------------------
configs = [
    "TMQ-O-1V",
    "TMQ-O-4V",
    "TMQ-N-1V",
    "TMQ-N-4V",
]

# ---------------------------------------------
# 2. Tableau des résultats 
# ---------------------------------------------
data = [
    # Test :  TMQ-O-1V    TMQ-O-4V    TMQ-N-1V    TMQ-N-4V
    [0.878,   0.848,     0.838,     0.808],   # Train ORIGINAL DE YANA
    [0.881,   0.855,     0.839,     0.816],   # Train TMQ-O-1V
    [0.872,   0.864,     0.826,     0.820],   # Train TMQ-O-4V
    [0.881,   0.858,     0.866,     0.839],   # Train TMQ-N-1V
    [0.878,   0.870,     0.857,     0.851],   # Train TMQ-N-4V
]

results = pd.DataFrame(data, index=["Yana-Original"] + configs, columns=configs)

# ---------------------------------------------
# 3. Heatmap seaborn
# ---------------------------------------------
sns.set_theme(style="white", font_scale=1.0)

plt.figure(figsize=(6, 5))

ax = sns.heatmap(
    results,
    annot=True,        # affiche les valeurs dans les cases
    fmt=".3f",          
    cmap="viridis",    # viridis pour Baptiste
    vmin=0.800, vmax=0.900 
)

ax.set_title("Intra-TMQ - PLCC", pad=12)
ax.set_xlabel("Test")
ax.set_ylabel("Entraînement")

plt.tight_layout()
plt.show()
# plt.savefig("heatmap_intra_TMQ_PLCC.png", dpi=300, bbox_inches="tight")


# -------------------------------
# 4. Configs TSMD
# -------------------------------

# TSMD n'existe qu'en rendu New
configs_tsmd = ["TSMD-1V", "TSMD-4V"]

data_tsmd = [
    # Test :   TSMD-1V   TSMD-4V
    [0.871,   0.888],   # Train TSMD-1V
    [0.872,   0.888],   # Train TSMD-4V
]

results_tsmd = pd.DataFrame(data_tsmd, index=configs_tsmd, columns=configs_tsmd)

sns.set_theme(style="white", font_scale=1.0)
plt.figure(figsize=(4, 3.5))

ax = sns.heatmap(
    results_tsmd,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    vmin=0.800, vmax=0.900
)
ax.set_title("Intra-TSMD (New) - PLCC", pad=12)
ax.set_xlabel("Test")
ax.set_ylabel("Entraînement")

plt.tight_layout()
plt.show()
# plt.savefig("heatmap_intra_TSMD_SRCC.png", dpi=300, bbox_inches="tight")

# -------------------------------
# 5. Configs Cross-Dataset
# -------------------------------

configs_new = [
    "TMQ-N-1V",
    "TMQ-N-4V",
    "TSMD-1V",
    "TSMD-4V",
]

# Table SRCC (ex.) : lignes = TRAIN, colonnes = TEST
data_cross_new = [
    # Test :   TMQ-N-1V  TMQ-N-4V  TSMD-1V   TSMD-4V
    [0.866,   0.839,   0.882,   0.897],   # Train TMQ-N-1V
    [0.857,   0.851,   0.879,   0.894],   # Train TMQ-N-4V
    [0.829,   0.819,   0.871,   0.888],   # Train TSMD-1V
    [0.790,   0.777,   0.872,   0.888],   # Train TSMD-4V
]

results_cross_new = pd.DataFrame(data_cross_new, index=configs_new, columns=configs_new)

sns.set_theme(style="white", font_scale=1.0)
plt.figure(figsize=(6, 5))

ax = sns.heatmap(
    results_cross_new,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    vmin=0.777, vmax=0.900
)
ax.set_title("Cross-base (New Render) - PLCC", pad=12)
ax.set_xlabel("Test")
ax.set_ylabel("Entraînement")

plt.tight_layout()
plt.show()
# plt.savefig("heatmap_crossbase_New_SRCC.png", dpi=300, bbox_inches="tight")