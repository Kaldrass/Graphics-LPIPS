# train_multiple.py
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============== À PERSONNALISER ==============
# Chemin vers TON script d'entraînement (celui que tu as posté)
TRAIN_SCRIPT = r"D:\These\Graphics-LPIPS\train.py"   # <-- mets le bon nom si différent

# Racine où seront créés les sous-dossiers d'expériences
CHECKPOINTS_DIR = r"D:\These\Graphics-LPIPS\checkpoints"

# Paramètres fixes communs à tous les runs (doivent correspondre aux --args de train.py)
BASE = dict(
    model="lpips",
    net="alex",
    use_gpu=True,
    gpu_ids=[0],
    nThreads=16,
    nepoch=5,
    nepoch_decay=5,
    npatches=150,
    nInputImg=4,
    lr=1e-4,
    testset_freq=2,
    display_freq=50000,
    print_freq=50000,
    save_latest_freq=20000,
    save_epoch_freq=1,
    display_id=0,
    display_winsize=256,
    display_port=8001,
    use_html=False,
    checkpoints_dir=CHECKPOINTS_DIR,
    name="Sweep",  # préfixe, un suffixe auto sera ajouté
    # Les 3 suivants reprennent tes defaults ; adapte si besoin
    src_root=r"D:\These\Projets\CompareMetrics\out\TMQ\Old_Render\Original",
    root_refPatches=r"\Source\1VP",
    root_distPatches=r"\Distorted\1VP",
    datasets=[r"./dataset/TexturedDB_80%_TrainList_withnbPatchesPerVP_threth0.6.csv",
              r"./dataset/TSMD/TSMD_80%_TrainList_scaled.csv"],
    testcsv=[r"./dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv",
             r"./dataset/TSMD/TSMD_20%_TestList_scaled.csv"],
)

# Hyperparamètres à balayer (toutes les combinaisons seront testées)
SWEEP = dict(
    lr=[1e-4],
    npatches=[150],
    nInputImg=[4],
    nepoch=[5],
    nepoch_decay=[5],
    # Entraîner ou non le trunk (tes flags)
    train_trunk=[False],
    from_scratch=[False],
)

# Si un run a déjà produit un checkpoint "latest_net_.pth", on le saute
SKIP_IF_CHECKPOINT_EXISTS = True

# Option: créer un résumé JSON des runs
WRITE_SUMMARY = True
SUMMARY_PATH = os.path.join(CHECKPOINTS_DIR, f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
# ============================================




def _to_cli_list(values):
    """Transforme une liste Python en forme CLI (répéter l'argument)."""
    out = []
    for v in values:
        out.append(str(v))
    return out

def _exists_latest_ckpt(exp_dir: str) -> bool:
    """Détecte un checkpoint typique: 'latest_net_.pth' dans exp_dir."""
    try:
        for n in os.listdir(exp_dir):
            if n.lower().startswith("latest_net_") and n.lower().endswith(".pth"):
                return True
    except FileNotFoundError:
        return False
    return False

def build_name(base_name, cfg):
    """Construit un nom d’expérience compact à partir d’un dict de paramètres."""
    # Crée un suffixe déterministe et lisible
    bits = []
    keys_for_name = ["lr", "npatches", "nInputImg", "train_trunk", "from_scratch", "nepoch", "nepoch_decay"]
    for k in keys_for_name:
        if k in cfg:
            v = cfg[k]
            if isinstance(v, float):
                v = f"{v:.0e}".replace("+0", "").replace("-0", "-")
            elif isinstance(v, bool):
                v = int(v)
            bits.append(f"{k[:3]}{v}")
    return f"{base_name}_" + "_".join(bits)

def run_one(config: dict):
    """Lance un entraînement avec la config donnée."""
    # Prépare le nom d’expérience et le dossier
    name = build_name(config["name"], config)
    exp_dir = os.path.join(config["checkpoints_dir"], name)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    if SKIP_IF_CHECKPOINT_EXISTS and _exists_latest_ckpt(exp_dir):
        print(f"[SKIP] {name} → checkpoint déjà présent.")
        return {"name": name, "status": "skipped", "checkpoints_dir": exp_dir, "config": config}

    # Construit la ligne de commande
    argv = [sys.executable, TRAIN_SCRIPT]

    # Arguments scalaires
    scalar_keys = [
        "model","net","nThreads","nepoch","nepoch_decay","npatches","nInputImg","lr",
        "testset_freq","display_freq","print_freq","save_latest_freq","save_epoch_freq",
        "display_id","display_winsize","display_port","checkpoints_dir","name","src_root","root_refPatches","root_distPatches"
    ]
    for k in scalar_keys:
        if k in config and config[k] is not None:
            argv += [f"--{k}", str(config[k])]

    # Booléens (flags)
    flag_keys_true = []
    if config.get("use_gpu", False): flag_keys_true.append("use_gpu")
    if config.get("use_html", False): flag_keys_true.append("use_html")
    if config.get("train_trunk", False): flag_keys_true.append("train_trunk")
    if config.get("from_scratch", False): flag_keys_true.append("from_scratch")
    for fk in flag_keys_true:
        argv += [f"--{fk}"]

    # Listes
    if "gpu_ids" in config and config["gpu_ids"]:
        argv += ["--gpu_ids"] + [str(x) for x in config["gpu_ids"]]
    if "datasets" in config and config["datasets"]:
        argv += ["--datasets"] + _to_cli_list(config["datasets"])
    if "testcsv" in config and config["testcsv"]:
        argv += ["--testcsv"] + _to_cli_list(config["testcsv"])

    print("\n=== Lancement ===")
    print("Nom :", name)
    print("CMD :", " ".join(argv))

    # Lance le processus
    try:
        subprocess.run(argv, check=True)
        status = "ok"
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] run {name} a échoué avec code {e.returncode}")
        status = f"error:{e.returncode}"

    return {"name": name, "status": status, "checkpoints_dir": exp_dir, "config": config}


def main():
    # Construire toutes les combinaisons du sweep
    sweep_keys = list(SWEEP.keys())
    sweep_vals = [SWEEP[k] for k in sweep_keys]
    combos = list(itertools.product(*sweep_vals))

    results = []
    print(f"[INFO] {len(combos)} combinaisons à lancer.")

    for vals in combos:
        cfg = BASE.copy()
        for k, v in zip(sweep_keys, vals):
            cfg[k] = v
        res = run_one(cfg)
        results.append(res)

    if WRITE_SUMMARY:
        try:
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print("[SUMMARY] écrit →", SUMMARY_PATH)
        except Exception as e:
            print("[SUMMARY] échec d’écriture:", e)

    print("\n=== Terminé ===")
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skipped")
    err = len(results) - ok - skip
    print(f"OK={ok} | SKIP={skip} | ERR={err}")


if __name__ == "__main__":
    main()
