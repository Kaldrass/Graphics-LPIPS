# check_checkpoints.py
import os
import hashlib
import config  # ton fichier config.py

base = "./checkpoints/" + config.model

for fold in range(5):
    path = os.path.join(base, f"fold_k{fold}", "latest_net_.pth")
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    print(f"Fold {fold} - {path} - size {size} - md5 {md5}")