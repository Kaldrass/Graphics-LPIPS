
import os, shutil, time, hashlib, threading
import csv
import random
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as T
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from collections import OrderedDict
from typing import Optional
import re


_COPY_SEM = threading.Semaphore(6) # 2 copies simultanées max
cv2.setNumThreads(0) # Avoid overthreading from open-cv

_to_tensor_01 = T.ToTensor()
_norm_m11 = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def imread_cached_rgb(path: str, use_cache: bool = True):
    """
    Legacy-like : lecture simple en RGB via PIL, sans cache ni conversion BGR.
    """
    if path is None or not os.path.exists(path):
        return None
    try:
        img = cv2.imread(path)[:,:,::-1]  # BGR -> RGB
        return img
    except Exception:
        return None
def pick_view_path(base_folder: str, view_idx: int, ext: str = "auto") -> Optional[str]:
    """Retourne le chemin d'une vue existante (view_{i}.ext). Si ext='auto', essaie png/jpg/jpeg."""
    candidates = [ext] if ext != "auto" else ["png", "jpg", "jpeg"]
    for e in candidates:
        p = os.path.join(base_folder, f"view_{view_idx}.{e}")
        if os.path.exists(p):
            return p
    return None

def ensure_on_ssd(src_path: str, src_root: Optional[str], cache_root: Optional[str],
                  retries: int = 30, sleep: float = 0.05) -> str:
    """
    Copie src_path vers cache_root en miroir de src_root. 
    Si cache_root est None, retourne src_path directement.
    Lève FileNotFoundError avec message clair si src n'existe pas.
    """
    from pathlib import Path
    import hashlib, shutil, time, threading, os

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src_path}")

    if not cache_root:
        return str(src)

    sroot = Path(src_root).resolve() if src_root else None
    croot = Path(cache_root).resolve()
    # Si déjà sur le même disque (même lettre sous Windows), le gain est faible → on évite la copie
    try:
        if src.drive and croot.drive and (src.drive.lower() == croot.drive.lower()):
            return str(src)
    except Exception:
        pass
    # Si le fichier est déjà dans le cache, inutile de copier
    if str(src).lower().startswith(str(croot).lower()):
        return str(src)
    
    try:
        rel = src.resolve().relative_to(sroot) if sroot else src.name
        dst = (croot / rel).resolve()
    except Exception:
        h = hashlib.sha1(str(src).encode('utf-8')).hexdigest()[:16]
        dst = (croot / "_outside_root" / h / src.name).resolve()

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return str(dst)

    lock = dst.with_suffix(dst.suffix + ".lock")
    for _ in range(retries):
        try:
            with _COPY_SEM:  # limite la concurrence globale
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                try:
                    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
                    shutil.copy2(str(src), str(tmp))
                    os.replace(str(tmp), str(dst))
                finally:
                    try:
                        os.remove(str(lock))
                    except FileNotFoundError:
                        pass
            return str(dst)
        except FileExistsError:
            if dst.exists():
                return str(dst)
            time.sleep(sleep)
        except (PermissionError, OSError) as e:
            # Ex: WinError 1450 → on attend un peu puis on retente; en dernier ressort, on retourne la source
            # print(f"[cache] copy retry due to: {e}")
            time.sleep(sleep)
    # Fallback: on ne bloque pas l'entraînement pour du cache
    # print("[cache] fallback to source (copy retries exhausted)")
    return str(src)

class TwoAFCDataset(Dataset):
    def initialize(self, dataroots, load_size=64, Trainset=False, maxNbPatches=150, 
                 root_refPatches=None, root_distPatches=None, src_root=None, target=None, img_ext='auto',
                 cache_root: str = r'C:\Graphics_LPIPS\cache'):
        self.target = target
        self.patch_entries = []
        self.transform = transforms.Compose([
            transforms.Resize(load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -> [-1, 1]
        ])

        # Racines des répertoires
        self.cache_root = cache_root #r'C:\Graphics_LPIPS\cache'
        self.src_root = src_root
        self.root_refPatches =  os.path.join(self.src_root, root_refPatches)
        self.root_distPatches = os.path.join(self.src_root, root_distPatches)
        self.img_ext = img_ext
        # root_judges = r'D:\These\Graphics-LPIPS\dataset\judge_trainingset' if Trainset else r'D:\These\Graphics-LPIPS\dataset\judge_testset'
        if (target == 'judges'):
            root_judges = r'D:\These\Graphics-LPIPS\dataset\judges'
        else:
            # Usually, the mos is located in the csv file used to create the dataloader
            root_judges = None
        # Limit of RAM cache for images per worker
        # SET ram_cache_limit_mb to 0 to disable RAM cache and use cv2.imread directly
        self._ssd_map = {}  # src_path -> ssd_path
        self._pil_cache = {}  # path -> PIL.Image
        self._judge_cache = {}
        
        if not isinstance(dataroots, list):
            dataroots = [dataroots]
        
        if Trainset:
            shuffled_inputfile = []
            print("SHUFFLINGGGGGGGGGGGG!!!")
            for idx, datafile in enumerate(dataroots):
                out_path = f'D:\\These\\Graphics-LPIPS\\dataset\\Trainset_shuffled_{idx+1}.csv'
                with open(datafile, 'r') as r, open(out_path, 'w') as w:
                    lines = r.readlines()
                    header, rows = lines[0], lines[1:]
                    random.shuffle(rows)
                    w.write(header + ''.join(rows))
                shuffled_inputfile.append(out_path)
            dataroots = shuffled_inputfile

        stimuli_id = 0
        for csv_file_path in dataroots:
            with open(csv_file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # header
                for row in reader:
                    model = row[0]
                    stimulus = row[1]
                    mos = float(row[2])
                    patch_csv_path = os.path.join(self.root_refPatches, model, 'patchs', f"{model}_patchlist.csv")
                    ref_view_folder = os.path.join(self.root_refPatches, model, 'views')
                    dis_view_folder = os.path.join(self.root_distPatches, stimulus, 'views')
                    if target == 'judges':
                        judge_path = os.path.join(root_judges, f"{stimulus}.npy")
                    else:
                        judge_path = None
                    with open(patch_csv_path, newline='') as pf:
                        patch_reader = csv.reader(pf)
                        patch_header = next(patch_reader)
                        patch_data = list(patch_reader)
                        patch_size = int(patch_header[4].split('=')[1])
                        nb_patches_per_view = [int(x.split('=')[1]) for x in patch_header[7:]]
                        nviews = len(nb_patches_per_view)

                    nb_patches_total = sum(nb_patches_per_view)
                    # nb_patches_view_one = nb_patches_per_view[0]
                    nb_full = maxNbPatches // nb_patches_total  
                    nb_rand = maxNbPatches % nb_patches_total 
                    # nb_full = maxNbPatches // nb_patches_view_one  
                    # nb_rand = maxNbPatches % nb_patches_view_one 

                    # print('Nbfull IMAGE %.1f'%nb_full)
                    # print('NbRandom patches %.1f'%nb_rand)
                    for _ in range(nb_full):
                        view_counter = 1
                        patch_seen = 0
                        for pd in patch_data:
                            x, y = int(pd[0]), int(pd[1])
                            self.patch_entries.append({
                                'ref_path': pick_view_path(ref_view_folder, view_counter, self.img_ext),
                                'dis_path': pick_view_path(dis_view_folder, view_counter, self.img_ext),
                                'x': x,
                                'y': y,
                                'mos': mos,
                                'patch_size': patch_size,
                                'judge_path': judge_path,
                                'stimuli_id': stimuli_id
                            })
                            patch_seen += 1
                            if patch_seen == nb_patches_per_view[view_counter - 1]:
                                view_counter += 1
                                patch_seen = 0

                    if nb_rand > 0:
                        selected = random.sample(range(len(patch_data)), nb_rand)
                        for idx in selected:
                            cumulative = 0
                            for v, nb in enumerate(nb_patches_per_view):
                                cumulative += nb
                                if idx < cumulative:
                                    view_num = v + 1
                                    break
                            x, y = int(patch_data[idx][0]), int(patch_data[idx][1])
                            self.patch_entries.append({
                                'ref_path': pick_view_path(ref_view_folder, view_num, self.img_ext),
                                'dis_path': pick_view_path(dis_view_folder, view_num, self.img_ext),
                                'x': x,
                                'y': y,
                                'mos': mos,
                                'patch_size': patch_size,
                                'judge_path': judge_path,
                                'stimuli_id': stimuli_id
                            })
                    stimuli_id += 1

    
    def __getitem__(self, idx):
        entry = self.patch_entries[idx]
        ref_src = entry['ref_path']
        dis_src = entry['dis_path']
        x, y = int(entry['x']), int(entry['y'])
        size = int(entry['patch_size'])
            
        # --- SSD CACHING ---     
        
        ref_path = self._ssd_map.get(ref_src)
        if ref_path is None:
            ref_path = ensure_on_ssd(ref_src, self.src_root, self.cache_root)
            self._ssd_map[ref_src] = ref_path
        
        dis_path = self._ssd_map.get(dis_src)
        if dis_path is None:
            dis_path = ensure_on_ssd(dis_src, self.src_root, self.cache_root)
            self._ssd_map[dis_src] = dis_path
        # ---  PIL/RGB  ---
        img = self._pil_cache.get(ref_path)
        if img is None:
            img = imread_cached_rgb(ref_path, use_cache=False)  # your PIL RGB reader
            self._pil_cache[ref_path] = img
        ref_img = img
        
        img = self._pil_cache.get(dis_path)
        if img is None:
            img = imread_cached_rgb(dis_path, use_cache=False)
            self._pil_cache[dis_path] = img
        dis_img = img

        if ref_img is None or dis_img is None:
            raise FileNotFoundError(f"Missing image: ref={ref_path}, dis={dis_path}")

        def load_patch(img, x, y, size):
            
            if img is None:
                raise RuntimeError(f"Image introuvable : {img}")
            patch = img[y:y+size, x:x+size]
            return self.transform(Image.fromarray(patch))

        ref_patch = load_patch(ref_img, entry['x'], entry['y'], entry['patch_size'])
        dis_patch = load_patch(dis_img, entry['x'], entry['y'], entry['patch_size'])
        # if the target is judges, load the judge file
        if self.target == 'judges':
            judge = torch.from_numpy(np.load(entry['judge_path'])).float().view(1, 1, 1)
        else:
            judge = None

        
        jp = entry['judge_path']
        judge = self._judge_cache.get(jp)
        # if judge is None:
            # j = np.load(jp).astype(np.float32)
            # judge = float(j) if j.ndim == 0 else j
            # self._judge_cache[jp] = judge

        out = {
            'ref': ref_patch,
            'p0': dis_patch,
            # If we don't have judges, we return the mos as target
            'judge': torch.tensor(judge, dtype=torch.float32) if judge is not None else torch.tensor(entry.get('mos', 0.0), dtype=torch.float32),
            'stimuli_id': torch.tensor(entry['stimuli_id'], dtype=torch.long),

            # meta facultatives (si ton code en amont les loggue)
            'ref_path': ref_path,
            'p0_path': dis_path,
            'x': x, 'y': y, 'patch_size': size,
            'mos': float(entry.get('mos', 0.0)),
        }
        return out
    def __len__(self):
        return len(self.patch_entries)
