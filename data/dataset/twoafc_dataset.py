# import os.path
# import torchvision.transforms as transforms
# from data.dataset.base_dataset import BaseDataset
# from data.image_folder import make_dataset
# from PIL import Image
# import numpy as np
# import torch
# from torch.utils.data import TensorDataset
# import csv
# import cv2
# import random
# import collections
# # from IPython import embed

# # class TwoAFCDataset(BaseDataset):
# #     """
# #     Dataset for Two-Alternative Forced Choice (2AFC) tasks.
# #     This dataset is designed to load reference images, distorted images, and judge scores
# #     from a CSV file. It supports both training and testing modes, with the option to shuffle
# #     the input CSV file for training datasets.
# #     Args:
# #         dataroots (str or list of str): Path(s) to the CSV file(s) containing the dataset.
# #         load_size (int): Size to which images will be resized.
# #         Trainset (bool): If True, the dataset is for training; if False, it is for testing.
# #         maxNbPatches (int): Maximum number of patches to load per stimulus.
# #     """

# #     def initialize(self, dataroots, load_size=64, Trainset=False, maxNbPatches=205):
# #         root_refPatches = r'D:\These\Projets\CompareMetrics\out\TMQ_ref_yf_1VP_Final'
# #         root_distPatches = r'D:\These\Projets\CompareMetrics\out\TMQ_dis_yf_1VP_Final'
# #         root_judges = r'D:\These\Graphics-LPIPS\dataset\judge_trainingset' if Trainset else r'D:\These\Graphics-LPIPS\dataset\judge_testset'

# #         # Définit la transformation d'image (Resize + Tensor + Normalisation)
# #         transform_list = [transforms.Resize(load_size),
# #                         transforms.ToTensor(),
# #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# #         self.transform = transforms.Compose(transform_list)
            
# #         # shuffle input csv file
# #         if(Trainset):
# #             shuffled_inputfile = [] 
# #             count_inputFile = 0
# #             print('SHUFFLINGGGGGGGGGGGG!!!')
# #             for datafile in dataroots:
# #                 count_inputFile = count_inputFile + 1
# #                 shuffledfileName = 'D:\\These\\Graphics-LPIPS\\dataset\\'+ 'Trainset_shuffled_' + str(count_inputFile) +'.csv'
# #                 shuffled_inputfile.append(shuffledfileName)
# #                 with open(datafile, 'r') as r, open(shuffledfileName, 'w') as w:
# #                     data = r.readlines()
# #                     header, rows = data[0], data[1:]
# #                     random.shuffle(rows)
# #                     rows = '\n'.join([row.strip() for row in rows])
# #                     w.write(header + rows)
            
# #             dataroots = shuffled_inputfile
    


# #         if not isinstance(dataroots, list):
# #             dataroots = [dataroots]

# #         self.ref_imgs = []
# #         self.p0_imgs = []
# #         self.judge_paths = []
# #         self.stimuliId = []
# #         stimuliId = 0
# #         nbiteration = 1

# #         for csv_file_path in dataroots:
# #             with open(csv_file_path, newline='') as csvfile:
# #                 reader = csv.reader(csvfile)
# #                 rows = list(reader)
# #                 header = rows[0]
# #                 data_rows = rows[1:]

# #                 for line_count, row in enumerate(data_rows):
# #                     model, stimulus, MOS = row[0], row[1], float(row[2])
# #                     patch_csv_path = os.path.join(root_refPatches, model, 'patchs', f"{model}_patchlist.csv")
# #                     ref_view_folder = os.path.join(root_refPatches, model, 'views')
# #                     dis_view_folder = os.path.join(root_distPatches, stimulus, 'views')
# #                     judge_path = os.path.join(root_judges, f"{stimulus}.npy")

# #                     # Lecture unique du fichier de patchs
# #                     with open(patch_csv_path, newline='') as patchfile:
# #                         patch_reader = csv.reader(patchfile)
# #                         patch_rows = list(patch_reader)
# #                         patch_header = patch_rows[0]
# #                         patch_data = patch_rows[1:]

# #                         patch_size = int(patch_header[4].split('=')[1])
# #                         nb_patches_per_view = [int(x.split('=')[1]) for x in patch_header[7:]]
# #                         nb_patches_total = sum(nb_patches_per_view)

# #                     nbfullimage = maxNbPatches // nb_patches_total
# #                     nbrandomPatches = maxNbPatches % nb_patches_total

# #                     for itr in range(nbiteration):
# #                         stimuliId += 1

# #                         # Preload all images by view number
# #                         ref_images = {}
# #                         dis_images = {}
# #                         for v in range(1, len(nb_patches_per_view) + 1):
# #                             img_path = os.path.join(ref_view_folder, f"view_{v}.png")
# #                             ref_img = cv2.imread(img_path)  
# #                             if ref_img is None:
# #                                 raise RuntimeError(f"Image non trouvée ou corrompue : {img_path}")
# #                             ref_img = ref_img[:, :, ::-1]
# #                             dis_img = cv2.imread(os.path.join(dis_view_folder, f"view_{v}.png"))[:, :, ::-1]
# #                             ref_images[v] = ref_img
# #                             dis_images[v] = dis_img

# #                         def extract_and_store_patch(row, view_index):
# #                             x, y = int(row[0]), int(row[1])
# #                             ref_patch = ref_images[view_index][y:y+patch_size, x:x+patch_size]
# #                             dis_patch = dis_images[view_index][y:y+patch_size, x:x+patch_size]
# #                             self.ref_imgs.append(self.transform(Image.fromarray(ref_patch)))
# #                             self.p0_imgs.append(self.transform(Image.fromarray(dis_patch)))
# #                             self.judge_paths.append(judge_path)
# #                             self.stimuliId.append(stimuliId)
# #                         patch_idx = 0
# #                         for _ in range(nbfullimage):
# #                             view_counter = 1
# #                             patch_seen = 0
# #                             for row in patch_data:
# #                                 extract_and_store_patch(row, view_counter)
# #                                 patch_seen += 1
# #                                 if patch_seen == nb_patches_per_view[view_counter - 1]:
# #                                     view_counter += 1
# #                                     patch_seen = 0
# #                                 patch_idx += 1

# #                         # Random complément pour atteindre maxNbPatches
# #                         if nbrandomPatches > 0:
# #                             selected = random.sample(range(len(patch_data)), nbrandomPatches)
# #                             for idx in selected:
# #                                 # Détermine à quelle vue appartient ce patch
# #                                 cumulative = 0
# #                                 for v, nb in enumerate(nb_patches_per_view):
# #                                     cumulative += nb
# #                                     if idx < cumulative:
# #                                         view_num = v + 1
# #                                         break
# #                                 extract_and_store_patch(patch_data[idx], view_num)
# #         # On regroupe les listes d'images en tenseurs
# #         self.ref_imgs = torch.stack(self.ref_imgs)       # shape: [N, 3, H, W]
# #         self.p0_imgs = torch.stack(self.p0_imgs)
# #         self.judges = torch.stack([
# #             torch.from_numpy(np.load(path)).float().view(1, 1, 1)
# #             for path in self.judge_paths
# #         ])
# #         self.stimuliId = torch.tensor(self.stimuliId).view(-1)
# #         # Transfert GPU
# #         # if torch.cuda.is_available():
# #             # print("Transfert des données en mémoire GPU...")
# #             # self.ref_imgs = self.ref_imgs.to("cuda:0")
# #             # self.p0_imgs = self.p0_imgs.to("cuda:0")
# #             # self.judges = self.judges.to("cuda:0")
# #             # self.stimuliId = self.stimuliId.to("cuda:0")
# #         self.dataset = TensorDataset(self.ref_imgs, self.p0_imgs, self.judges, self.stimuliId)

# #         print(f"Nombre total de stimuli : {stimuliId}")
# #         print(f"Nombre total de patches : {len(self.p0_imgs)}")

# #     def __getitem__(self, index):
# #         ref, p0, judge, stim = self.dataset[index]
# #         return {
# #             'ref': ref,
# #             'p0': p0,
# #             'judge': judge,
# #             'stimuli_id': stim,
# #         }
# #     def __len__(self):
# #         return len(self.dataset)


# class TwoAFCDataset(BaseDataset):
#     """
#     Dataset for Two-Alternative Forced Choice (2AFC) tasks.
#     This dataset is designed to load reference images, distorted images, and judge scores
#     from a CSV file. It supports both training and testing modes, with the option to shuffle
#     the input CSV file for training datasets.
#     Args:
#         dataroots (str or list of str): Path(s) to the CSV file(s) containing the dataset.
#         load_size (int): Size to which images will be resized.
#         Trainset (bool): If True, the dataset is for training; if False, it is for testing.
#         maxNbPatches (int): Maximum number of patches to load per stimulus.
#     """

#     def initialize(self, dataroots, load_size=64, Trainset=False, maxNbPatches=205):
#         self.patch_entries = []
#         self.transform = transforms.Compose([
#             transforms.Resize(load_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

#         root_refPatches = r'D:\These\Projets\CompareMetrics\out\_TMQ_GAUTIER_\REF_4VP'#r'D:\These\Projets\CompareMetrics\out\TMQ_ref_yf_1VP_Final'
#         root_distPatches = r'D:\These\Projets\CompareMetrics\out\_TMQ_GAUTIER_\DIS_4VP'
#         root_judges = r'D:\These\Graphics-LPIPS\dataset\judge_trainingset' if Trainset else r'D:\These\Graphics-LPIPS\dataset\judge_testset'

#         stimuli_id = 0

#         if(Trainset):
#                 shuffled_inputfile = [] 
#                 count_inputFile = 0
#                 print('SHUFFLINGGGGGGGGGGGG!!!')
#                 for datafile in dataroots:
#                     count_inputFile = count_inputFile + 1
#                     shuffledfileName = 'D:\\These\\Graphics-LPIPS\\dataset\\'+ 'Trainset_shuffled_' + str(count_inputFile) +'.csv'
#                     shuffled_inputfile.append(shuffledfileName)
#                     with open(datafile, 'r') as r, open(shuffledfileName, 'w') as w:
#                         data = r.readlines()
#                         header, rows = data[0], data[1:]
#                         random.shuffle(rows)
#                         rows = '\n'.join([row.strip() for row in rows])
#                         w.write(header + rows)
                
#                 dataroots = shuffled_inputfile

#         if not isinstance(dataroots, list):
#             dataroots = [dataroots]
#         for csv_file_path in dataroots:
#             with open(csv_file_path, newline='') as csvfile:
#                 reader = csv.reader(csvfile)
#                 header = next(reader)
#                 for row in reader:
#                     model, stimulus, MOS = row[0], row[1], float(row[2])
#                     patch_csv_path = os.path.join(root_refPatches, model, 'patchs', f"{model}_patchlist.csv")
#                     ref_view_folder = os.path.join(root_refPatches, model, 'views')
#                     dis_view_folder = os.path.join(root_distPatches, stimulus, 'views')
#                     judge_path = os.path.join(root_judges, f"{stimulus}.npy")

#                     with open(patch_csv_path, newline='') as patchfile:
#                         patch_reader = csv.reader(patchfile)
#                         patch_header = next(patch_reader)
#                         patch_data = list(patch_reader)

#                         patch_size = int(patch_header[4].split('=')[1])
#                         nb_patches_per_view = [int(x.split('=')[1]) for x in patch_header[7:]]
#                         nb_patches_total = sum(nb_patches_per_view)
#                         nb_patches_view_one = nb_patches_per_view[0]

#                     # nb_full = maxNbPatches // nb_patches_total
#                     nb_full = maxNbPatches // nb_patches_view_one # TRAINING ONLY ON VIEW ONE
#                     # nb_rand = maxNbPatches % nb_patches_total
#                     nb_rand = maxNbPatches % nb_patches_view_one

#                     for _ in range(nb_full):
#                         view_counter = 1
#                         patch_seen = 0
#                         #----------------------------------------------------------------------------------
#                         # MULTI VIEWS
#                         for row_patch in patch_data:
#                             x, y = int(row_patch[0]), int(row_patch[1])
#                             self.patch_entries.append({
#                                 'ref_path': os.path.join(ref_view_folder, f"view_{view_counter}.png"),
#                                 'dis_path': os.path.join(dis_view_folder, f"view_{view_counter}.png"),
#                                 'x': x,
#                                 'y': y,
#                                 'patch_size': patch_size,
#                                 'judge_path': judge_path,
#                                 'stimuli_id': stimuli_id
#                             })
#                             patch_seen += 1
#                             if patch_seen == nb_patches_per_view[view_counter - 1]:
#                                 view_counter += 1
#                                 patch_seen = 0
#                         #----------------------------------------------------------------------------------
#                         # SINGLE VIEW (VIEW 1 ONLY)
#                         # for row_patch in patch_data[:nb_patches_view_one]:
#                         #     x, y = int(row_patch[0]), int(row_patch[1])
#                         #     self.patch_entries.append({
#                         #         'ref_path': os.path.join(ref_view_folder, "view_1.png"),
#                         #         'dis_path': os.path.join(dis_view_folder, "view_1.png"),
#                         #         'x': x,
#                         #         'y': y,
#                         #         'patch_size': patch_size,
#                         #         'judge_path': judge_path,
#                         #         'stimuli_id': stimuli_id
#                         #     })
#                     if nb_rand > 0:
#                         #----------------------------------------------------------------------------------
#                         # MULTI VIEWS
#                         selected = random.sample(range(len(patch_data)), nb_rand)
#                         for idx in selected:
#                             cumulative = 0
#                             for v, nb in enumerate(nb_patches_per_view):
#                                 cumulative += nb
#                                 if idx < cumulative:
#                                     view_num = v + 1
#                                     break
#                             x, y = int(patch_data[idx][0]), int(patch_data[idx][1])
#                             self.patch_entries.append({
#                                 'ref_path': os.path.join(ref_view_folder, f"view_{view_num}.png"),
#                                 'dis_path': os.path.join(dis_view_folder, f"view_{view_num}.png"),
#                                 'x': x,
#                                 'y': y,
#                                 'patch_size': patch_size,
#                                 'judge_path': judge_path,
#                                 'stimuli_id': stimuli_id
#                             })
#                         #----------------------------------------------------------------------------------
#                         # SINGLE VIEW (VIEW 1 ONLY)
#                         # selected = random.sample(range(nb_patches_view_one), nb_rand)
#                         # for idx in selected:
#                         #     x, y = int(patch_data[idx][0]), int(patch_data[idx][1])
#                         #     self.patch_entries.append({
#                         #         'ref_path': os.path.join(ref_view_folder, "view_1.png"),
#                         #         'dis_path': os.path.join(dis_view_folder, "view_1.png"),
#                         #         'x': x,
#                         #         'y': y,
#                         #         'patch_size': patch_size,
#                         #         'judge_path': judge_path,
#                         #         'stimuli_id': stimuli_id
#                         #     })
#                     stimuli_id += 1

#     def __getitem__(self, index):
#         entry = self.patch_entries[index]

#         def load_patch(path, x, y, size):
#             img = cv2.imread(path)
#             if img is None:
#                 raise RuntimeError(f"Image introuvable : {path}")
#             img = img[:, :, ::-1]  # BGR → RGB
#             patch = img[y:y+size, x:x+size]
#             return self.transform(Image.fromarray(patch))

#         ref_patch = load_patch(entry['ref_path'], entry['x'], entry['y'], entry['patch_size'])
#         dis_patch = load_patch(entry['dis_path'], entry['x'], entry['y'], entry['patch_size'])
#         judge = torch.from_numpy(np.load(entry['judge_path'])).float().view(1, 1, 1)

#         return {
#             'ref': ref_patch,
#             'p0': dis_patch,
#             'judge': judge,
#             'stimuli_id': entry['stimuli_id'],
#             'ref_path': entry['ref_path'],
#             'p0_path': entry['dis_path'],
#             'judge_path': entry['judge_path']
#         }

#     def __len__(self):
#         return len(self.patch_entries)
import os, shutil, time, hashlib, threading
import csv
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from collections import OrderedDict
from typing import Optional
_img_cache = OrderedDict()
_img_cache_bytes = 0
_IMG_CACHE_LIMIT = 8 * 1024**3   # ~8 Go à ajuster
_img_cache_lock = threading.Lock()

cv2.setNumThreads(0) # Avoid overthreading from open-cv

def imread_cached_bgr(path: str):
    global _img_cache, _img_cache_bytes
    with _img_cache_lock:
        arr = _img_cache.get(path)
        if arr is not None:
            _img_cache.move_to_end(path)
            return arr
    buf = np.fromfile(path, dtype=np.uint8)           # Windows-safe (pas de handle persistant)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)         # BGR uint8
    if img is None:
        raise RuntimeError(f"Image introuvable : {path}")
    sz = img.nbytes
    with _img_cache_lock:
        _img_cache[path] = img
        _img_cache_bytes += sz
        while _img_cache_bytes > _IMG_CACHE_LIMIT and _img_cache:
            _, old = _img_cache.popitem(last=False)
            _img_cache_bytes -= old.nbytes
    return img

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
        except PermissionError:
            time.sleep(sleep)

    # Au pire, on lit à la source
    return str(src)

class TwoAFCDataset(Dataset):
    def __init__(self, dataroots, load_size=64, Trainset=False, maxNbPatches=205, 
                 root_refPatches=None, root_distPatches=None, src_root=None, target=None, img_ext='auto'):
        self.target = target
        self.patch_entries = []
        # self.transform = transforms.Compose([
        #     transforms.Resize(load_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # Racines des répertoires
        self.cache_root = r'C:\Graphics_LPIPS\cache'
        self.src_root = src_root
        self.root_refPatches =  self.src_root + root_refPatches
        self.root_distPatches = self.src_root + root_distPatches
        self.img_ext = img_ext
        root_judges = r'D:\These\Graphics-LPIPS\dataset\judge_trainingset' if Trainset else r'D:\These\Graphics-LPIPS\dataset\judge_testset'

        if Trainset:
            shuffled_inputfile = []
            for idx, datafile in enumerate(dataroots):
                out_path = f'D:\\These\\Graphics-LPIPS\\dataset\\Trainset_shuffled_{idx+1}.csv'
                with open(datafile, 'r') as r, open(out_path, 'w') as w:
                    lines = r.readlines()
                    header, rows = lines[0], lines[1:]
                    random.shuffle(rows)
                    w.write(header + ''.join(rows))
                shuffled_inputfile.append(out_path)
            dataroots = shuffled_inputfile

        if not isinstance(dataroots, list):
            dataroots = [dataroots]

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
                    judge_path = os.path.join(root_judges, f"{stimulus}.npy")

                    with open(patch_csv_path, newline='') as pf:
                        patch_reader = csv.reader(pf)
                        patch_header = next(patch_reader)
                        patch_data = list(patch_reader)
                        patch_size = int(patch_header[4].split('=')[1])
                        nb_patches_per_view = [int(x.split('=')[1]) for x in patch_header[7:]]

                    nb_patches_total = sum(nb_patches_per_view)
                    nb_full = maxNbPatches // nb_patches_total
                    nb_rand = maxNbPatches % nb_patches_total

                    for _ in range(nb_full):
                        view_counter = 1
                        patch_seen = 0
                        for pd in patch_data:
                            x, y = int(pd[0]), int(pd[1])
                            self.patch_entries.append({
                                'ref_path': os.path.join(ref_view_folder, f"view_{view_counter}.png"),
                                'dis_path': os.path.join(dis_view_folder, f"view_{view_counter}.png"),
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
                                'ref_path': os.path.join(ref_view_folder, f"view_{view_num}.png"),
                                'dis_path': os.path.join(dis_view_folder, f"view_{view_num}.png"),
                                'x': x,
                                'y': y,
                                'mos': mos,
                                'patch_size': patch_size,
                                'judge_path': judge_path,
                                'stimuli_id': stimuli_id
                            })
                    stimuli_id += 1

    
    def __getitem__(self, index):
        entry = self.patch_entries[index]
            
        def load_patch(path, x, y, size):
            img_path = ensure_on_ssd(path, self.src_root, self.cache_root)
            img = cv2.imread(img_path)                                  # BGR
            if img is None:
                raise RuntimeError(f"Image not found : {path}")
            patch_bgr = img[y:y+size, x:x+size]
            # (facultatif mais utile) : vérifie que le crop est bien dans l'image
            if patch_bgr.shape[0] != size or patch_bgr.shape[1] != size:
                raise ValueError(f"Patch out of bounds: img={img.shape}, "
                                f"x={x},y={y},size={size}")
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)      # contigu, strides positifs
            t = torch.from_numpy(patch_rgb).permute(2,0,1).contiguous().float().div_(255.0)
            t.sub_(0.5).div_(0.5)
            return t

        ref_patch = load_patch(entry['ref_path'], entry['x'], entry['y'], entry['patch_size'])
        dis_patch = load_patch(entry['dis_path'], entry['x'], entry['y'], entry['patch_size'])
        try:
            if self.target == "mos":
                judge = torch.tensor(entry['mos'], dtype=torch.float32).view(1,1,1)
            else:
                judge = torch.from_numpy(np.load(entry['judge_path'])).float().view(1,1,1)
        except Exception as e:
            raise RuntimeError(f"Error loading {entry['judge_path']}: {e}")

        return {
            'ref': ref_patch,
            'p0': dis_patch,
            'judge': judge,
            'mos': torch.tensor(entry['mos']).view(1, 1, 1),
            'stimuli_id': entry['stimuli_id']
        }
    def __len__(self):
        return len(self.patch_entries)
