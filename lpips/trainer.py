import os
import math
import time
from collections import defaultdict
from typing import Dict, Any, Iterable, Tuple
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from typing import Optional
from itertools import groupby
from operator import itemgetter
from statistics import mean
from scipy import stats

import tqdm  # <-- ajoute ceci à tes imports
# -----------------------------------------------------------------------------
# Trainer — version corrigée avec backward, AMP bf16/fp16, TF32, channels_last,
#           agrégation par stimulus robuste (nb de patches variable),
#           évaluation DSIS tolérante aux formats numpy/torch.
# -----------------------------------------------------------------------------


class Trainer:
    def __init__(
            self,
            net: str = 'alex',
            rankLoss: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            device: torch.device | str = "cuda",
            use_amp: bool = False,
            amp_dtype: str = "float16",  # "bfloat16" ou "float16"
            enable_tf32: bool = False,
            compile_model: bool = False,

        ) -> None:
            self.device = torch.device(device)

            # Autoriser la construction vide (compat avec l'ancien train.py)
            self.optimizer = optimizer  # peut rester None jusqu'à initialize()

            # Options précision
            self.use_amp = bool(use_amp)
            self.amp_dtype = (
                torch.bfloat16 if str(amp_dtype).lower() in ("bf16", "bfloat16") else torch.float16
            )

            if enable_tf32:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True if enable_tf32 else False
                    torch.backends.cudnn.allow_tf32 = True if enable_tf32 else False
                except Exception:
                    pass

            # GradScaler : utile seulement en fp16
            if self.use_amp and self.amp_dtype is torch.float16:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                self.scaler = torch.amp.GradScaler("cuda", enabled=False)

            # (Optionnel) torch.compile — seulement si un vrai net a été fourni
           
            self.legacy_mode = True
            if self.legacy_mode:
                self.use_amp = False
                self.amp_dtype = None
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                except Exception:
                    pass
            # Tensors courants injectés par set_input()
            self.ref = None
            self.p0 = None
            self.stimulus = None
            self.judge = None
            self.loss_total = None
            self.mos_predict = None
            self.mos = None
            self.fixed_patches_per_stimulus = None  # for old reshaping compatibility
            self.use_gpu = True
    def initialize(self, model='lpips', net='alex', colorspace='Lab',
               pnet_rand=False, pnet_tune=False, model_path=None,
               use_gpu=True, printNet=False, spatial=False,
               is_train=False, lr=.001, beta1=0.5, version='0.1', gpu_ids=[0]):
        """
        Shim de compatibilité pour l'ancienne API.
        Reconstruit self.net / self.rankLoss / self.optimizer comme avant.
        """
        import torch
        try:
            import lpips
        except Exception as e:
            raise RuntimeError("Le module 'lpips' est requis par initialize().") from e

        # Etat
        self.net = net
        self.model = model
        self.is_train = bool(is_train)
        self.spatial = bool(spatial)
        self.model_name = f"{model} [{net}]"

        # Device / GPU
        if use_gpu and torch.cuda.is_available():
            dev = torch.device(f"cuda:{gpu_ids[0] if gpu_ids else 0}")
        else:
            dev = torch.device("cpu")
        self.device = dev  # maj du device du trainer

        # ---- Construction du réseau métrique ----
        if model.lower() == 'lpips':            # LPIPS calibré linéaire
            self.net = lpips.LPIPS(
                pretrained=not self.is_train,
                net=net,
                version=version,
                lpips=True,
                spatial=spatial,
                pnet_rand=pnet_rand,
                pnet_tune=pnet_tune,
                use_dropout=True,
                model_path=model_path,
                eval_mode=False
            )
        elif model.lower() == 'baseline':        # backbone off-the-shelf
            self.net = lpips.LPIPS(
                pnet_rand=pnet_rand,
                net=net,
                lpips=False
            )
        elif model.lower() in ['l2']:
            self.net = lpips.L2(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'L2'
        elif model.lower() in ['dssim', 'ssim']:
            self.net = lpips.DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError(f"Model [{model}] not recognized.")

        # Paramètres entraînables
        self.parameters = list(self.net.parameters())

        # ---- Mode train/test + loss + optimizer ----
        if self.is_train:
            self.rankLoss = lpips.BCERankingLoss()
            self.lr = lr
            self.old_lr = lr
            self.optimizer = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()


        # ---- Placement device et DataParallel  ----
        self.net = self.net.to(dev)
        if use_gpu and torch.cuda.is_available() and len(gpu_ids) > 0:
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if self.is_train and self.rankLoss is not None:
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0])
        else:
            if self.is_train and self.rankLoss is not None:
                self.rankLoss = self.rankLoss.to(dev)

        # Optionnel: impression du réseau (si dispo)
        if printNet:
            try:
                from networks import print_network
                print('---------- Networks initialized -------------')
                print_network(self.net)
                print('--------------------------------------------')
            except Exception:
                # pas bloquant si util manquant
                pass

        return self
    # ----------------------------- Entrées / Préproc -------------------------
    @torch.no_grad()

    def set_input(self, input_dict):
        # Tensors CPU/GPU possibles selon ton prefetcher → mets-les sur device ensuite.
        self.ref = input_dict["ref"].to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        self.p0  = input_dict["p0"].to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        self.judge = input_dict["judge"].to(self.device, dtype=torch.float32, non_blocking=True).view(-1)
        self.stimulus = (input_dict.get("stimulus", input_dict.get("stimuli_id"))).to(self.device, dtype=torch.long, non_blocking=True).view(-1)
        if not hasattr(self, "_io_dbg"):
            print("[DBG] ref range after dataset:", float(self.ref.min()), float(self.ref.max()), self.ref.dtype)
            self._io_dbg = True
        self.var_ref = Variable(self.ref,requires_grad=True)
        self.var_p0 = Variable(self.p0,requires_grad=True)
    # ----------------------------- Forward/Backward --------------------------
    def _ensure_loss_tensor(self, loss: Any) -> torch.Tensor:
        if loss is None:
            raise RuntimeError("loss_total is None – nothing to backprop.")
        if isinstance(loss, (list, tuple)):
            loss = sum(
                l if isinstance(l, torch.Tensor) else torch.tensor(float(l), device=self.device)
                for l in loss
            )
        elif not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), device=self.device)
        if loss.dim() != 0:
            loss = loss.mean()
        return loss

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1(reference)
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''
        return self.net.forward(in0, in1, retPerLayer=retPerLayer)
    def forward_train(self):
       
        self.d0 = self.forward(self.var_ref, self.var_p0)  # [N, 1] ou [N]
        d0 = self.d0
      
        self.var_judge = Variable(1.0 * self.judge).view(d0.size())
      
        judge_list = self.var_judge.detach().flatten().cpu().tolist()
        mos = [mean(map(itemgetter(1), group))
            for key, group in groupby(zip(self.stimulus, judge_list), key=itemgetter(0))]
        if isinstance(self.stimulus, torch.Tensor):
            stimulus_list = self.stimulus.detach().flatten().cpu().tolist()
        else:
            stimulus_list = list(self.stimulus)


        # mos_true_list = []
        # for key, group in groupby(zip(stimulus_list, judge_list), key=itemgetter(0)):
        #     vals = [v for _, v in group]
        #     mos_true_list.append(sum(vals) / float(len(vals)))

        NbuniqueStimuli = len(mos)
        # N = d0.numel()
        NbpatchesPerStimulus = len(judge_list) // NbuniqueStimuli

        target_device = (
            self.gpu_ids[0] if hasattr(self, "gpu_ids") and len(self.gpu_ids) > 0 else d0.device
        )

        self.mos = torch.tensor(mos, dtype=torch.float32, device=target_device)
        self.mos = torch.reshape(self.mos, (NbuniqueStimuli, 1, 1, 1))

        d0_flat = d0.view(-1)  # [N]
        self.d0_reshaped = torch.reshape(
            d0_flat, (NbuniqueStimuli, NbpatchesPerStimulus, 1, 1)
        )  # ex: (5,10,1,1)

        # self.mos_predict : [NbStimuli,1,1,1]
        self.mos_predict = torch.mean(self.d0_reshaped, dim=1, keepdim=True)

        self.loss_total = self.rankLoss.forward(self.mos_predict, self.mos)

        return self.loss_total

    def backward_train(self):
        loss = self.loss_total
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, dtype=torch.float32, device=getattr(self, "device", None))
        torch.mean(loss).backward()
        
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer.zero_grad()
        self.backward_train()
        self.optimizer.step()
        self.clamp_weights()
        
    def clamp_weights(self):
        with torch.no_grad():
            for m in self.net.modules():
                if hasattr(m, "weight") and hasattr(m, "kernel_size") and m.kernel_size == (1,1):
                    m.weight.data = torch.clamp(m.weight.data, min=0.0)
    # ------------------------------- Évaluation (legacy) ------------------------------
    @torch.inference_mode()
    def Testset_DSIS(self, loader, name="Test"):

        val_loss_sum = 0.0
        val_mse_sum  = 0.0
        val_steps    = 0

        MOSpredicteds_f = []  # floats
        MOSs_f          = []  # floats

        device = self.gpu_ids[0] if hasattr(self, "gpu_ids") and len(self.gpu_ids) > 0 else "cuda:0"

        for data in tqdm.tqdm(loader, desc=name):
            # Déballage & device
            ref = data["ref"].to(device, non_blocking=True)
            p0  = data["p0"].to(device, non_blocking=True)
            gt  = data["judge"].to(device, non_blocking=True)
            stimulus = data["stimuli_id"]  # reste CPU pour le groupby Python

            # Réseau via le wrapper legacy
            d0 = self.forward(ref, p0).to(device)  # [N] ou [N,1] → on aplati ensuite

            # --- Agrégation 'legacy' (groupby Python) ---
            gt_list = gt.detach().cpu().numpy().flatten().tolist()
            # groupby exige contiguïté par stimulus (comme l'ancien)
            mos_list = [mean(map(itemgetter(1), group))
                        for key, group in groupby(zip(stimulus, gt_list), key=itemgetter(0))]
            NbStim = len(mos_list)
            NbPatchesPerStim = len(gt_list) // NbStim

            MOS = torch.tensor(mos_list, dtype=torch.float32, device=device).view(NbStim, 1, 1, 1)
            d0_reshaped = d0.view(-1).view(NbStim, NbPatchesPerStim, 1, 1)
            MOSpred = torch.mean(d0_reshaped, dim=1, keepdim=True)  # [NbStim,1,1,1]

            # Loss (si dispo)
            if hasattr(self, "rankLoss") and callable(getattr(self.rankLoss, "forward", None)):
                loss = self.rankLoss.forward(MOSpred, MOS)
                val_loss_sum += float(loss.detach().cpu().numpy())
            else:
                loss = None

            # MSE manuel
            mse = (MOSpred - MOS) ** 2
            val_mse_sum += float(mse.mean().detach().cpu().numpy())

            val_steps += 1

            # --- Accumulation en floats (évite les tensors dans des listes) ---
            MOSpredicteds_f.extend(MOSpred.detach().cpu().numpy().flatten().tolist())
            MOSs_f.extend(MOS.detach().cpu().numpy().flatten().tolist())

        # Statistiques finales
        srocc = stats.spearmanr(MOSpredicteds_f, MOSs_f)[0] if len(MOSs_f) > 1 else np.nan
        loss_mean = val_loss_sum / max(val_steps, 1)
        mse_mean  = val_mse_sum  / max(val_steps, 1)

        print(f"[{name}] steps={val_steps}  Loss={loss_mean:.6f}  MSE={mse_mean:.6f}  SROCC={srocc:.6f}")

        return {"loss": loss_mean, "MSE": mse_mean, "SROCC": float(srocc)}

    # ------------------------------ Sauvegarde --------------------------------
    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)
    # ------------------------------ Utilitaires -------------------------------
    def update_learning_rate(self, nepoch_decay: int):
        """
        Décroissance linéaire simple : à chaque appel on retranche lr/nepoch_decay.
        Attend que self.lr, self.old_lr et self.optimizer existent (set par initialize()).
        """
        if self.optimizer is None:
            raise RuntimeError("update_learning_rate: optimizer is None. Appelle initialize(...) d'abord.")
        if nepoch_decay is None or nepoch_decay <= 0:
            # rien à faire si pas de phase de décroissance
            return self.optimizer.param_groups[0]["lr"]

        # Valeurs de référence (si pas déjà posées)
        if not hasattr(self, "lr"):
            self.lr = float(self.optimizer.param_groups[0]["lr"])
        if not hasattr(self, "old_lr"):
            self.old_lr = float(self.optimizer.param_groups[0]["lr"])

        lrd = float(self.lr) / float(nepoch_decay)   # quantité à retrancher à chaque epoch de décroissance
        lr = max(self.old_lr - lrd, 0.0)             # évite les LR négatifs

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        name = getattr(self, "model_name", "LPIPS")
        print("update lr [%s] decay: %.6f -> %.6f" % (name, self.old_lr, lr))
        self.old_lr = lr
        return lr
    
    def get_current_errors(self) -> Dict[str, float]:
        out = {}
        if isinstance(self.loss_total, torch.Tensor):
            try:
                out["loss_total"] = float(self.loss_total.detach().float().mean().item())
            except Exception:
                out["loss_total"] = float(self.loss_total.mean().item())
        if isinstance(self.mos_predict, torch.Tensor):
            out["mos_pred_mean"] = float(self.mos_predict.detach().float().mean().item())
        if isinstance(self.mos, torch.Tensor):
            out["mos_true_mean"] = float(self.mos.detach().float().mean().item())
        return out
    def _denorm_to_uint8(self, x: torch.Tensor) -> np.ndarray:
        """
        x: Tensor [N,C,H,W] ou [C,H,W], range [-1,1] ou [0,1].
        Retourne un numpy uint8 [H,W,C] (B=1er élément si batch).
        """
        if x is None:
            raise RuntimeError("Aucune image disponible (self.ref/self.p0 est None).")
        if x.ndim == 3:
            x = x.unsqueeze(0)  # -> [1,C,H,W]
        # si ce sont des uint8 CHW (cas dataset), convertis en float et normalise
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        # remet en [0,1] si besoin (on suppose [-1,1] après set_input)
        if x.min() < 0.0 or x.max() > 1.0:
            x = (x + 1.0) * 0.5
        x = x.clamp(0, 1)

        # on ne prend que le 1er élément du batch pour l’aperçu
        x0 = x[0].detach().to("cpu")
        # CHW -> HWC
        x0 = x0.permute(1, 2, 0).contiguous().numpy()
        x0 = (x0 * 255.0 + 0.5).astype(np.uint8)
        return x0

    def get_current_visuals(self, k: int = 1) -> dict:
        """
        Retourne un petit dict d'images pour l'affichage (compatible anciens visualizers).
        - 'ref': première image ref du batch en uint8 HxWxC
        - 'p0' : première image disto du batch en uint8 HxWxC
        - 'diff' (optionnel) : |ref - p0| en uint8 pour repérage
        """
        if self.ref is None or self.p0 is None:
            # compatible avec les anciens scripts: retourner un dict vide au lieu de lever
            return {}

        ref_img = self._denorm_to_uint8(self.ref)
        p0_img  = self._denorm_to_uint8(self.p0)

        # diff simple (même forme HxWxC)
        diff = np.abs(ref_img.astype(np.int16) - p0_img.astype(np.int16)).astype(np.uint8)

        return {
            "ref": ref_img,
            "p0": p0_img,
            "diff": diff
        }
