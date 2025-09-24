
from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import zoom
from tqdm import tqdm
import lpips
import os
from scipy import stats
import statsmodels.api as sm
import collections
from itertools import groupby
from operator import itemgetter
from statistics import mean
import torch.backends.cuda as cuda_back
import torch.backends.cudnn as cudnn
import importlib.util
import contextlib

class Trainer():
    def name(self):
        return self.model_name

    def initialize(self, model='lpips', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False, 
            is_train=False, lr=.001, beta1=0.5, version='0.1', gpu_ids=[0], use_amp=False, amp_dtype="fp16"):
        '''
        INPUTS
            model - ['lpips'] for linearly calibrated network
                    ['baseline'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]'%(model,net)
        self.use_amp = use_amp and use_gpu
        self.amp_dtype = torch.bfloat16 #if str(amp_dtype).lower() in ("bf16", "bfloat16") else torch.float16
        torch.set_float32_matmul_precision('high') # use "high" precision for float32 matrix multiplications (for newer pytorch versions)
        cuda_back.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

        if(self.model == 'lpips'): # pretrained net + linear layer
            self.net = lpips.LPIPS(pretrained=not is_train, net=net, version=version, lpips=True, spatial=spatial, 
                pnet_rand=pnet_rand, pnet_tune=pnet_tune, 
                use_dropout=True, model_path=model_path, eval_mode=False)
        elif(self.model=='baseline'): # pretrained network
            self.net = lpips.LPIPS(pnet_rand=pnet_rand, net=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = lpips.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = lpips.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to map the distance d0 (average over the patches) of the stimulus image to the MOS
            self.rankLoss = lpips.BCERankingLoss()
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
            self.scaler = torch.amp.GradScaler(enabled = False) # for mixed precision training
        else: # test mode
            self.net.eval()


        if(use_gpu):
            print('Using GPU %s'%gpu_ids[0])
            self.net.to(gpu_ids[0], memory_format=torch.channels_last)
            has_triton = importlib.util.find_spec("triton") is not None
            if has_triton:
                try:
                    torch.set_float32_matmul_precision('high')
                    cuda_back.matmul.allow_tf32 = True
                    cudnn.allow_tf32 = True
                    self.net = torch.compile(self.net, mode="max-autotune") # requires pytorch 2.0
                except Exception as e:
                    print("torch.compile failed, probably because pytorch version is < 2.0. Error message: ", e)
            else:
                print("triton not installed, cannot use torch.compile. Install triton for faster training/inference (pip install triton).")
            if(len(gpu_ids) > 1):
                self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1(reference)
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''
        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.optimizer_net.zero_grad(set_to_none=True)
        self.forward_train()
        loss = torch.mean(self.loss_total)
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_net)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer_net.step()
        # self.backward_train()
        # self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        with torch.no_grad():  # désactive les gradients temporairement
            for module in self.net.modules():
                if hasattr(module, 'weight') and getattr(module, 'kernel_size', None) == (1, 1):
                    module.weight.clamp_(min=0)
    
    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_judge = data['judge']
        self.input_mos = data['mos']
        self.stimulus = data['stimuli_id']

        if self.use_gpu:
            dev = self.gpu_ids[0]
            # Si c'est déjà en CUDA (prefetcher), on ne recopie pas
            if not self.input_ref.is_cuda:
                self.input_ref = self.input_ref.to(device=dev, non_blocking=True)\
                                            .contiguous(memory_format=torch.channels_last)
            if not self.input_p0.is_cuda:
                self.input_p0  = self.input_p0.to(device=dev, non_blocking=True)\
                                            .contiguous(memory_format=torch.channels_last)
            if not self.input_judge.is_cuda:
                self.input_judge = self.input_judge.to(device=dev, non_blocking=True)
            if not torch.is_tensor(self.stimulus) or not self.stimulus.is_cuda:
                self.stimulus    = self.stimulus.to(device=dev, non_blocking=True)

        self.var_ref = self.input_ref.detach()
        self.var_p0 = self.input_p0.detach()
        # self.var_ref = Variable(self.input_ref,requires_grad=True) # Obsolete, but kept for reference
        # self.var_p0 = Variable(self.input_p0,requires_grad=True)
        
    def forward_train(self): # run forward pass
        with torch.autocast(device_type="cuda", enabled=(self.use_amp and self.use_gpu), dtype=self.amp_dtype):
            self.d0 = self.forward(self.var_ref, self.var_p0)
        # self.var_judge = Variable(1.*self.input_judge).view(self.d0.size()) # self.var_judge is the same as self.input_judge
        self.var_judge = self.input_judge.view(self.d0.size())

        # In the following: we aggregate var_judge & d0 per stimulus (over all the patches of the same stimulus)
        judge = (self.var_judge).flatten().tolist()

        mos = [mean(map(itemgetter(1), group))
            for key, group in groupby(zip(self.stimulus, judge), key=itemgetter(0))]
        
        NbuniqueStimuli = len(mos) 
        NbpatchesPerStimulus = len(judge)//NbuniqueStimuli # we selected the same nb of patches for each stimulus 
        
        self.mos = torch.tensor(mos, dtype=torch.float32, device=self.gpu_ids[0])
        self.mos = torch.reshape(self.mos, (NbuniqueStimuli,1,1,1))
        
        self.d0_reshaped = torch.reshape(self.d0, (NbuniqueStimuli,NbpatchesPerStimulus,1,1)) #(5,10,1,1) : 5 stimuli * 10 patches/stimulus => after aggregation : 5 MOS_predicted values
        self.mos_predict = torch.mean(self.d0_reshaped, 1, True)
        pred = self.mos_predict.float()
        target = self.mos.float()
        # For verification:
        # res = 0
        # for v in d0:
            # res += v
        # print('sum Lpips values %.6f'%res)
        # print('sum Lpips values/NbPatchesPerStimulus = %.6f, which must be equal to sum mos_predicted: %.6f'%(res/NbpatchesPerStimulus, torch.sum(self.mos_predict)))

        self.loss_total = self.rankLoss(pred, target) # with aggregation
       
        return self.loss_total

    def backward_train(self):
        pass
        # torch.mean(self.loss_total).backward() #torch.mean is useless since we have only one "loss_total" value/batch, and this function is excecuted per batch 

    
    def get_current_errors(self):
        loss = self.loss_total
        if isinstance(loss, torch.Tensor):
            loss_scalar = loss.detach()
            if loss_scalar.dim() != 0:
                loss_scalar = loss_scalar.mean()
            loss_value = loss_scalar.to(torch.float32).cpu().item()
        else:
            loss_value = float(loss)
        retDict = OrderedDict([('loss_total', loss_value)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.detach().size()[2]

        ref_img = lpips.tensor2im(self.var_ref.detach())
        p0_img = lpips.tensor2im(self.var_p0.detach())

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis)])                   

    def save(self, path, label):
        net_to_save = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        self.save_network(net_to_save, path, '', label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        print('Saving network to %s'%path)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr


    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')

    def to_numpy(tensor):
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        return tensor.numpy()

def Testset_DSIS(data_loader, func, funcLoss=None, name='', use_amp=False, amp_dtype=torch.bfloat16):
    """
    Évalue un DataLoader DSIS :
      - func(ref, p0) -> d0 (distance par patch)
      - funcLoss(pred, target) -> loss scalaire (optionnel)
    Attend que data_loader.load_data() renvoie un torch DataLoader.
    Chaque batch doit contenir au moins: 'ref', 'p0', 'judge', 'stimuli_id'.
    'ref'/'p0' peuvent être np.ndarray uint8 (N,C,H,W) ou torch.Tensor.
    'judge' peut être np.ndarray/tensor (N,1,1,1) ou (N,).
    """
    import contextlib
    import numpy as np
    import torch
    from tqdm import tqdm

    # --------- helpers ----------
    def _to_device_img(x, device):
        """x: np.uint8 (N,C,H,W) ou torch (N,C,H,W); return torch.float32 normalisé [-1,1] sur device."""
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            t = x
        if t.dtype == torch.uint8:
            t = t.to(device=device, non_blocking=True, dtype=torch.float32)
            t.div_(255.0).sub_(0.5).div_(0.5)  # [0,255] -> [-1,1]
        else:
            t = t.to(device=device, non_blocking=True)
            if t.dtype != torch.float32:
                t = t.float()
        return t.contiguous(memory_format=torch.channels_last)

    def _to_device_vec(x, device):
        """x: np.ndarray/tensor, renvoie torch.float32 1D (N,) sur device."""
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            t = x
        t = t.to(device=device, non_blocking=True)
        if t.dtype != torch.float32:
            t = t.float()
        return t.view(-1)

    def _to_device_ids(x, device):
        """stimuli_id en 1D long sur device (accepte list/np/tensor)."""
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, (list, tuple)):
            t = torch.tensor(x)
        else:
            t = x
        return t.to(device=device, dtype=torch.long).view(-1)

    def _group_mean(values_1d, ids_1d):
        """
        Moyenne par id (aucune hypothèse de nb constant de patches).
        values_1d: (N,), ids_1d: (N,) long.
        Retourne (mean_per_id[K], ids_unique[K])
        """
        u, inv = torch.unique(ids_1d, return_inverse=True, sorted=True)
        K = u.numel()
        sums = torch.zeros(K, device=values_1d.device, dtype=values_1d.dtype)
        cnts = torch.zeros(K, device=values_1d.device, dtype=values_1d.dtype)
        sums.scatter_add_(0, inv, values_1d)
        cnts.scatter_add_(0, inv, torch.ones_like(values_1d, dtype=values_1d.dtype))
        means = sums / torch.clamp_min(cnts, 1e-12)
        return means, u

    # --------- boucle d'éval ----------
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    tot_samples = 0
    sum_loss = 0.0
    sum_mse = 0.0
    nb_steps = 0

    all_pred = []  # liste de floats (MOS prédits par stimulus)
    all_gt   = []  # liste de floats (MOS GT par stimulus)

    dl = data_loader.load_data()

    for data in tqdm(dl, desc=name):
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if use_amp and device.type == 'cuda':
                stack.enter_context(torch.autocast(device_type="cuda", dtype=amp_dtype))

            # 1) Prépare batch (images -> normalisées; labels/ids en 1D)
            ref  = _to_device_img(data['ref'], device)     # (N,C,H,W), float32 [-1,1]
            p0   = _to_device_img(data['p0'],  device)     # (N,C,H,W), float32 [-1,1]
            gt   = _to_device_vec(data['judge'], device)   # (N,)
            sid  = _to_device_ids(data['stimuli_id'], device)  # (N,) long

            # 2) Forward par patch -> d0 (N,)
            d0 = func(ref, p0)
            d0 = _to_device_vec(d0, device)                # robust: aplati en (N,)

            # 3) Moyenne par stimulus (pas besoin de nb_patches fixe)
            mos_pred, ids_u = _group_mean(d0, sid)         # (K,), (K,)
            mos_gt,   _     = _group_mean(gt, sid)         # (K,)

            # 4) Loss / MSE (si demandée), au format attendu (K,1,1,1)
            if funcLoss is not None:
                pred4 = mos_pred.view(-1,1,1,1)            # (K,1,1,1)
                gt4   = mos_gt.view(-1,1,1,1)              # (K,1,1,1)
                loss_val = funcLoss(pred4, gt4)
                sum_loss += float(loss_val.detach().cpu())
            mse_val = torch.mean((mos_pred - mos_gt) ** 2).detach().cpu().item()
            sum_mse += mse_val

            # 5) stats / logs
            tot_samples += int(gt.numel())
            nb_steps += 1

            all_pred.extend(mos_pred.detach().cpu().tolist())
            all_gt.extend(mos_gt.detach().cpu().tolist())

    # --------- agrégations finales ----------
    # Spearman (scipy si dispo; sinon fallback)
    try:
        from scipy import stats as _scistats
        srocc = float(_scistats.spearmanr(all_pred, all_gt)[0])
    except Exception:
        # Fallback spearman (rangs) en torch
        def _rank(a):
            t = torch.tensor(a, dtype=torch.float64)
            # rangs denses stables
            vals, inv = torch.sort(t)
            ranks = torch.empty_like(inv, dtype=torch.float64)
            ranks[inv] = torch.arange(1, len(t)+1, dtype=torch.float64)
            return ranks
        rp = _rank(all_pred)
        rg = _rank(all_gt)
        rp = rp - rp.mean()
        rg = rg - rg.mean()
        srocc = float((rp @ rg) / (rp.norm() * rg.norm() + 1e-12))

    avg_loss = (sum_loss / nb_steps) if (nb_steps > 0 and funcLoss is not None) else 0.0
    avg_mse  = (sum_mse  / nb_steps) if nb_steps > 0 else 0.0

    print(f'Testset samples = {tot_samples}')
    print(f'Testset steps   = {nb_steps}')
    if funcLoss is not None:
        print(f'Testset Loss   = {avg_loss:.6f}')
    print(f'Testset MSE    = {avg_mse:.6f}')
    print(f'SROCC          = {srocc:.6f}')

    return {'loss': avg_loss, 'MSE': avg_mse, 'SROCC': srocc}

