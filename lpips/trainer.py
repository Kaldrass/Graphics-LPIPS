
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


def Testset_DSIS(data_loader, func, funcLoss = None, name=''): #added by yana
    total = 0
    SROCC = 0
    val_loss = 0
    val_MSE = 0
    val_steps = 0
    d0s = []
    gts = []
    MOSpredicteds = []
    MOSs = []
    
    device = torch.device("cuda:0")
    for data in tqdm(data_loader.load_data(), desc=name):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            ref = data['ref'].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            p0 = data['p0'].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            d0 = func(ref, p0)  # use_gpu flag activated
            gt = data['judge'].to(device, non_blocking=True)

            stimulus = data['stimuli_id']
            #stimulus = [p0path.split("\\PlaylistsStimuli_patches_withVP\\")[1] for p0path in data['p0_path']]
            #stimulus = [stim.rsplit("_P", 1)[0] for stim in stimulus] # split according to the last occurence of "_P"

            gt_ = (gt).cpu().numpy().flatten().tolist()
        
            mos = [mean(map(itemgetter(1), group))
                for key, group in groupby(zip(stimulus, gt_), key=itemgetter(0))]
            NbuniqueStimuli = len(mos) 
            NbpatchesPerStimulus = len(gt_)//NbuniqueStimuli 
        
            MOS = torch.tensor(mos, dtype=torch.float32, device=device) #fp32
            MOS = torch.reshape(MOS, (NbuniqueStimuli,1,1,1))
        
            d0_reshaped = torch.reshape(d0, (NbuniqueStimuli,NbpatchesPerStimulus,1,1)) 
            MOSpredicted = torch.mean(d0_reshaped, 1, True).float() #fp32
            
            loss = funcLoss(MOSpredicted, MOS) 
            val_loss += loss.detach().cpu().item() #detach().numpy() (if we remove "with torch.no_grad():" )
            
            # compute MSE manually
            MSE = torch.mean((MOSpredicted-MOS) ** 2).detach().cpu().item() 
            val_MSE += MSE
            
            total += gt.size(0)
            val_steps += 1
            
            # # concatenate data to compute SROCC
            # MOSpredicteds += MOSpredicted
            # MOSs += MOS
            ####### TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            MOSpredicteds += MOSpredicted.detach().float().cpu().numpy().flatten().tolist()
            MOSs += MOS.detach().float().cpu().numpy().flatten().tolist()
    srocc = stats.spearmanr(MOSpredicteds, MOSs)[0]
    loss = val_loss / val_steps
    MSE = val_MSE / val_steps
    
    print('Testset Total %.3f'%total)
    print('Testset val step = nb batches =  %.3f'%val_steps)
    print('Testset Loss %.3f'%loss)
    print('Testset MSE %.3f'%MSE)
    print('SROCC %.3f'%srocc)
    
    resDict = dict([('loss', loss),
                    ('MSE', MSE),
                    ('SROCC', srocc)])
    return(resDict)

