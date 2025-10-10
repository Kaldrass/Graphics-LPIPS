import torch.backends.cudnn as cudnn
cudnn.benchmark=False
import torch
# torch.set_float32_matmul_precision("high")  # OK avec TF32/bf16

import numpy as np
import time
import os
import lpips
from data import data_loader as dl
import argparse
from util.visualizer import Visualizer
from IPython import embed
from Test_TestSet import Test_TestSet
import csv
import multiprocessing



class CUDAPrefetcher:
    def __init__(self, loader_iterable, device):
        self.loader = iter(loader_iterable)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.next = None
        self.preload()

    def preload(self):
        import numpy as np
        try:
            batch = next(self.loader)   # dict de np.ndarray
        except StopIteration:
            self.next = None
            return
        with torch.cuda.stream(self.stream):
            moved = {}
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)  # CPU tensor
                    # On "pin" pour copie H2D async
                    if t.dtype == torch.uint8 and t.dim() == 4:  # images NCHW
                        t = t.pin_memory().to(self.device, non_blocking=True).to(torch.float32)
                        # t.div_(255.0).sub_(0.5).div_(0.5) # Normalisation is done later, not needed to do it here
                        t = t.contiguous(memory_format=torch.channels_last)
                    elif t.dtype == torch.float32:
                        t = t.pin_memory().to(self.device, non_blocking=True)
                    else:
                        t = t.pin_memory().to(self.device, non_blocking=True)
                    moved[k] = t
                elif torch.is_tensor(v):
                    # cas improbable si un tensor passe quand même
                    t = v
                    if t.device.type == 'cpu':
                        t = t.pin_memory().to(self.device, non_blocking=True)
                    else:
                        t = t.to(self.device, non_blocking=True)
                    if t.dim() == 4 and t.dtype == torch.float32:
                        t = t.contiguous(memory_format=torch.channels_last)
                    moved[k] = t
                else:
                    moved[k] = v
            self.next = moved

    def __iter__(self): return self
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next is None: raise StopIteration
        batch = self.next
        self.preload()
        return batch
    
# collate function to handle numpy arrays in the batch (for CUDAPrefetcher), avoid shared memory issue, return numpa arrays, not tensors
def collate_to_numpy(batch):
    import numpy as np
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ('ref', 'p0', 'judge', 'mos'):
            arrs = []
            for v in vals:
                if torch.is_tensor(v):
                    arrs.append(v.numpy())               # tensor CPU -> np
                elif isinstance(v, np.ndarray):
                    arrs.append(v)
                else:
                    arrs.append(np.asarray(v))
            out[k] = np.stack(arrs, axis=0)              # N,C,H,W ou N,1,1,1
        else:
            out[k] = np.array(vals)                      # ids etc.
    return out


os.environ['PYTHONWARNINGS'] = 'ignore'
train_name = 'TMQ_OR_1VP_org_dbg'
train_view_nbr = 1
target = 'judges'#'judges'  # 'mos' or 'judges', for TMQ put judges
view_method = 'Original' # 'Fibonacci', 'Y_fixed_0.3', 'Polyhedron', 'Original'
render_method = 'Old_Render' # 'New_Render' or 'Old_render'
database = 'TMQ' # 'TSMD' or 'BASICS(PC)_DB' or 'TMQ'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=['./dataset/TexturedDB_80%_TrainList_withnbPatchesPerVP_threth0.6.csv', './dataset/TSMD/TSMD_80%_TrainList_scaled.csv'], help='datasets to train on')
    parser.add_argument('--testcsv', type=str, nargs='+', default=['./dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv', './dataset/TSMD/TSMD_20%_TestList_scaled.csv'], help='datasets to test on')
    

    parser.add_argument('--src_root', type=str, nargs='+', default="D:\\These\\Projets\\CompareMetrics\\out\\"+ database +"\\"+ render_method +"\\" + view_method, help='root folder containing ref and dist folders')
    parser.add_argument('--root_refPatches', type=str, nargs='+', default="\\Source\\"+ str(train_view_nbr) +'VP', help='reference patches relative location')
    parser.add_argument('--root_distPatches', type=str, nargs='+', default="\\Distorted\\" + str(train_view_nbr) + 'VP', help='distorted patches relative location')

    parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
    parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
    #parser.add_argument('--batch_size', type=int, default=50, help='batch size to test image patches in')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU', default=True)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')

    parser.add_argument('--nThreads', type=int, default=16, help='number of threads to use in data loader') 
    
    parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
    parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
    parser.add_argument('--npatches', type=int, default=150, help='# randomly sampled image patches')
    parser.add_argument('--nInputImg', type=int, default=4, help='# stimuli/images in each batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='# initial learning rate')
    
    parser.add_argument('--testset_freq', type=int, default=2, help='frequency of evaluating the testset')
    parser.add_argument('--display_freq', type=int, default=50000, help='frequency (in instances) of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=50000, help='frequency (in instances) of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency (in instances) of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom display, [0] for no displaying')
    parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
    parser.add_argument('--display_port', type=int, default=8001,  help='visdom display port')
    parser.add_argument('--use_html', action='store_true', help='save off html pages')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory')
    parser.add_argument('--name', type=str, default=train_name, help='directory name for training')

    parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
    parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
    parser.add_argument('--train_plot', action='store_true', help='plot saving')

    opt = parser.parse_args()
    opt.batch_size = opt.npatches * opt.nInputImg
    
    opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)
    # initialize model
    trainer = lpips.Trainer()
    # trainer.initialize(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, is_train=True, lr=opt.lr,
    #   pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk, gpu_ids=opt.gpu_ids)
    trainer.initialize(model=opt.model, net=opt.net, use_gpu=True, is_train=True, lr=opt.lr,
        pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk, gpu_ids=[0])
    # trainer.set_precision_flags(use_amp=False, amp_dtype=None, enable_tf32=False)
    
    print("Model on:", next(trainer.net.parameters()).device)
    # print("[AMP]", trainer.use_amp, trainer.amp_dtype)
    load_size = 64 # default value is 64

    visualizer = Visualizer(opt)

    # load data from all test sets 
    # The random patches for the test set are only sampled once at the beginning of training in order to avoid noise in the validation loss.
    Testset = opt.testcsv[1 if target=='mos' else 0]
    data_loader_testSet = dl.CreateDataLoader(Testset,dataset_mode='2afc', Nbpatches= opt.npatches, 
                                              pin_memory=False, drop_last=False, prefetch_factor=None, nThreads=0,
                                              src_root=opt.src_root, root_refPatches=opt.root_refPatches, root_distPatches=opt.root_distPatches,
                                              target = target) 
    test_TestSet = Test_TestSet(opt)
    total_steps = 0
    # fid = open(os.path.join(opt.checkpoints_dir,opt.name,'train_log.txt'),'w+')
    # f_hyperParam = open(os.path.join(opt.checkpoints_dir,opt.name,'tuning_hyperparam.csv'),'a') 
    # if os.stat(os.path.join(opt.checkpoints_dir,opt.name,'tuning_hyperparam.csv')).st_size == 0:
        # f_hyperParam.write("nepoch,nepoch_decay,npatches,nInputImg,lr,epoch,TrainLoss,testLoss,SROCC_testset\n")
    
    start_time = time.time()
    print('Start training with the following options:')
    for k, v in sorted(vars(opt).items()):  
        print('%s: %s' % (str(k), str(v)))
    print('Total number of patches: %d, batch size: %d, input images per batch: %d' % (opt.npatches, opt.batch_size, opt.nInputImg))
    print('Total number of epochs: %d, learning rate: %.6f' % (opt.nepoch + opt.nepoch_decay, opt.lr))



    for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
            # Load training data to sample random patches every epoch
            data_loader = dl.CreateDataLoader(opt.datasets[1 if target=='mos' else 0],dataset_mode='2afc', trainset=True, Nbpatches=opt.npatches, 
                                        load_size = load_size, batch_size=opt.batch_size, serial_batches=True, nThreads=opt.nThreads, 
                                                pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True, # prefetch_factor=2,
                                        src_root=opt.src_root, root_refPatches=opt.root_refPatches, root_distPatches=opt.root_distPatches, 
                                        target=target)
            dataset = data_loader.load_data()
            dataset_size = len(data_loader)
            D = len(dataset)
    
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            num_batches = len(dataset)
            num_samples = len(dataset.dataset)
            # print('Epoch %d, dataset size: %d' % (epoch, dataset_size))<
            print(f'Epoch {epoch}, batches: {num_batches}, samples: {num_samples}, bs={opt.batch_size}, workers={opt.nThreads}')

            device = torch.device('cuda:0')
            prefetch = CUDAPrefetcher(dataset, device)

            epoch_start_time = time.time()
            nb_batches = 0 
            Loss_trainset = 0 
            for i, data in enumerate(prefetch): 
                iter_start_time = time.time()
                total_steps += opt.batch_size
                epoch_iter = total_steps - dataset_size * (epoch - 1)

                trainer.set_input(data)
                if i == 0:
                    try:
                        pdev = next(trainer.net.parameters()).device
                    except StopIteration:
                        pdev = "no-params"
                    print(f"[DEV] torch.cuda.is_available={torch.cuda.is_available()} | "
                        f"net={pdev} | ref={trainer.ref.device} | p0={trainer.p0.device} | "
                        f"amp={trainer.use_amp} dtype={trainer.amp_dtype}")
                    print("ref dtype/range:", trainer.ref.dtype, float(trainer.ref.min()), float(trainer.ref.max()))
                    assert torch.cuda.is_available()
                    assert str(pdev).startswith("cuda")
                    assert str(trainer.ref.device).startswith("cuda")
                    assert str(trainer.p0.device).startswith("cuda")
                if i%50 == 0:
                    print('Epoch %d, Batch %d / %d, Total Steps %d' % (epoch, i, dataset_size, total_steps))
                trainer.optimize_parameters()

                # if total_steps % opt.display_freq == 0:
                #     visualizer.display_current_results(trainer.get_current_visuals(), epoch)

                errors = trainer.get_current_errors() # current error per batch
                Loss_trainset += errors['loss_total'] # total loss over trainset = sum(Loss/batch)/nb_batches
                nb_batches += 1 
                # torch.cuda.empty_cache()
                # if total_steps % opt.print_freq == 0:
                    # t = (time.time()-iter_start_time)/opt.batch_size
                    # t2o = (time.time()-epoch_start_time)/3600.
                    # t2 = t2o*D/(i+.0001)
                    # visualizer.print_current_errors(epoch, epoch_iter, errors, t, t2=t2, t2o=t2o, fid=fid)

                    #for key in errors.keys():
                        #visualizer.plot_current_errors_save(epoch, float(epoch_iter)/dataset_size, opt, errors, keys=[key,], name=key, to_plot=opt.train_plot)

                    # if opt.display_id > 0:
                        # visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

                # if total_steps % opt.save_latest_freq == 0:
                    # print('saving the latest model (epoch %d, total_steps %d)' %(epoch, total_steps))
                    # trainer.save(opt.save_dir, 'latest')

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                trainer.save(opt.save_dir, 'latest')
                trainer.save(opt.save_dir, epoch)
                
                print('nb batch %.1f'%nb_batches)
                Loss_trainset = Loss_trainset/nb_batches
                print('Epoch Loss %.6f'%Loss_trainset)
                resPerEpoch = dict([('Trainset_Totalloss', Loss_trainset)])
                
                for key in resPerEpoch.keys():
                    visualizer.plot_current_errors_save(epoch, float(0), opt, resPerEpoch, keys=[key,], name=key, to_plot=opt.train_plot)


            # Evaluate the Test set at the End of the epoch
            if epoch % opt.testset_freq == 0:
                # --- clean loader unwrap ---
                ld = data_loader_testSet
                if hasattr(ld, "load_data"):
                    tmp = ld.load_data()
                    if hasattr(tmp, "__iter__"):
                        ld = tmp
                    elif hasattr(tmp, "dataloader"):
                        ld = tmp.dataloader
                else:
                    ld = getattr(ld, "dataloader", ld)

                # --- evaluation ---
                res_testset = trainer.Testset_DSIS(ld)
                print(f"[TestSet] SROCC={res_testset['SROCC']:.4f}")

                with torch.no_grad():
                    if "mos_pred" in res_testset and "mos_true" in res_testset:
                        pred = torch.from_numpy(res_testset["mos_pred"]).to(trainer.device)
                        true = torch.from_numpy(res_testset["mos_true"]).to(trainer.device)
                        test_loss = float(trainer.rankLoss(pred, true).mean().item())
                    else:
                        # fallback: on n'a que la loss agrégée legacy, ou rien → NaN
                        test_loss = float(res_testset.get("loss", float("nan")))
                print(f"[TestSet] loss={test_loss:.6f}")

                # --- plotting ---
                res_plot = {
                    "SROCC": float(res_testset["SROCC"]),
                    "loss":  test_loss,                 
                }
                keys_to_plot = ["SROCC", "loss"]

                test_TestSet.plot_TestSet_save(
                    epoch=epoch,
                    res=res_plot,
                    keys=keys_to_plot,
                    name="TestSet",                     
                    to_plot=opt.train_plot,
                    what_to_plot="TestSet_Res",
                )

                # --- logging CSV/Text ---
                info = (
                    f"{opt.nepoch},{opt.nepoch_decay},{opt.npatches},{opt.nInputImg},"
                    f"{opt.lr},{epoch},{Loss_trainset},{test_loss},{res_testset['SROCC']}\n"
                )
            else:
                info = (
                    f"{opt.nepoch},{opt.nepoch_decay},{opt.npatches},{opt.nInputImg},"
                    f"{opt.lr},{epoch},{Loss_trainset}\n"
                )

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

            #f_hyperParam.write(info)
            
            if epoch > opt.nepoch:
                trainer.update_learning_rate(opt.nepoch_decay)

    # trainer.save_done(True)
    # fid.close()
    #f_hyperParam.close()
    print( 'End of %d epochs. Time taken: %d sec' %(opt.nepoch + opt.nepoch_decay,  time.time() -  start_time))
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
    # embed()  # Uncomment to debug with IPython