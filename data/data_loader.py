def CreateDataLoader(InputData,dataroot='./dataset',dataset_mode='2afc', trainset=False, Nbpatches= 205, load_size=64,batch_size=1,
                     serial_batches=True,nThreads=4, pin_memory=True, root_refPatches=None, root_distPatches=None, src_root = None, 
                     target=None, **dl_kwargs):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    allowed = {
        "pin_memory", "persistent_workers", "prefetch_factor",
        "timeout", "worker_init_fn", "generator", "collate_fn"
    }
    dl_kwargs = {k: v for k, v in dl_kwargs.items() if k in allowed}
    if not nThreads or nThreads <= 0:
        dl_kwargs.pop("persistent_workers", None)
        dl_kwargs.pop("prefetch_factor", None)
    data_loader = CustomDatasetDataLoader()
    worker_init_fn=lambda _: print("Worker started")
    #data_loader.initialize(InputData,dataroot=dataroot+'/'+dataset_mode,dataset_mode=dataset_mode,load_size=load_size,batch_size=batch_size,serial_batches=serial_batches, nThreads=nThreads)
    data_loader.initialize(data_csvfile=InputData, trainset=trainset, Nbpatches=Nbpatches, dataset_mode=dataset_mode,
                           load_size=load_size,batch_size=batch_size,serial_batches=serial_batches, nThreads=nThreads, 
                           pin_memory=pin_memory, root_refPatches=root_refPatches, root_distPatches=root_distPatches, 
                           src_root=src_root, target=target, **dl_kwargs)
    return data_loader
