from functools import partial

from torch.utils.data import DataLoader, DistributedSampler

from datasets import Cityscapes, Kitti360, SemKittiDvps
from datasets.replay_buffer import ReplayBuffer
from io_utils import logging
from misc.train_utils import collate_batch


def gen_train_dataloader(args, cfg, rank, world_size) -> DataLoader:
    logging.log_info("Creating adaptation dataloader...", debug=args.debug)

    if cfg.adapt.mode in ["online", "off"]:
        if cfg.dataset.name == "sem_kitti_dvps":
            target_dataset = SemKittiDvps("sequence", cfg.dataset, return_depth=True,
                                          sequences=cfg.dataset.sequences)
        elif cfg.dataset.name == "kitti_360":
            target_dataset = Kitti360("sequence", cfg.dataset, return_depth=True,
                                      sequences=cfg.dataset.sequences,
                                      sequence_reference_mode="rgb")
        else:
            raise NotImplementedError(f"Dataset {cfg.dataset.name} is not yet implemented")
        if cfg.adapt.source_dataset.name == "cityscapes":
            source_dataset = Cityscapes("train", cfg.adapt.source_dataset)
        else:
            raise NotImplementedError(f"Source dataset {cfg.adapt.source_dataset.name} is not "
                                      f"yet implemented")
        batch_size = 1
        shuffle = False
        source_dataset_samples = cfg.adapt.replay_buffer.source_size

    else:
        raise ValueError(f"Unsupported mode: {cfg.adapt.mode}")

    replay_buffer = ReplayBuffer(target_dataset, cfg.adapt, mode="train")
    replay_buffer.initialize_source_buffer(source_dataset, source_dataset_samples)

    collate_fn = partial(collate_batch, samples_per_gpu=batch_size)
    if not args.debug:
        adapt_sampler = DistributedSampler(replay_buffer, world_size, rank, shuffle=shuffle)
        adapt_dl = DataLoader(replay_buffer,
                              sampler=adapt_sampler,
                              collate_fn=collate_fn,
                              batch_size=batch_size,
                              pin_memory=True,
                              num_workers=cfg.train.nof_workers_per_gpu)
    else:
        adapt_dl = DataLoader(replay_buffer,
                              shuffle=shuffle,
                              collate_fn=collate_fn,
                              batch_size=batch_size,
                              pin_memory=True,
                              num_workers=cfg.train.nof_workers_per_gpu)

    return adapt_dl


def gen_val_dataloader(args, cfg, rank, world_size) -> DataLoader:
    logging.log_info("Creating val dataloader...", debug=args.debug)

    if cfg.adapt.mode in ["online", "off"]:
        if cfg.dataset.name == "sem_kitti_dvps":
            target_dataset = SemKittiDvps("sequence", cfg.dataset, return_depth=True,
                                          sequences=cfg.dataset.sequences)
        elif cfg.dataset.name == "kitti_360":
            target_dataset = Kitti360("sequence", cfg.dataset, return_depth=True,
                                      sequences=cfg.dataset.sequences,
                                      sequence_reference_mode="semantic")
        else:
            raise NotImplementedError(f"Dataset {cfg.dataset.name} is not yet implemented")
        dataset_val = ReplayBuffer(target_dataset, cfg.adapt, mode="val")

    else:
        raise ValueError(f"Unsupported mode: {cfg.adapt.mode}")

    collate_fn_val = partial(collate_batch, samples_per_gpu=cfg.val.batch_size_per_gpu)
    if not args.debug:
        val_sampler = DistributedSampler(dataset_val, world_size, rank, shuffle=True)
        val_dl = DataLoader(dataset_val,
                            sampler=val_sampler,
                            batch_size=cfg.val.batch_size_per_gpu,
                            collate_fn=collate_fn_val,
                            pin_memory=True,
                            num_workers=cfg.val.nof_workers_per_gpu)
    else:
        val_dl = DataLoader(dataset_val,
                            batch_size=cfg.val.batch_size_per_gpu,
                            collate_fn=collate_fn_val,
                            pin_memory=True,
                            num_workers=cfg.val.nof_workers_per_gpu)

    return val_dl

def gen_source_val_dataloader(args, cfg, rank, world_size) -> DataLoader:
    logging.log_info("Creating source val dataloader...", debug=args.debug)

    # Note the augmentation is automatically disabled when calling a dataset with mode="val"
    if cfg.adapt.source_dataset.name == "cityscapes":
        dataset_val = Cityscapes("val", cfg.adapt.source_dataset, return_depth=True)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} is not yet implemented")

    collate_fn_val = partial(collate_batch, samples_per_gpu=cfg.val.batch_size_per_gpu)
    if not args.debug:
        val_sampler = DistributedSampler(dataset_val, world_size, rank, shuffle=False)
        val_dl = DataLoader(dataset_val,
                            sampler=val_sampler,
                            batch_size=cfg.val.batch_size_per_gpu,
                            collate_fn=collate_fn_val,
                            pin_memory=True,
                            num_workers=cfg.val.nof_workers_per_gpu)
    else:
        val_dl = DataLoader(dataset_val,
                            batch_size=cfg.val.batch_size_per_gpu,
                            collate_fn=collate_fn_val,
                            pin_memory=True,
                            num_workers=cfg.val.nof_workers_per_gpu)

    return val_dl
