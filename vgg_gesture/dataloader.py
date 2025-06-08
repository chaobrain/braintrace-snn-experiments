import platform

import tonic
import torch
from torchvision import transforms


def dataloader(args):
    data_path = args.data_path
    if platform.system() == 'Windows':
        workers = 0
    else:
        workers = 4
    train_loader, val_loader, trainset_len, testset_len = (
        dataloader_gesture(args.batch_size, args.val_batch_size, workers, data_path)
    )
    args.full_train_len = trainset_len
    args.full_test_len = testset_len
    args.n_classes = 11
    args.n_steps = 20
    args.n_inputs = 2
    args.dt = 75e-3
    args.classif = True
    args.delay_targets = 5
    args.skip_test = False
    return train_loader, val_loader


def dataloader_gesture(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", reproducibility=False):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    trainset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=True)
    testset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=False)

    slicing_time_window = 1575000
    slicer = tonic.slicers.SliceByTime(time_window=slicing_time_window)

    frame_transform = tonic.transforms.Compose(
        [  # tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=75000),
            torch.tensor, transforms.Resize(32)
        ]
    )
    frame_transform_test = tonic.transforms.Compose(
        [  # tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size,
                                     time_window=75000),
            torch.tensor,
            transforms.Resize(32, antialias=True)
        ]
    )

    trainset_ori_sl = tonic.SlicedDataset(
        trainset_ori,
        slicer=slicer,
        metadata_path=data_path + '/metadata/online_dvsg_train',
        transform=frame_transform
    )

    print(f"Went from {len(trainset_ori)} samples in the original "
          f"dataset to {len(trainset_ori_sl)} in the sliced version.")
    print(f"Went from {len(testset_ori)} samples in the original "
          f"dataset to {len(testset_ori)} in the sliced version.")

    frame_transform2 = tonic.transforms.Compose(
        [  # tonic.transforms.DropEvent(p=0.1),
            torch.tensor,
            transforms.RandomCrop(32, padding=4)
        ]
    )

    trainset = tonic.DiskCachedDataset(
        trainset_ori_sl,
        cache_path=data_path + '/cache/online_fast_dataloading_train',
        transform=frame_transform2
    )
    testset = tonic.DiskCachedDataset(
        testset_ori,
        cache_path=data_path + '/cache/online_fast_dataloading_test',
        transform=frame_transform_test
    )

    if reproducibility:
        import numpy as np
        import random
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True),
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True),
            worker_init_fn=seed_worker,
            generator=g,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True)
        )
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True)
        )

    return train_loader, val_loader, len(trainset_ori_sl), len(testset_ori)


if __name__ == '__main__':
    class A:
        batch_size = 16
        val_batch_size = 16


    args = A()
    args.data_path = '../data'
    train_loader, val_loader = dataloader(args)
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
