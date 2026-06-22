import os

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora


def setup_distributed(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.distributed = args.world_size > 1
    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        torch.cuda.set_device(0)


def main():
    args = get_arguments()
    setup_distributed(args)
    set_random_seed(args.seed)

    try:
        if args.batch_size < 1:
            raise ValueError('batch_size must be at least 1')
        if args.accumulation_steps < 1:
            raise ValueError('accumulation_steps must be at least 1')
        batch_divisor = args.world_size * args.accumulation_steps
        if args.batch_size % batch_divisor != 0:
            raise ValueError(
                f'Effective global batch size {args.batch_size} must be '
                f'divisible by world size × accumulation steps '
                f'({batch_divisor})')
        per_gpu_microbatch_size = args.batch_size // batch_divisor

        clip_model, preprocess = clip.load(args.backbone)
        clip_model.eval()
        logit_scale = 100

        if args.rank == 0:
            print(
                f"Distributed training: {args.world_size} GPU(s); "
                f"effective global batch size: {args.batch_size}; "
                f"per-GPU microbatch size: {per_gpu_microbatch_size}; "
                f"accumulation steps: {args.accumulation_steps}; "
                f"batch formula: {per_gpu_microbatch_size} × "
                f"{args.world_size} × {args.accumulation_steps} = "
                f"{args.batch_size}")
            print("Preparing dataset.")

        dataset = build_dataset(
            args.dataset, args.root_path, args.shots, preprocess,
            setting=args.setting)

        if args.rank == 0 and args.setting == 'base2new':
            print(
                f'Base-to-novel split: {len(dataset.classnames)} base classes, '
                f'{len(dataset.test_new_classnames)} novel classes.')

        if args.dataset == 'imagenet':
            val_loader = torch.utils.data.DataLoader(
                dataset.val, batch_size=args.eval_batch_size, num_workers=8,
                shuffle=False, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                dataset.test, batch_size=args.eval_batch_size, num_workers=8,
                shuffle=False, pin_memory=True)
            test_new_loader = (
                torch.utils.data.DataLoader(
                    dataset.test_new, batch_size=args.eval_batch_size,
                    num_workers=8, shuffle=False, pin_memory=True)
                if dataset.test_new is not None else None)
        else:
            val_loader = build_data_loader(
                data_source=dataset.val, batch_size=args.eval_batch_size,
                is_train=False, tfm=preprocess, shuffle=False, num_workers=8)
            test_loader = build_data_loader(
                data_source=dataset.test, batch_size=args.eval_batch_size,
                is_train=False, tfm=preprocess, shuffle=False, num_workers=8)
            test_new_loader = (
                build_data_loader(
                    data_source=dataset.test_new,
                    batch_size=args.eval_batch_size, is_train=False,
                    tfm=preprocess, shuffle=False, num_workers=8)
                if dataset.test_new is not None else None)

        if test_new_loader is not None:
            test_loader = (test_loader, test_new_loader)

        train_loader = None
        if not args.eval_only:
            if args.log_interval < 1:
                raise ValueError('log_interval must be at least 1')
            if args.val_interval < 0:
                raise ValueError('val_interval cannot be negative')
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=224, scale=(0.08, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711))
            ])
            train_sampler = None
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset.train_x, num_replicas=args.world_size,
                    rank=args.rank, shuffle=True, seed=args.seed)

            if args.dataset == 'imagenet':
                train_loader = torch.utils.data.DataLoader(
                    dataset.train_x, batch_size=per_gpu_microbatch_size,
                    num_workers=8, shuffle=train_sampler is None,
                    sampler=train_sampler, pin_memory=True)
            else:
                train_loader = build_data_loader(
                    data_source=dataset.train_x,
                    batch_size=per_gpu_microbatch_size,
                    tfm=train_transform, is_train=True,
                    shuffle=train_sampler is None, sampler=train_sampler,
                    num_workers=8)

        run_lora(
            args, clip_model, logit_scale, dataset,
            train_loader, val_loader, test_loader)
    finally:
        if args.distributed:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
