import os
import torch
import random
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument(
            '--dataset', type=str, help="The name of the dataset")
    parser.add_argument(
            '--pretrained', type=str, default=None,
            help="Where the pretrained weight is")
    parser.add_argument(
            '--checkpoint', type=str, help="Where the checkpoint is stored")
    parser.add_argument(
            '--epochs', type=int, default=100)
    parser.add_argument(
            '--batch-size', type=int, default=512)
    parser.add_argument(
            '--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
            '--num-workers', type=int, default=16)

    # Partial function
    parser.add_argument(
            '--finetune', action='store_true',
            help="Finetune the last layer or train the whole model")
    parser.add_argument(
            '--test', action='store_true', help="Only test on eval dataset")

    # Reproducibility
    parser.add_argument(
            '--deterministic', action='store_true',
            help="Make the result reproducible")
    parser.add_argument(
            '--seed', type=int, help="Fix random seed for reproducibility")

    # Output results for CAM, T-SNE etc.
    parser.add_argument(
            '--result', type=str, help="The path where the result will be.")
    args = parser.parse_args()

    if args.deterministic or args.seed:
        set_deterministic(args.seed)

    if args.checkpoint is not None:
        _check_dir(args.checkpoint)

    if args.result is not None:
        _check_dir(args.result)

    return args


def _check_dir(ckpt_dir):
    ckpt_dir = os.path.dirname(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
        print("[ INFO ] directory {ckpt_dir} not exists, created.")

    return


def set_deterministic(seed):
    print(f"[ INFO ] Set deterministic, seed is {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backend.cudnn.benchmark = False

    return
