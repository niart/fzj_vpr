#!/bin/python
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "../utils")
sys.path.insert(1, "../")
from train_hybrid_vae_guided_base import Guide, HybridGuidedVAETrainer
import matplotlib

matplotlib.use("Agg")
from hybrid_beta_vae import Reshape, VQVAE
from decolle.utils import (
    parse_args,
    train,
    test,
    accuracy,
    save_checkpoint,
    load_model_from_checkpoint,
    prepare_experiment,
    write_stats,
    cross_entropy_one_hot,
    MultiOpt,
)

import datetime, os, socket
import numpy as np
import torch
from torch import nn
from itertools import chain
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchneuromorphic import transforms
from tqdm import tqdm
from utils import generate_process_target
import argparse
from torch.nn.parallel import DataParallel

epsilon = sys.float_info.epsilon
np.set_printoptions(precision=4)


if __name__ == "__main__":
    # parse args for params file, and dataset_path, that should take care of pretty much everything I think...
    # I might need to add something for lights, but I'll figure it out

    parser = argparse.ArgumentParser("HGVAE")

    parser.add_argument(
        "--params-file",
        default="/root/sabrina/fzj_vpr/train/train_params.yml",
        type=str,
        help="Path to the parameter config file.",
    )
    parser.add_argument(
        "--data-file",
        default="/root/autodl-tmp/dataset/chop_still_50_firstcamera_0",
        type=str,
        help="Path to the file the data is in, should be hdf5 compatible with torchneuromorphic.",
    )
    parser.add_argument(
        "--data-file-test",
        default="/root/autodl-tmp/dataset/chop_still_50_firstcamera_1",
        type=str,
        help="Path to the file the data is in, should be hdf5 compatible with torchneuromorphic.",
    )
    parser.add_argument("--ds", default=4, type=int, help="input downsample factor.")
    args = parser.parse_args()

    param_file = (
        args.params_file
    )  # Alternatively, directly put the path to the config file, for example: 'parameters/params_hybridvae_dvsgestures-guidedbeta-noaug-Copy1.yml'
    dataset_path_train = (
        args.data_file
    )  # Alternatively, directly put the path to the config file, for example: '/home/kennetms/Documents/data/dvs_gestures.hdf5'
    dataset_path_test = args.data_file_test
    ds = args.ds

    HGVAE = HybridGuidedVAETrainer(
        param_file, dataset_path_train, dataset_path_test, use_other=False, ds=ds
    )

    print(HGVAE.net)
    # HGVAE.net = DataParallel(HGVAE.net)
    HGVAE.train_eval_plot_loop()
