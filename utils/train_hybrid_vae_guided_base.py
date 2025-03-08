# I changed the pipeline of loading dataset in this file.
#!/bin/python
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
import sys
import os
import csv
import pickle

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "../utils")
import matplotlib

matplotlib.use("Agg")
from hybrid_beta_vae import Reshape, VQVAE

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory and append it to sys.path
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
import glob
from torch.utils.data import DataLoader

# Now you can import the function from the decolle package
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
)

import datetime, socket
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import importlib
from itertools import chain
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from decolle.utils import MultiOpt

from tqdm import tqdm
import math
from utils import generate_process_target

from torchvision.transforms import *

# from torchneuromorphic.transforms import *

import pdb

from scipy.ndimage import zoom

epsilon = sys.float_info.epsilon
np.set_printoptions(precision=4)


class Guide(nn.Module):
    """
    A torch neural network classifier that takes a partial latent space
    as input and outputs a class.
    An "excititory" guide takes a latent space of length num_classes
    the "inhibitory" guide takes the remaining latent space
    and these are trained jointly with the VAE encoder.
    Inspired by the work in https://arxiv.org/abs/2004.01255
    """

    def __init__(self, dimz, num_classes, excite, hidden_layers):
        """
        inputs:
            - int dimz: The dimension of the latent space
            - int num_classes: How many output classes
            - bool excite: If true, use excititory portion of latent space (num_classes)
                           If false use inhibitory portion of the latent space (remainder)
            - OrderedDict hidden_layers: The layers of the network to use
        """

        super(Guide, self).__init__()
        if excite:
            input_size = num_classes
        else:
            input_size = dimz - num_classes

        output_size = num_classes

        self.num_classes = num_classes

        self.model = nn.Sequential(hidden_layers)

        # init model weights
        for l in self.model:
            if isinstance(l, nn.Linear):
                torch.nn.init.kaiming_uniform_(l.weight, nonlinearity="leaky_relu")

    def forward(self, x):
        # i = 0
        # print("forward")
        # print("input shape",x.shape)
        for l in self.model:
            x = l(x)
            # print("next shape",x.shape)
            # if isinstance(l, nn.Linear):
            #     if i == 0:
            #         i+=1
            #     else:
            #         soft_inp = x[0]
            #         with torch.no_grad():
            #             soft_inp_mean = torch.mean(soft_inp)
        return x  # , soft_inp, soft_inp_mean

    def excite_z(self, z):
        """
        Obtains and outputs the excititory portion of the latent space
        """
        exc_z = torch.zeros((z.shape[0], self.num_classes))

        for i in range(z.shape[0]):
            exc_z[i] = z[i, : self.num_classes]  # [t[i]]

        return exc_z

    def inhibit_z(self, z):
        """
        Obtains and outputs the inhibitory portion of the latent space
        """
        inhib_z = torch.zeros((z.shape[0], z.shape[1] - self.num_classes))

        for i in range(z.shape[0]):
            inhib_z[i] = z[i, self.num_classes :]
        return inhib_z


class GestureDataset:
    def __init__(
        self,
        data,
        train=True,
        transform=None,
        target_transform=None,
        download_and_create=True,
        chunk_size=50,
        return_meta=False,
        time_shuffle=False,
    ):

        self.download_and_create = download_and_create
        self.data = data
        self.train = train
        self.chunk_size = chunk_size
        self.return_meta = return_meta
        self.time_shuffle = time_shuffle
        self.transform = transform

    def __len__(self):
        return len(self.data)
        # return 1000

    def one_hot_encode(self, class_index):
        if class_index < 0 or class_index > 15:
            raise ValueError("Class index must be between 0 and 15")
        # 创建一个长度为16的全0列表
        encoding = [0] * 16
        # 将给定类的索引位置设置为1
        encoding[class_index] = 1
        return encoding

    def __getitem__(self, key):
        data_path = self.data[key]
        data_dict = np.load(data_path, allow_pickle=True).item()

        # 提取data和label
        data = data_dict["data"]
        label = data_dict["label"]
        one_hot_label = self.one_hot_encode(int(label))

        # Try using one_hot from torch??
        # label = torch.tensor(int(data_dict["label"]), dtype=torch.long)
        # one_hot_label = torch.nn.functional.one_hot(label, num_classes=16).float()

        # data = data.astype(np.float32)
        # print(data.shape)

        # if self.transform is not None:
        # data = self.transform(data)
        # Parameters
        time_steps = 50
        num_classes = 16
        target = [one_hot_label[:] for _ in range(time_steps)]
        return (
            data,
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(0, dtype=torch.float32),
            torch.tensor(0, dtype=torch.float32),
        )

    #     # Try using one_hot from torch??
    #     target = one_hot_label.unsqueeze(0).repeat(time_steps, 1)  # Ensure proper shape
    #     return (
    #     torch.tensor(data, dtype=torch.float32),
    #     target,  # Now correctly formatted as a tensor
    #     torch.tensor([0.0], dtype=torch.float32),  # Use a list to avoid scalar conversion issues
    #     torch.tensor([0.0], dtype=torch.float32),  # Use a list to avoid scalar conversion issues
    # )


class HybridGuidedVAETrainer:
    def __init__(
        self, param_file, dataset_path, dataset_path_test, use_other=False, ds=4
    ):
        """
        Initializes all variables related to loading the data
        and training the hg vae model

        inputs:
            - string param_file: The path to the .yml parameter file used to get network parameters
            - string dataset_path: The path to the .hdf5 file containing the dataset data
            - bool use_other: For DVSGesture, indicates if the other class should be used in training
        """

        self.use_other = use_other

        self.args = parse_args(
            param_file
        )  #'parameters/params_hybridvae_dvsgestures-guidedbeta-noaug.yml')

        self.params, self.writer, dirs = prepare_experiment(
            name=__file__.split("/")[-1].split(".")[0], args=self.args
        )
        self.log_dir = dirs["log_dir"]
        self.checkpoint_dir = '/root/autodl-tmp/default' #dirs["checkpoint_dir"]

        self.starting_epoch = self.params["start_epoch"]

        self.args.resume_from = self.params["resume_from"]  #
        if self.args.resume_from != "None":
            self.checkpoint_dir = self.args.resume_from
        verbose = self.args.verbose

        # self.device = self.params["device"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        ## Load Data

        self.filter_data, self.process_target = generate_process_target(self.params)

        # self.generate_data_batch_from_aedat4(dataset_path, dataset_path_test, ds)
        self.generate_data_batches(dataset_path, dataset_path_test, ds)

        # d, t = next(iter(train_dl))
        self.input_shape = self.data_batch.shape[-3:]

        # Backward compatibility
        if "dropout" not in self.params.keys():
            self.params["dropout"] = [0.5]

        ## Create Model, Optimizer and Loss
        if self.params["out_features"] is not None:
            out_features = self.params["out_features"]
        else:
            out_features = 128

        if self.params["ngf"] is not None:
            ngf = self.params["ngf"]
        else:
            ngf = 16

        self.net = VQVAE(
            input_shape=self.params["input_shape"],
            ngf=ngf,
            out_features=out_features,
            seq_len=self.params["chunk_size_train"],
            dimz=self.params["dimz"],
            encoder_params=self.params,
        ).to(self.device)

        from decolle.init_functions import init_LSUV

        init_LSUV(self.net.encoder, self.data_batch.to(self.device))

        if self.params["is_guided"]:
            from collections import OrderedDict

            inhib_layers = OrderedDict([])

            for i, size in enumerate(self.params["inhib_layers"][:-1]):
                if i == 0:
                    inhib_layers[f"lin{i}"] = nn.Linear(
                        self.params["dimz"] - self.params["num_classes"], size
                    )
                else:
                    inhib_layers[f"lin{i}"] = nn.Linear(
                        self.params["inhib_layers"][i - 1], size
                    )
                inhib_layers[f"norm{i}"] = nn.BatchNorm1d(size)
                inhib_layers[f"relu{i}"] = nn.LeakyReLU(
                    negative_slope=0.2, inplace=True
                )

            inhib_layers[f"lin{i+1}"] = nn.Linear(
                self.params["inhib_layers"][-1], self.params["num_classes"]
            )

            self.inhib = Guide(
                self.params["dimz"], self.params["num_classes"], False, inhib_layers
            ).to(self.device)

            # The loss for the guided part of the VAE
            self.inhib_criterion = nn.MultiLabelSoftMarginLoss(
                reduction="sum"
            )  # nn.CrossEntropyLoss() #nn.NLLLoss()

            self.opt_excititory = torch.optim.Adam(
                self.net.cls_sq.parameters(), lr=self.params["learning_rate"][2]
            )  # torch.optim.Adam(chain(*[net.encoder.get_trainable_parameters(),excite.model.parameters()]), lr=params['learning_rate'][2])
            self.opt_inhibitory = torch.optim.Adam(
                self.inhib.model.parameters(), lr=self.params["learning_rate"][3]
            )  # torch.optim.Adam(chain(*[net.encoder.get_trainable_parameters(),inhib.model.parameters()]), lr=params['learning_rate'][3])

        # DECOLLE needs different learning rates
        opt1 = torch.optim.Adamax(
            self.net.encoder.get_trainable_parameters(),
            lr=self.params["learning_rate"][0],
            betas=self.params["betas"],
            eps=1e-4,
        )
        opt2 = torch.optim.Adam(
            # chain(*[self.net.encoder_head.parameters(), self.net.decoder.parameters()]),
            chain(*[self.net.bottleneck.parameters(), self.net.vq_layer.parameters(), self.net.decoder.parameters()]),
            lr=self.params["learning_rate"][1],
        )
        if self.params["is_guided"]:
            self.opt = MultiOpt(opt1, opt2, self.opt_excititory, self.opt_inhibitory)
        else:
            self.opt = MultiOpt(opt1, opt2)
        # opt = torch.optim.Adamax(net.parameters(), lr = params['learning_rate'], eps=1e-6)

        ##Resume if necessary
        if self.args.resume_from != "None":
            print("Checkpoint directory " + self.checkpoint_dir)
            if not os.path.exists(self.checkpoint_dir) and not self.args.no_save:
                os.makedirs(self.checkpoint_dir)
            self.starting_epoch = load_model_from_checkpoint(
                self.checkpoint_dir,
                self.net,
                self.opt,
                excite=self.net.cls_sq,
                inhib=self.inhib,
                n_checkpoint=-1,
            )
            print(
                "Learning rate = {}. Resumed from checkpoint".format(
                    self.opt.param_groups[-1]["lr"]
                )
            )

            orig = (
                self.process_target(self.data_batch)
                .detach()
                .cpu()
                .view(*[[-1] + self.params["output_shape"]])[:, 0:1]
            )
            # print(orig.shape)
            figure2 = plt.figure(99, figsize=(60, 60))
            num_images = orig.shape[0]

            for i in range(num_images):
                # Rotate each image by 90 degrees clockwise
                rotated_image = orig[i, 0, :, :]  # np.rot90(orig[i, 0, :, :], k=-1)

                # Create subplots for each image
                plt.subplot(1, num_images, i + 1)
                plt.imshow(
                    rotated_image,
                    cmap="rainbow",
                    vmin=rotated_image.min(),
                    vmax=rotated_image.max() * 0.5,
                )

            # plt.imshow(make_grid(orig, scale_each=True, normalize=True).transpose(0,2).numpy())
            if not self.args.no_save:
                self.writer.add_figure("original_train", figure2, global_step=1)

        # Printing parameters
        if self.args.verbose:
            print("Using the following parameters:")
            m = max(len(x) for x in self.params)
            for k, v in zip(self.params.keys(), self.params.values()):
                print("{}{} : {}".format(k, " " * (m - len(k)), v))

        print("\n------Starting training Hybrid VAE-------")
        # --------TRAINING LOOP----------
        self.num_classes = self.params["num_classes"]

    def load_data(self, root, batch_size, chunk_size):
        """Creates a DataLoader for train or test data."""
        file_paths = glob.glob(os.path.join(root, "*.npy"))
        sample_shuffle = self.params["sample_shuffle"]
        num_workers = self.params["num_dl_workers"]
        dataset = GestureDataset(file_paths, chunk_size=chunk_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sample_shuffle,
            num_workers=num_workers,
        )

    def generate_data_batches(self, dataset_path, dataset_path_test, ds=4):
        """Initializes train and test dataloaders."""
        self.train_dl = self.load_data(
            dataset_path, self.params["batch_size"], self.params["chunk_size_train"]
        )
        self.test_dl = self.load_data(
            dataset_path_test, self.params["batch_size"], self.params["chunk_size_test"]
        )

        self.data_batch, self.target_batch, _, _ = next(iter(self.train_dl))
        self.data_batch = self.data_batch.to(self.device)
        self.target_batch = self.target_batch.to(self.device)
        print(f"Data batch shape: {self.data_batch.shape}")

    def loss_fn(self, recon_x, x, quantized, z_e, vae_beta=1):
        """
        VAE loss function using KL Divergence

        inputs:
            - torch recon_x: reconstruction of data from latent space
            - torch x: the original data
            - float mu: the mean
            - float logvar: the variance
            - float vae_beta: beta parameter that is a coefficient of the kld loss

        outputs:
            - float: The total loss between the reconstruction loss and kld loss
        """
        # pdb.set_trace()
        # Reconstruction Loss
        llhood = torch.nn.functional.mse_loss(recon_x, x)
        # # negKLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        # ) / len(self.train_dl)
        # # print(llhood,kld_loss)

        # Compute loss for VQ (Commitment + Codebook)
        commitment_loss = F.mse_loss(quantized.detach(), z_e)
        codebook_loss = F.mse_loss(quantized, z_e.detach())
        vq_loss = 0.5 * commitment_loss + codebook_loss

        return llhood + vae_beta * vq_loss

    # def train_step(self, s, x):
    #     """
    #     Normal VAE training step without the guided part

    #     inputs:
    #         - torch s: raw event data
    #         - torch x: time surface of event data

    #     outputs:
    #         - float loss.data: the loss
    #     """
    #     self.net.train()
    #     y, mu, logvar = self.net(s)
    #     loss = self.loss_fn(y, x, mu, logvar, self.params["vae_beta"])
    #     loss.backward()
    #     self.opt.step()
    #     self.opt.zero_grad()
    #     return loss.data

    def print_grads(self):
        """
        prints out the gradients of the networks
        to help with debugging the networks
        and training them better
        """

        print("excite")
        for l in self.net.cls_sq.model:
            if isinstance(l, nn.Linear) and l.weight.grad is not None:
                print(l.weight.grad.abs().mean())
            else:
                print(None)
        print("inhib")
        for l in self.inhib.model:
            if isinstance(l, nn.Linear) and l.weight.grad is not None:
                print(l.weight.grad.abs().mean())
            else:
                print(None)
        print("encoder head")
        for k, v in self.net.encoder_head.items():
            if isinstance(v, nn.Linear):
                print(v.weight.grad.abs().mean())

    def calc_entropy(self, soft_inp, bins):
        soft_inp = soft_inp.detach().cpu().numpy()

        soft_hist, bin_edges = np.histogram(soft_inp, bins=bins, density=False)

        # print("histogram",soft_hist)

        entropy = 0
        for i in range(len(soft_hist)):
            entropy -= (soft_hist[i] / bins) * math.log(
                soft_hist[i] / bins + epsilon, 2
            )

        return entropy

    def batch_one_hot(self, targets):
        """
        converts target vector into a one hot vector
        """
        one_hot = torch.zeros((targets.shape[0], self.num_classes))

        for i in range(targets.shape[0]):
            one_hot[i][targets[i]] = 1

        return one_hot

    def train_step_guided(self, s, x, t, vae_beta=1):  # t is the target class
        """
        Guided VAE training step inspired by https://arxiv.org/abs/2004.01255

        inputs:
            - torch s: raw event data
            - torch x: time surface of event data
            - torch t: target classes for each datapoint
            - float vae_beta: beta term from beta vae literature that helps with the loss

        outputs:
            - float loss.data: the loss of the vae model
            - float excite.data: the loss of the excititory guide classifier
            - float inhib.data: the loss of the inhibitory guide classifier
            - float clas_loss.data: classifier loss

        """
        # Need to modify it so that it is indeed true that one neuron corresponds to one class
        # and that ideally I know which neuron corresponds to which class
        # I think I can do this by disabling gradients for the not learning neurons, like what I did with SOEL
        # but I'm not positive how to do this in torch, need to think about this
        # could also be a matter of not propogating forward as well

        self.net.train()
        # excite.train()
        self.inhib.train()

        # VAE, and the real excitation net?
        # should be going through both the vae and real excitation net in this case together
        self.opt.zero_grad()
        # opt_excite.zero_grad()
        hot_ts = self.batch_one_hot(t)
        y, quantized, z_e, clas = self.net(s)
        loss = self.loss_fn(y, x, quantized, z_e, vae_beta)

        clas_loss = (
            self.inhib_criterion(clas, hot_ts.to(self.device))
            * self.params["class_weight"]
        )
        vae_loss = loss
        loss += clas_loss
        loss.backward()
        self.opt.step()
        # opt_excite.step()

        # excitation net??? I'm pretty confused as to what this block is really doing.
        # it's feeding in the inhibition z's to the inhibition net and going backwards on it with the loss
        # and actual targets.
        self.opt.zero_grad()
        # opt_inhib.zero_grad()

        # z = self.net.reparameterize(mu, logvar).detach()
        # !!!!!!!!!!!!!!!!
        quantized = self.net.vq_layer(z_e) 
        z = quantized
        
        inhib_z = self.inhib.inhibit_z(z).to(self.device)

        excite_output = self.inhib(inhib_z)

        loss = (
            self.inhib_criterion(excite_output, hot_ts.to(self.device))
            * self.params["class_weight"]
        )  # now MultiLabelSoftMarginLoss
        excite_loss = loss
        loss.backward()
        self.opt.step()
        # opt_inhib.step()

        # excitation net??? no it's not, it's the inhibition again, this time it's being trained adversarially with itself I guess
        # or something. I need to really go over this again I think. It seems to work, but I'm having trouble understanding
        # exactly why or if there's stuff that could be changed to make it better.
        self.opt.zero_grad()
        self.opt_excititory.zero_grad()  # separate but same optimizer that is in the multiopt??? Doe this do anything???
        # mu, logvar = self.net.encode(s)
        # z = self.net.reparameterize(mu, logvar)
        # !!!!!!!!!!!!!!!!
        quantized = self.net.vq_layer(z_e) 
        z = quantized
        
        inhib_z = self.inhib.inhibit_z(z)
        # print(inhib_z.shape)
        inhib_output = self.inhib.model(inhib_z.to(self.device))
        soft_entropy = 0  # calc_entropy(soft_inp,bins=params['num_classes'])

        # inhib net. adversarial train to encode for irrelevant features to classes
        inhib_hot_ts = torch.empty_like(hot_ts).fill_(0.5)
        loss = (
            self.inhib_criterion(inhib_output, inhib_hot_ts.to(self.device))
            * self.params["class_weight"]
        )  # now MultiLabelSoftMarginLoss
        inhib_loss = loss
        loss.backward()
        self.opt.step()
        self.opt_excititory.step()  # need to check the impact doing this has, if any

        return (
            vae_loss.data,
            excite_loss.data,
            inhib_loss.data,
            clas_loss.data,
            soft_entropy,
            0,
        )  # soft_mean=0

    def train(self):
        """
        The training loop for training a normal VAE model.
        Iterates through data batches loaded from the dataloader
        and inputs them into self.train_step() for learning

        outputs:
            - float loss_batch: the loss of the vae model
        """

        self.net.train()
        loss_batch = []
        for x, t in tqdm(iter(self.train_dl)):
            x_c = x.to(self.device)
            frames = self.process_target(x_c)
            loss_ = self.train_step(x_c, frames.to(self.device))
            loss_batch.append(loss_.detach().cpu().numpy())
        return np.mean(loss_batch)

    def train_guided_aug(self):
        """
        The training loop for training the hybrid guided VAE model.
        Iterates through data batches loaded from the dataloader
        and inputs them into self.train_step_guided() for learning

        outputs:
            - float loss_batch: the loss of the vae model
            - float excite_batch: the loss of the excititory guide classifier
            - float inhib_batch: the loss of the inhibitory guide classifier
            - float abs_batch: absolute difference of the excititory and inhibitory losses
            - float entropy_batch:
            - float mean_batch: mean of the losses
        """

        self.net.train()
        loss_batch = []
        excite_batch = []
        inhib_batch = []
        abs_batch = []
        entropy_batch = []
        mean_batch = []
        for i in range(self.params["num_augs"]):
            for x, t, l, u in tqdm(iter(self.train_dl)):
                # if not self.use_other:
                #     new_t = t[t[:,-1,:].argmax(1)!=10]
                # else:
                #     new_t = t
                new_t = t
                new_t = new_t[:, -1, :].argmax(1)

                if new_t.shape[0] <= 1:
                    continue

                # if not self.use_other:
                #     x = x[t[:,-1,:].argmax(1)!=10]
                x_c = x.to(self.device)
                frames = self.process_target(x_c, i - 1)
                loss_, excite_loss_, inhib_loss_, loss_abs_, soft_entropy, soft_mean = (
                    self.train_step_guided(
                        x_c,
                        frames.to(self.device),
                        new_t.long(),
                        self.params["vae_beta"],
                    )
                )
                loss_batch.append(loss_.detach().cpu().numpy())
                excite_batch.append(excite_loss_.detach().cpu().numpy())
                inhib_batch.append(inhib_loss_.detach().cpu().numpy())
                abs_batch.append(loss_abs_.detach().cpu().numpy())
                entropy_batch.append(soft_entropy)
                mean_batch.append(soft_mean)  # .detach().cpu().numpy())
        return (
            np.mean(loss_batch, dtype=np.float64),
            np.mean(excite_batch, dtype=np.float64),
            np.mean(inhib_batch, dtype=np.float64),
            np.mean(abs_batch, dtype=np.float64),
            np.mean(entropy_batch, dtype=np.float64),
            np.mean(mean_batch, dtype=np.float64),
        )

    def get_latent_space(self, dl, iterations=1):
        """
        Inputs data generated from the dataloader
        into the VAE encoder to output the corresponding latent space
        and returns the latent space with the corresponding targets.

        inputs:
            - one of self.train_dl or self.test_dl
            - int iterations: how many times to get the latent space for plotting, default 1

        outputs:
            - ndarray lats: The latent space for each datapoint in the dataset
            - ndarray tgts: The targets corresponding to each datapoint in the dataset
            - ndarray usrs: The users that performed the gesture of each datapoint
            - ndarray lights: The lighting condition under which each gesture was performed
        """

        # all_d = []
        lats = []
        tgts = []

        for i in range(iterations):
            for x, t, l, u in tqdm(iter(dl)):  # x, t in tqdm(iter(dl)):
                # if not self.use_other:
                #     new_t = t[t[:,-1,:].argmax(1)!=10]
                # else:
                #     new_t = t
                new_t = t

                new_t = new_t[:, -1, :].argmax(1)

                # print(new_t.shape[0])
                if new_t.shape[0] < 1:
                    continue

                # if not self.use_other:
                #     x = x[t[:,-1,:].argmax(1)!=10]
                #     l = np.zeros(x.shape) #np.asarray(l)[t[:,-1,:].argmax(1)!=10]
                #     u = np.zeros(x.shape)#np.asarray(u)[t[:,-1,:].argmax(1)!=10]
                with torch.no_grad():
                    # mu, logvar = self.net.encode(x.to(self.device))
                    # lat = self.net.reparameterize(mu, logvar).detach().cpu().numpy()
                    quantized, _ = self.net.encode(x.to(self.device))  # Get quantized latent space
                    lat = quantized.detach().cpu().numpy()
                    # Check for NaNs and replace them with 0
                    if np.isnan(lat).any() or np.isinf(lat).any():
                        print("⚠️ Warning: NaN or Inf detected in latent space. Replacing with zeros.")
                        lat = np.nan_to_num(lat)  # Replace NaNs and Infs with 0

                    lats += lat.tolist()
                    tgts += new_t.tolist()
                    # all_d += process_target(x).tolist()
        return np.array(lats), np.array(tgts)  # [:,-1,:].argmax(1)

    def tsne_project(self, lats, tgts):
        """
        Calculates tsne projections of the latent space
        and color codes them according to the provided targets
        for visualization of how close or far data points are in the reduced tsne projection
        to get a sense of how well the model is disentangling the data in the latent space

        inputs:
            - ndarray lats: The latent space for each datapoint in the dataset
            - ndarray tgts: The targets corresponding to each datapoint in the dataset
            - bool do_plot: If true, plots the tsnes and returns them as figures
                            If False just returns the tsne projection
            - bool use_user: whether or not to do tsne of user data
            - bool use_light: whether or not to do tsne of lighting conditions
        outputs:
            - ndarray lat_tsne: The tsne projection of the latent space
            - pyplot figure fig: The plot of the tsne projection of the latent space
            - pyplot figure fig2: plot of user tsne
            - pyplot figure fig3: plot of lighting condition tsne
            - pyplot figure fig4: The plot of the tsne projection of the excititory portion of the latent space
            - pyplot figure fig5: The plot of the tsne projection of the inhibitory portion of the latent space
        """
        from sklearn.manifold import TSNE

        colors = [
            "#eced0f",
            "#bcfb1e",
            "#7afb1e",
            "#3ced1e",
            "#f8b51e",
            "#d5ce5f",
            "#82ce7a",
            "#32b569",
            "#f8611e",
            "#d06b7a",
            "#756bc6",
            "#2561b5",
            "#ea191e",
            "#b01969",
            "#5e19b5",
            "#1b19d4",
        ]

        # if np.isnan(lats).any() or np.isinf(lats).any():
        #     print("⚠️ Warning: NaN/Inf detected in latent space for t-SNE. Replacing with zeros.")
        #     lats = np.nan_to_num(lats)

        # lat_tsne = TSNE(n_components=2).fit_transform(lats)

        # inhib_tsne = TSNE(n_components=2).fit_transform(
        #     self.inhib.inhibit_z(torch.from_numpy(lats)).numpy()
        # )
        # exc_tsne = TSNE(n_components=2).fit_transform(
        #     self.inhib.excite_z(torch.from_numpy(lats)).numpy()
        # )
        
        if np.isnan(lats).any() or np.isinf(lats).any():
            print("⚠️ Warning: NaN/Inf detected in latent space for t-SNE. Replacing with zeros.")
            lats = np.nan_to_num(lats)

        # Check if variance is zero before running t-SNE
        if np.std(lats[:, 0]) == 0:
            print("⚠️ Warning: Zero variance detected in t-SNE input. Skipping.")
            lat_tsne = np.zeros((lats.shape[0], 2))
        else:
            lat_tsne = TSNE(n_components=2).fit_transform(lats)

        # Convert lats to PyTorch tensor for inhibit_z and excite_z
        lats_tensor = torch.from_numpy(lats).float()

        inhib_latents = self.inhib.inhibit_z(lats_tensor).numpy()
        excite_latents = self.inhib.excite_z(lats_tensor).numpy()

        # Ensure no NaNs or zero variance in inhibitory/excitatory t-SNE
        if np.isnan(inhib_latents).any() or np.isinf(inhib_latents).any():
            print("⚠️ Warning: NaN/Inf detected in inhibitory latent space. Replacing with zeros.")
            inhib_latents = np.nan_to_num(inhib_latents)

        if np.isnan(excite_latents).any() or np.isinf(excite_latents).any():
            print("⚠️ Warning: NaN/Inf detected in excitatory latent space. Replacing with zeros.")
            excite_latents = np.nan_to_num(excite_latents)

        if np.std(inhib_latents[:, 0]) == 0:
            print("⚠️ Warning: Zero variance detected in inhibitory t-SNE input. Skipping.")
            inhib_tsne = np.zeros((inhib_latents.shape[0], 2))
        else:
            inhib_tsne = TSNE(n_components=2).fit_transform(inhib_latents)

        if np.std(excite_latents[:, 0]) == 0:
            print("⚠️ Warning: Zero variance detected in excitatory t-SNE input. Skipping.")
            exc_tsne = np.zeros((excite_latents.shape[0], 2))
        else:
            exc_tsne = TSNE(n_components=2).fit_transform(excite_latents)

        # plot
        fig = plt.figure(figsize=(16, 10))
        fig4 = plt.figure(figsize=(16, 10))
        fig5 = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot()
        ax4 = fig4.add_subplot()
        ax5 = fig5.add_subplot()
        for i in range(self.params["num_classes"]):  # 1):
            idx = tgts == i
            ax.scatter(lat_tsne[idx, 0], lat_tsne[idx, 1], c=colors[i], label=str(i))
            ax4.scatter(exc_tsne[idx, 0], exc_tsne[idx, 1], c=colors[i], label=str(i))
            ax5.scatter(
                inhib_tsne[idx, 0], inhib_tsne[idx, 1], c=colors[i], label=str(i)
            )
        ax.legend()
        ax4.legend()
        ax5.legend()

        return lat_tsne, fig, fig4, fig5

    def eval_accuracy(self, lats, tgts, is_excite=True):
        """
        Evaluates the accuracy of the guided classifier joint trained with the VAE
        Typically want the accuracy of the excititory classifier but
        can also do it with the inhibitory classifier

        inputs:
            - ndarray lats: the latent space
            - ndarray tgts: the target classes
            - bool is_excite: whether or not it is the excititory classifiers accuracy evaluated

        outputs:
            float: The accuracy of all correctly classified instances over all instances
        """

        correct_count, all_count = 0, 0
        if is_excite:
            zs = self.inhib.excite_z(torch.from_numpy(lats))
        else:
            zs = self.inhib.inhibit_z(torch.from_numpy(lats))
        self.net.cls_sq.eval()
        for i in range(len(tgts)):
            self.net.eval()
            with torch.no_grad():
                logps = self.net.cls_sq(
                    torch.unsqueeze(zs[i], 0).to(self.device)
                )  # was net.model(...)
            ps = torch.exp(logps.to(self.device))
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = tgts[i]  # .numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

        return correct_count / all_count

    def latent_traversal(self, lats, tgts, clas, n_plots=10):
        """
        Do a latent traversal of the latent space over a particular class
        to see if the latent variable trained to be guided on the class
        is correctly linked to the target class

        inputs:
            - ndarray lats:
            - ndarray tgts:
            - int clas: which class to do the traversal over
            - int n_plots: how times to plot images over the traversal

        outputs:
            - pyplot plot fig: The figure containing the plots of the reconstructed images
        """

        # do a latent traversal of a gesture (say right hand wave or something) and see if it produces waves
        # get first instance of a right hand wave latent space. Then change value of corresponding latent variable, should produce different waves
        num_classes = self.params[
            "num_classes"
        ]  # params is a global variable right? yes, yes it is

        # to determine disentanglement and ability to traverse latent space,
        # try to set the attribute variables to minimum
        # then traverse along one of them and see if it transitions to the attribute
        # that is desired
        lat = lats[tgts == clas][
            2
        ]  # baseline latent example of class we will traverse along

        # get min values of relevant latent dimensions

        # for i in range(num_classes):
        #    lat[i] = min(lats.T[i])

        # get min and max values of latent dimension 1
        min_lat = min(lats.T[clas])
        max_lat = max(lats.T[clas])

        trav_space = abs(min_lat) + max_lat

        fig, axs = plt.subplots(1, n_plots, figsize=(16, 10))

        for i in range(n_plots):
            lat[clas] = max_lat - (trav_space / n_plots) * i

            with torch.no_grad():
                # lat = torch.tensor(lat, dtype=torch.float).to(self.device)
                # lat = lat.clone().detach().to(self.device)
                lat = torch.tensor(lat, dtype=torch.float32).to(self.device)

                decoded = self.net.decode(lat).cpu()
                if i == 0:
                    first_decoded = decoded
                elif i == n_plots - 1:
                    last_decoded = decoded

            axs[i].imshow(decoded[0, 0].T)

        return fig

    def latent_traversal_switch(self, lats, tgts, clas1, clas2, n_plots=10):
        """
        Do a latent traversal of the latent space between two classes
        to see if the latent variable trained to be guided on the classes
        are disentangled

        inputs:
            - ndarray lats:
            - ndarray tgts:
            - int clas1: which class to start doing the traversal over
            - int clas2: which class should the traversal end up on
            - int n_plots: how times to plot images over the traversal

        outputs:
            - pyplot plot fig: The figure containing the plots of the reconstructed images
        """

        min_lat1 = min(lats.T[clas1])
        max_lat1 = max(lats.T[clas1])

        min_lat2 = min(lats.T[clas2])
        max_lat2 = max(lats.T[clas2])

        num_classes = self.params["num_classes"]

        lat = lats[tgts == clas1][0]

        # for i in range(num_classes):
        #    lat[i] = min(lats.T[i])

        lat[clas1] = min_lat1  # max_lat1
        lat[clas2] = max_lat2  # min_lat1

        trav_space1 = abs(min_lat1) + max_lat1
        trav_space2 = abs(min_lat2) + max_lat2

        fig, axs = plt.subplots(1, n_plots + 1, figsize=(16, 10))

        for i in range(n_plots + 1):
            if i > 0:
                lat[clas1] = lat[clas1] + (trav_space1 / n_plots)
                lat[clas2] = lat[clas2] - (trav_space2 / n_plots)

            with torch.no_grad():
                # lat = torch.tensor(lat, dtype=torch.float).to(self.device)
                # lat = torch.from_numpy(lat).float().to(self.device)
                if isinstance(lat, np.ndarray):
                    lat = torch.from_numpy(lat).float().to(self.device)
                else:
                    lat = lat.clone().detach().to(self.device)

                decoded = self.net.decode(lat).cpu()

            axs[i].imshow(decoded[0, 0].T)

        return fig

    def latent_traversal_inhib(self, lats, tgts, clas1, n_plots=10):
        """
        Do a latent traversal of the inhibitory latent space of a particular class
        to see if there are discernable other features that the remaining latent space
        is being trained on during disentanglement

        inputs:
            - ndarray lats:
            - ndarray tgts:
            - int clas1: which class to do the traversal over
            - int n_plots: how times to plot images over the traversal

        outputs:
            - pyplot plot fig: The figure containing the plots of the reconstructed images
        """

        min_lat1 = min(lats.T[clas1])
        max_lat1 = max(lats.T[clas1])

        min_lats = []  # list of min value of all latent variables
        max_lats = []  # list of max value of all latent variables

        num_classes = self.params["num_classes"]

        lat = lats[tgts == clas1][0]

        # minimize all latent dimension variables
        for i in range(self.params["dimz"]):
            # lat[i] = min(lats.T[i])

            if i >= num_classes:
                min_lats.append(min(lats.T[i]))
                max_lats.append(max(lats.T[i]))

        min_lats = torch.from_numpy(np.asarray(min_lats))
        max_lats = torch.from_numpy(np.asarray(max_lats))

        # maximize one of the target class latents such as right hand wave
        lat[clas1] = min_lat1  # max_lat1

        fig, axs = plt.subplots(1, n_plots + 1, figsize=(16, 10))

        for i in range(n_plots + 1):
            if i > 0:
                lat[
                    num_classes
                    * i : (
                        num_classes * i
                        + (self.params["dimz"] - num_classes) // num_classes
                    )
                ] = max_lats[
                    num_classes
                    * (i - 1) : (
                        num_classes * (i - 1)
                        + (self.params["dimz"] - num_classes) // num_classes
                    )
                ]

            with torch.no_grad():
                # lat = torch.tensor(lat, dtype=torch.float).to(self.device)
                # lat = torch.from_numpy(lat).float().to(self.device)
                if isinstance(lat, np.ndarray):
                    lat = torch.from_numpy(lat).float().to(self.device)
                else:
                    lat = lat.clone().detach().to(self.device)


                decoded = self.net.decode(lat).cpu()

            axs[i].imshow(decoded[0, 0].T)

        return fig

    def train_eval_plot_loop(self):
        """
        The main function more or less.
        Runs all of the training  and testing epochs in loops
        Gets and plots latent spaces as tsnes, latent traversals, and reconstructions
        saves checkpoints to view in tensorboard and for loading models later for additional training or inference
        """

        if not self.args.no_train:
            orig = (
                self.process_target(self.data_batch)
                .detach()
                .cpu()
                .view(*[[-1] + self.params["output_shape"]])[:, 0:1]
            )
            figure2 = plt.figure(99, figsize=(60, 60))
            ###
            num_images = orig.shape[0]
            for i in range(num_images):
                # Rotate each image by 90 degrees clockwise
                rotated_image = orig[i, 0, :, :]  # np.rot90(orig[i, 0, :, :], k=-1)

                # Create subplots for each image
                plt.subplot(1, num_images, i + 1)
                plt.imshow(
                    rotated_image,
                    cmap="rainbow",
                    vmin=rotated_image.min(),
                    vmax=rotated_image.max() * 0.5,
                )
            ###
            # plt.imshow(make_grid(orig, scale_each=True, normalize=True).transpose(0,2).numpy())
            if not self.args.no_save:
                self.writer.add_figure("original_train", figure2, global_step=1)

            for e in tqdm(range(self.starting_epoch, self.params["num_epochs"])):
                interval = e // self.params["lr_drop_interval"]
                for i, opt_ in enumerate(self.opt):
                    lr = self.opt.param_groups[-1]["lr"]
                    if interval > 0:
                        opt_.param_groups[-1]["lr"] = np.array(
                            self.params["learning_rate"][i]
                        ) / (interval * self.params["lr_drop_factor"])
                        print(
                            "Changing learning rate from {} to {}".format(
                                lr, opt_.param_groups[-1]["lr"]
                            )
                        )
                    else:
                        opt_.param_groups[-1]["lr"] = np.array(
                            self.params["learning_rate"][i]
                        )
                        print(
                            "Changing learning rate from {} to {}".format(
                                lr, opt_.param_groups[-1]["lr"]
                            )
                        )

                if (e % self.params["test_interval"]) == 0 and e != 0:
                    print("---------------Epoch {}-------------".format(e))
                    if not self.args.no_save:
                        print("---------Saving checkpoint---------")
                        save_checkpoint(
                            e,
                            self.checkpoint_dir,
                            self.net,
                            self.opt,
                            self.net.cls_sq,
                            self.inhib,
                        )

                    # test here

                    # tsne
                    lats, tgts = self.get_latent_space(self.train_dl, iterations=1)
                    lats_test, tgts_test = self.get_latent_space(
                        self.test_dl, iterations=3
                    )
                    if np.isnan(lats).any() or np.isinf(lats).any():
                        print("⚠️ Warning: NaN detected in latent space before PCA. Replacing with zeros.")
                        lats = np.nan_to_num(lats)

                    if np.isnan(lats_test).any() or np.isinf(lats_test).any():
                        print("⚠️ Warning: NaN detected in test latent space before PCA. Replacing with zeros.")
                        lats_test = np.nan_to_num(lats_test)


                    # latent space traversal
                    fig = self.latent_traversal(lats, tgts, 1)

                    fig_test = self.latent_traversal(lats_test, tgts_test, 1)

                    fig_switch = self.latent_traversal_switch(lats, tgts, 1, 2)

                    fig_inhib = self.latent_traversal_inhib(lats, tgts, 1)

                    _, figure, fig6, fig8 = self.tsne_project(lats, tgts)
                    _, figure2, fig7, fig9 = self.tsne_project(lats_test, tgts_test)

                    if not self.args.no_save:
                        self.writer.add_figure("latent_traversal", fig, global_step=e)
                        self.writer.add_figure(
                            "latent_traversal_test", fig_test, global_step=e
                        )
                        self.writer.add_figure(
                            "latent_traversal_switch", fig_switch, global_step=e
                        )
                        self.writer.add_figure(
                            "latent_traversal_inhib", fig_inhib, global_step=e
                        )
                        self.writer.add_figure("tsne_train", figure, global_step=e)
                        self.writer.add_figure("tsne_test", figure2, global_step=e)
                        self.writer.add_figure("exc_train", fig6, global_step=e)
                        self.writer.add_figure("exc_test", fig7, global_step=e)
                        self.writer.add_figure("inhib_train", fig8, global_step=e)
                        self.writer.add_figure("inhib_test", fig9, global_step=e)

                    # reconstruction !!!!!!!!!!!!!!
                    recon_batch, quantized, z_e, clas = self.net(
                        self.data_batch.to(self.device)
                    )
                    # print("recon_batch.shape = ", recon_batch.shape)
                    recon_batch_c = recon_batch.detach().cpu()
                    figure = plt.figure(figsize=(60, 60))
                    # Get the number of images in the batch
                    num_images = recon_batch_c.shape[0]

                    for i in range(num_images):
                        img = recon_batch_c[i].view([-1] + self.params["output_shape"])[
                            :, 0:1
                        ]
                        rotated_image = img[
                            0, 0, :, :
                        ].numpy()  # np.rot90(img[0, 0, :, :].numpy(), k=-1)

                        # Create subplots for each image
                        plt.subplot(1, num_images, i + 1)
                        plt.imshow(
                            rotated_image,
                            cmap="rainbow",
                            vmin=rotated_image.min(),
                            vmax=rotated_image.max() * 0.5,
                        )

                    if not self.args.no_save:
                        self.writer.add_figure("recon_train", figure, global_step=e)

                # train_here
                if self.params["is_guided"]:
                    loss_, excite_loss_, inhib_loss_, loss_abs_, entropy, means = (
                        self.train_guided_aug()
                    )
                    if not self.args.no_save:
                        self.writer.add_scalar("inhibitory_net_loss_1", excite_loss_, e)
                        self.writer.add_scalar("inhibitory_net_loss_2", inhib_loss_, e)
                        self.writer.add_scalar("clas_loss", loss_abs_, e)
                    # writer.add_scalar('entropy', entropy, e)
                    # writer.add_scalar('mean_values_softmax_inp', means, e)

                    # for i in range()
                    # tsne
                    lats, tgts = self.get_latent_space(self.train_dl, iterations=1)
                    lats_test, tgts_test = self.get_latent_space(
                        self.test_dl, iterations=1
                    )

                    train_acc = self.eval_accuracy(lats, tgts, True)
                    test_acc = self.eval_accuracy(lats_test, tgts_test, True)
                    if not self.args.no_save:
                        self.writer.add_scalar("vaeclas_net_train_acc", train_acc, e)
                        self.writer.add_scalar("vaeclas_net_test_acc", test_acc, e)
                else:
                    loss_ = self.train()
                if not self.args.no_save:
                    self.writer.add_scalar("train_loss", loss_, e)

                plt.close("all")  # close figures so they don't use too much memory...
                torch.cuda.empty_cache() ### suggested by Thorben


if __name__ == "__main__":
    # use this for testing. Should have separate files with command line arguments and such to handle training and testing stuff

    param_file = "/fzj_vpr/utils/test_params.yml"
    dataset_path = "/Users/nwang/all_preprocessed_dataset/chop_still_50_firstcamera_0/"
    dataset_path_test = (
        "/Users/nwang/all_preprocessed_dataset/chop_still_50_firstcamera_1/"
    )

    HGVAE = HybridGuidedVAETrainer(param_file, dataset_path, dataset_path_test)

    HGVAE.train_eval_plot_loop()
