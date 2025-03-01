import torch
import torch.nn as nn
import numpy as np
from .hybrid_beta_vae import VAE
from .quantizer import VectorQuantizer
# from models.decoder import Decoder


class VQVAE(VAE):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings_exc = 64, n_embeddings_inh = 64,
                 exc_embedding_dim = 16, inh_embedding_dim = 48, beta,
                 num_classes,
                 save_img_embedding_map=False):
        """
        n_embeddings_exc: number of embeddings for the excitation codebook.
        n_embeddings_inh: number of embeddings for the inhibition codebook.
        exc_embedding_dim: dimensionality of excitation embeddings.
        inh_embedding_dim: dimensionality of inhibition embeddings.
        beta: commitment cost.
        num_classes: number of location classes.


        comments:
        - The model is a VAE with a discrete bottleneck.
        - There are two codebooks, one for the excitation and one for the inhibition.
        - The encoder outputs a tensor of shape (batch, exc_embedding_dim + inh_embedding_dim).
        - The decoder takes a tensor of shape (batch, exc_embedding_dim + inh_embedding_dim) as input.
        - For Z , I think there are two options:
            First,it canbe split into two parts, one for the location and one for the non-location features.
            Second, it can be directly linearly transformed into two parts.
        - I am not sure about the use of self.cls_sq; Please consider.(--Zihan)
        """

        z_e = self.encoder(x)
        #option 1: split z_e into two parts
        z_e1, z_e2 = torch.split(z_e, [48, self.exc_embedding_dim], dim=1)  

        z_e_exc = self.pre_quant_conv_exc(z_e1)  # for location features
        z_e_inh = self.pre_quant_conv_inh(z_e2)  # for non-location features
        # option 2: linearly transform z_e into two parts
        
        # z_e1 = self.linear_ex(z_e)
        # z_e2 = self.linear_in(z_e)
        # z_e_exc = self.pre_quant_conv_exc(z_e1)  # for location features
        # z_e_inh = self.pre_quant_conv_inh(z_e2)  # for non-location features

        loss_exc, z_q_exc, perplexity_exc, _, _ = self.exc_vector_quantization(z_e_exc)
        loss_inh, z_q_inh, perplexity_inh, _, _ = self.inh_vector_quantization(z_e_inh)

        embedding_loss = loss_exc + loss_inh
        perplexity = (perplexity_exc + perplexity_inh) / 2.0
        
        z_q = torch.cat([z_q_exc, z_q_inh], dim=1)
        # reconstruct input
        x_hat = self.decoder(z_q)
        clas = self.cls_sq(excite_z)

        if verbose:
            print('Input shape:', x.shape)
            print('Encoder output shape:', z_e.shape)
            print('Exc quantized shape:', z_q_exc.shape)
            print('Inh quantized shape:', z_q_inh.shape)
            print('Reconstructed image shape:', x_hat.shape)

        return embedding_loss, x_hat, perplexity, z_q_exc, z_q_inh,clas