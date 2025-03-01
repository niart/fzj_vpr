import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta,
                 num_classes,
                 excitation_indices, inhibition_indices,
                 save_img_embedding_map=False):
        """
        excitation_indices: list of codebook indices that are designated for excitation (location features).
        inhibition_indices: list of codebook indices for inhibition (non-location features).
        num_classes: number of location classes (e.g., 16).


        comments:
        -just a glamsp of the model, not the full code.
        -Complete before finnal conversation.

        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)

        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)

        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)


        self.num_classes = num_classes
        self.exc_indices = excitation_indices
        self.inh_indices = inhibition_indices




    def forward(self, x, verbose=False):

        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        
        
        B, C, H, W = z_q.shape
        z_q_flat = z_q.view(B, C, -1)  # shape (B, C, H*W)
        min_enc_indices_flat = min_encoding_indices.view(B, -1)  # shape (B, H*W)
        
        device = z_q.device


        exc_mask = torch.zeros_like(min_enc_indices_flat, dtype=torch.bool, device=device)
        for idx in self.exc_indices:
            exc_mask |= (min_enc_indices_flat == idx)
        # Compute sum per sample over excitation positions
        exc_sum = torch.where(exc_mask.unsqueeze(1), z_q_flat, torch.zeros_like(z_q_flat)).sum(dim=2)
        exc_count = exc_mask.sum(dim=1).unsqueeze(1).float() + 1e-8
        exc_rep = exc_sum / exc_count   # shape (B, C)
        
       
        inh_mask = torch.zeros_like(min_enc_indices_flat, dtype=torch.bool, device=device)
        for idx in self.inh_indices:
            inh_mask |= (min_enc_indices_flat == idx)
        inh_sum = torch.where(inh_mask.unsqueeze(1), z_q_flat, torch.zeros_like(z_q_flat)).sum(dim=2)
        inh_count = inh_mask.sum(dim=1).unsqueeze(1).float() + 1e-8
        inh_rep = inh_sum / inh_count   # shape (B, C)
        
       
        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('quantized data shape:', z_q.shape)
            print('recon data shape:', x_hat.shape)
            print('perplexity:', perplexity)
        
        return embedding_loss, x_hat, perplexity, exc_rep, inh_rep 