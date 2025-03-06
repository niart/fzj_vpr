#modified by Zihan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2

    """

    def __init__(self, n_e = 32, e_dim = 64, beta = 0.25)#, device = 'cuda:0'):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)#.to(device)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)
        orignal:
        z.shape = (batch, channel, height, width)
        changed:
        Z_q.shape = (batch, channel)
        channel = 64
        modified by Zihan

        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        # z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        # import ipdb; ipdb.set_trace()
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape 
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices



#`python -m fzj_vpr.utils.quantizer` to test
"""
test result:
Loss:  tensor(1.2141, device='cuda:0', grad_fn=<AddBackward0>)
Perplexity:  tensor(19.2304, device='cuda:0')
z_q:  torch.Size([32, 64])
min_encodings:  torch.Size([32, 32])
min_encoding_indices:  torch.Size([32, 1])

Zihan
"""
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = VectorQuantizer(32, 64, 0.25 , device)
    z = torch.randn(32, 64).to(device)
    loss, z_q, perplexity, min_encodings, min_encoding_indices = vq(z)
    print("Loss: ", loss)
    print("Perplexity: ", perplexity)
    print("z_q: ", z_q.shape)
    print("min_encodings: ", min_encodings.shape)
    print("min_encoding_indices: ", min_encoding_indices.shape)