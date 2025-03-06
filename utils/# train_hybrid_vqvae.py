# train_hybrid_vqvae.py - Modified training script for Hybrid Guided VQ-VAE

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from hybrid_beta_vae_vq import VQVAE

class VQVAELoss(nn.Module):
    def __init__(self, beta=1.0, class_weight=1.0):
        super(VQVAELoss, self).__init__()
        self.beta = beta
        self.class_weight = class_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, recon_x, x, quantized, z_e, clas, target_class, inhib_output):
        # Reconstruction loss
        recon_loss = self.mse_loss(recon_x, x)
        
        # Codebook loss (Commitment loss)
        commitment_loss = self.beta * torch.mean((quantized.detach() - z_e) ** 2)
        
        # Quantization loss
        quantization_loss = torch.mean((quantized - z_e.detach()) ** 2)
        
        # Excitation classifier loss
        excite_loss = self.ce_loss(clas, target_class)
        
        # Inhibition classifier loss
        inhib_loss = self.ce_loss(inhib_output, target_class)
        
        # Total loss
        loss = recon_loss + commitment_loss + quantization_loss + self.class_weight * (excite_loss + inhib_loss)
        
        return loss, recon_loss, commitment_loss, quantization_loss, excite_loss, inhib_loss

class VQVAETrainer:
    def __init__(self, dataset, params, device="cuda"): 
        self.device = device
        self.params = params
        self.dataset = dataset
        
        # Initialize model
        self.model = VQVAE(
            input_shape=params["input_shape"],
            ngf=params["ngf"],
            out_features=params["out_features"],
            seq_len=params["seq_len"],
            dimz=params["dimz"],
            num_embeddings=params["num_embeddings"],
            encoder_params=params,
        ).to(self.device)
        
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["learning_rate"], betas=params["betas"])
        
        # Define loss function
        self.loss_fn = VQVAELoss(beta=params["vq_beta"], class_weight=params["class_weight"])
        
        # Load dataset
        self.train_loader = DataLoader(self.dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4)
        
    def train_step(self, x, target_class):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        recon_x, encoding_indices, clas = self.model(x.to(self.device))
        inhib_output = self.model.cls_sq(self.model.inhib.inhibit_z(self.model.encode(x.to(self.device))[0]))
        
        # Compute loss
        quantized, _ = self.model.vq_layer(self.model.bottleneck(self.model.encoder(x.to(self.device))[0]))
        loss, recon_loss, commitment_loss, quantization_loss, excite_loss, inhib_loss = self.loss_fn(
            recon_x, x.to(self.device), quantized, self.model.bottleneck(self.model.encoder(x.to(self.device))[0]), clas, target_class.to(self.device), inhib_output
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), recon_loss.item(), commitment_loss.item(), quantization_loss.item(), excite_loss.item(), inhib_loss.item()
    
    def train(self, epochs):
        for epoch in range(epochs):
            total_loss, total_recon, total_commit, total_quant, total_excite, total_inhib = 0, 0, 0, 0, 0, 0
            for x, target_class in tqdm(self.train_loader):
                loss, recon_loss, commit_loss, quant_loss, excite_loss, inhib_loss = self.train_step(x, target_class)
                total_loss += loss
                total_recon += recon_loss
                total_commit += commit_loss
                total_quant += quant_loss
                total_excite += excite_loss
                total_inhib += inhib_loss
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(self.train_loader):.4f}, Recon: {total_recon/len(self.train_loader):.4f}, Commit: {total_commit/len(self.train_loader):.4f}, Quant: {total_quant/len(self.train_loader):.4f}, Excite: {total_excite/len(self.train_loader):.4f}, Inhib: {total_inhib/len(self.train_loader):.4f}")
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
