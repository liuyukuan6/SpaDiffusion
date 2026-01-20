import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from tqdm import tqdm
import numpy as np
import os
import scanpy as sc
import torch.nn as nn 
from scipy import sparse
from torch_geometric.utils import add_self_loops, from_scipy_sparse_matrix
import math
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from scipy import sparse, stats
from scipy.stats import pearsonr, spearmanr





class LatentDataset(Dataset):
    def __init__(self, x_rna, x_atac, true_counts=None):
        self.x_rna = x_rna
        self.x_atac = x_atac
        self.true_counts = true_counts

    def __len__(self):
        return len(self.x_rna)

    def __getitem__(self, idx):
        # 输入 RNA latent，条件 ATAC latent，可选的真实计数
        if self.true_counts is not None:
            return self.x_rna[idx], self.x_atac[idx], self.true_counts[idx]
        else:
            return self.x_rna[idx], self.x_atac[idx]

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, timesteps, embedding_dim, dtype=torch.float32):
        device = timesteps.device
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)
        emb = timesteps.to(dtype).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb

class add_latent_noise(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    def forward(self, x_0, t, noise=None):
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(t.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(t.device)
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        x_noisy = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise
    


class ResBlock(nn.Module):

    def __init__(self,dim,dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim,dim)
            )
    def forward(self,x):
        return x + self.block(x)

class CrossAttention(nn.Module):
    def __init__(self, RNA_emb_dim, ATAC_emb_dim, feature_dim, proj_dim=64, num_heads=8, dropout=0.1):

        super().__init__()
        
        self.k_proj = nn.Linear(1, proj_dim)
        self.v_proj = nn.Linear(1, proj_dim)
        self.q_proj = nn.Linear(1, proj_dim)
        
        self.num_heads = num_heads
        
        self.cross_attention = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.out_proj = nn.Linear(proj_dim, 1)
        
        self.rna_norm = nn.LayerNorm(RNA_emb_dim)
        self.atac_norm = nn.LayerNorm(ATAC_emb_dim)

    def forward(self, query_emb, context_emb, mode="R2A"):

        query_norm = self.rna_norm(query_emb)
        context_norm = self.atac_norm(context_emb)

        query_unsqueezed = query_norm.unsqueeze(-1)
        context_unsqueezed = context_norm.unsqueeze(-1)

        q = self.q_proj(query_unsqueezed)
        k = self.k_proj(context_unsqueezed)
        v = self.v_proj(context_unsqueezed)

        attn_out, attn_weight = self.cross_attention(q, k, v)
        
        attn_out = self.out_proj(attn_out)

        output = query_unsqueezed + attn_out
        output = output.squeeze(-1)

        return output, context_emb, attn_weight

class DualCrossAttentionModule(nn.Module):
    def __init__(self, feature_dim, proj_dim=64, num_heads=8, dropout=0.1):
        super().__init__()

        self.rna_res_block1 = ResBlock(feature_dim, dropout)
        self.atac_res_block1 = ResBlock(feature_dim, dropout)
        self.rna_res_block2 = ResBlock(feature_dim, dropout)    
        self.atac_res_block2 = ResBlock(feature_dim, dropout)   

        self.attn_atac_from_rna = CrossAttention(
            RNA_emb_dim=feature_dim, 
            ATAC_emb_dim=feature_dim, 
            feature_dim=feature_dim, 
            proj_dim=proj_dim,       
            num_heads=num_heads, 
            dropout=dropout
        )
        
        self.attn_rna_from_atac = CrossAttention(
            RNA_emb_dim=feature_dim, 
            ATAC_emb_dim=feature_dim, 
            feature_dim=feature_dim,
            proj_dim=proj_dim,       
            num_heads=num_heads, 
            dropout=dropout
        )


        self.w_prime_rna = nn.Linear(feature_dim, feature_dim)
        self.w_prime_atac = nn.Linear(feature_dim, feature_dim)
        self.rna_res_block_final = ResBlock(feature_dim, dropout)
        self.atac_res_block_final = ResBlock(feature_dim, dropout)

    def forward(self, rna_feature, atac_feature):

        rna_x = self.rna_res_block1(rna_feature)
        atac_x = self.atac_res_block1(atac_feature)
        
        rna_res2_out = self.rna_res_block2(rna_x)
        atac_res2_out = self.atac_res_block2(atac_x)

        attn_rna_out, _, rna_weight = self.attn_rna_from_atac(rna_res2_out, atac_res2_out)
        attn_atac_out, _, atac_weight = self.attn_atac_from_rna(atac_res2_out, rna_res2_out)

        rna_sum = rna_res2_out + self.w_prime_rna(attn_atac_out)
        atac_sum = atac_res2_out + self.w_prime_atac(attn_rna_out)
        
        rna_final = self.rna_res_block_final(rna_sum)
        atac_final = self.atac_res_block_final(atac_sum)

        return rna_final, atac_final, rna_weight, atac_weight

class DiffusionBackbone(nn.Module):
    def __init__(self, feature_dim, time_emb_dim, cross_attn_proj_dim=64, num_heads=8, dropout=0.1):

        super().__init__()

        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, feature_dim)

        self.input_block = DualCrossAttentionModule(feature_dim, proj_dim=cross_attn_proj_dim, num_heads=num_heads, dropout=dropout)
        self.middle_block = DualCrossAttentionModule(feature_dim, proj_dim=cross_attn_proj_dim, num_heads=num_heads, dropout=dropout)
        self.output_block = DualCrossAttentionModule(feature_dim, proj_dim=cross_attn_proj_dim, num_heads=num_heads, dropout=dropout)

        self.atac_noise_head = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim,feature_dim)
        )

    def forward(self, r_t_input, a_t_input, time_t):
        time_emb = self.time_embedding(time_t, self.time_proj.in_features)
        time_emb_proj = self.time_proj(time_emb)

        rna_attn_weights = []
        atac_attn_weights = []

        r_t = r_t_input + time_emb_proj
        a_t = a_t_input + time_emb_proj

        r_out, a_out, r_w, a_w = self.input_block(r_t, a_t)
        rna_attn_weights.append(r_w)
        atac_attn_weights.append(a_w)
        r_out, a_out, r_w, a_w = self.middle_block(r_out, a_out)
        rna_attn_weights.append(r_w)
        atac_attn_weights.append(a_w)
        r_out, a_out, r_w, a_w = self.output_block(r_out, a_out)
        rna_attn_weights.append(r_w)
        atac_attn_weights.append(a_w)

        r_t_minus_1 = r_t_input + r_out 
        a_t_minus_1 = a_t_input + a_out
        
        return r_t_minus_1, a_t_minus_1, rna_attn_weights, atac_attn_weights



def compute_nb_loss(
    recon_mu_log: torch.Tensor,
    recon_theta_log: torch.Tensor,
    true_counts: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:

    recon_mu_log = torch.clamp(recon_mu_log, min=-15.0, max=15.0)
    recon_theta_log = torch.clamp(recon_theta_log, min=-15.0, max=15.0)


    recon_mu = torch.exp(recon_mu_log)                     
    recon_theta = F.softplus(recon_theta_log) + eps        

    logits = torch.log(recon_mu + eps) - torch.log(recon_theta + eps)

    nb_dist = NegativeBinomial(total_count=recon_theta, logits=logits)
    log_prob = nb_dist.log_prob(true_counts)

    if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
        return F.mse_loss(recon_mu, true_counts)

    return -log_prob.mean()




def train_model_with_vae_loss(
    model, 
    vae_model, 
    dataloader, 
    noise_adder, 
    optimizer, 
    num_epochs, 
    device,
    lambda_cosine_latent=0.0, 
    lambda_vae=1.0, 
    true_counts=None, 
    edge_index=None,
    vae_start_epoch=200
):


    model.train()
    vae_model.train()
    
    for param in vae_model.decoder.parameters():
        param.requires_grad = False
    for param in vae_model.encoder.parameters():
        param.requires_grad = False
    
    sqrt_alphas_cumprod = noise_adder.sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = noise_adder.sqrt_one_minus_alphas_cumprod.to(device)
    

    history = {"train_loss": [], "diffusion_loss": [], "vae_loss": []}
    

    with tqdm(total=num_epochs, desc="Total Progress", unit="epoch", position=0) as pbar_epoch:
        
        for epoch in range(num_epochs):
            model.train()
            vae_model.train()
            
            if epoch < vae_start_epoch:
                current_lambda_vae = 0.0
                train_vae = False
                for param in vae_model.decoder.parameters():
                    param.requires_grad = False
            else:
                current_lambda_vae = lambda_vae
                train_vae = True
                for param in vae_model.decoder.parameters():
                    param.requires_grad = True
            
            epoch_total_loss = 0.0
            epoch_diffusion_loss = 0.0
            epoch_vae_loss = 0.0
            num_batches = 0

            stage_info = "Diffusion only" if epoch < vae_start_epoch else "Diffusion + VAE"
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [{stage_info}]", leave=False, position=1)
            
            for batch_idx, batch_data in enumerate(progress_bar):
                if len(batch_data) == 3:
                    x_rna, x_atac, batch_true_counts = batch_data
                else:
                    x_rna, x_atac = batch_data
                    batch_true_counts = None
                
                x_rna = x_rna.to(device)
                x_atac = x_atac.to(device)
                batch_size = x_rna.shape[0]
                
                # ========== Diffusion Loss ==========
                t = torch.randint(0, noise_adder.num_timesteps, (batch_size,), device=device)
                x_rna_noisy, noise = noise_adder(x_rna, t)
                pred_noise, _, _, _ = model(x_rna_noisy, x_atac, t)
                diffusion_loss = F.mse_loss(pred_noise, noise)
                
                if lambda_cosine_latent > 0:
                    sqrt_alpha_t = sqrt_alphas_cumprod[t].reshape(-1, 1)
                    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
                    pred_x0 = (x_rna_noisy - sqrt_one_minus_alpha_t * pred_noise) / (sqrt_alpha_t + 1e-8)
                    cos_loss = 1 - F.cosine_similarity(pred_x0, x_rna, dim=1).mean()
                    diffusion_loss = diffusion_loss + lambda_cosine_latent * cos_loss
                
                # ========== VAE Loss ==========
                vae_loss = torch.tensor(0.0, device=device)
                
                if train_vae and current_lambda_vae > 0:
                    sqrt_alpha_t = sqrt_alphas_cumprod[t].reshape(-1, 1)
                    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
                    sqrt_alpha_t_safe = torch.clamp(sqrt_alpha_t, min=1e-6)
                    
                    pred_x0_latent = (x_rna_noisy - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t_safe
                    pred_x0_latent = torch.clamp(pred_x0_latent, min=-10.0, max=10.0).detach()
                    
                    decoder_output = vae_model.decoder(pred_x0_latent)
                    
                    if batch_true_counts is not None:
                        if isinstance(batch_true_counts, np.ndarray):
                            true_counts_batch = torch.tensor(batch_true_counts, dtype=torch.float32).to(device)
                        else:
                            true_counts_batch = batch_true_counts.to(device)
                    elif true_counts is not None:
                        start_idx = batch_idx * dataloader.batch_size
                        end_idx = min(start_idx + batch_size, len(true_counts))
                        if start_idx >= len(true_counts) or end_idx - start_idx < batch_size:
                            continue
                        true_counts_batch = torch.tensor(true_counts[start_idx:end_idx], dtype=torch.float32).to(device)
                    else:
                        continue 
                    

                    if getattr(vae_model, 'reconstruction_loss', 'mse') == 'nb':

                        recon_mu_log, recon_theta_log = torch.chunk(decoder_output, 2, dim=-1)
                        vae_loss = compute_nb_loss(recon_mu_log, recon_theta_log, true_counts_batch)
                    else:
                        recon_probs = torch.sigmoid(decoder_output)
                        vae_loss = F.mse_loss(recon_probs, true_counts_batch)
                

                if torch.isnan(diffusion_loss) or torch.isnan(vae_loss):
                    continue
                
                combined_loss = diffusion_loss + current_lambda_vae * vae_loss
                
                if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                    continue
                
                optimizer.zero_grad()
                combined_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if train_vae:
                    torch.nn.utils.clip_grad_norm_(vae_model.decoder.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_total_loss += combined_loss.item()
                epoch_diffusion_loss += diffusion_loss.item()
                epoch_vae_loss += vae_loss.item()
                num_batches += 1

                postfix_dict = {
                    'diff': f"{diffusion_loss.item():.4f}"
                }
                if train_vae:
                    postfix_dict['vae'] = f"{vae_loss.item():.4f}"
                    postfix_dict['total'] = f"{combined_loss.item():.4f}"
                
                progress_bar.set_postfix(postfix_dict)
            
            if num_batches > 0:
                avg_total = epoch_total_loss / num_batches
                avg_diff = epoch_diffusion_loss / num_batches
                avg_vae = epoch_vae_loss / num_batches
                
                history["train_loss"].append(avg_total)
                history["diffusion_loss"].append(avg_diff)
                history["vae_loss"].append(avg_vae)
                
            if epoch == vae_start_epoch - 1:

                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"Phase switch: Starting from epoch {vae_start_epoch + 1}, both Diffusion and VAE will be trained simultaneously")
                tqdm.write(f"{'='*60}\n")
            

            pbar_epoch.update(1)

            pbar_epoch.set_postfix({'Last_Loss': f"{avg_total:.4f}"} if num_batches > 0 else {})

    model.training_history = history
    return model

@torch.no_grad()
def ddim_sample(model, x_atac, noise_adder, num_steps=50, eta=0.0, device='cpu', 
                clip_value=10.0, verbose=True):

    if verbose:
        print(f"Starting DDIM sampling (steps={num_steps}, eta={eta})...")
    
    model.eval()
    batch_size, latent_dim = x_atac.shape
    
    x = torch.randn(batch_size, latent_dim).to(device)
    
    alphas_cumprod = noise_adder.alphas_cumprod.to(device)
    
    timesteps = torch.linspace(0, noise_adder.num_timesteps - 1, num_steps, dtype=torch.long).to(device)
    
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        
        pred_noise, _, _, _ = model(x, x_atac, t_batch)
        
        pred_noise = torch.clamp(pred_noise, -clip_value, clip_value)
        
        alpha_bar_t = alphas_cumprod[t]
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t + 1e-8)
        
        pred_x0 = torch.clamp(pred_x0, -clip_value, clip_value)
        
        if i > 0:
            t_prev = timesteps[i - 1]
            alpha_bar_t_prev = alphas_cumprod[t_prev]
            
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * 
                (1 - alpha_bar_t / alpha_bar_t_prev)
            )
            
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * pred_noise
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
            
            if eta > 0:
                noise = torch.randn_like(x)
                x = x + sigma * noise
        else:
            x = pred_x0
        

        if verbose and (i % 10 == 0 or i == len(timesteps) - 1):
            print(f"Step {len(timesteps)-i}/{len(timesteps)}: "
                    f"Mean={x.mean().item():.4f}, "
                    f"Std={x.std().item():.4f}, "
                    f"Range=[{x.min().item():.4f}, {x.max().item():.4f}]")
    
    if verbose:
        print("DDIM sampling completed!")
    
    return x