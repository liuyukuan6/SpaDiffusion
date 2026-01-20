import os 
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from typing import List,Tuple,Optional
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops,from_scipy_sparse_matrix
from torch.distributions import NegativeBinomial, Normal, kl_divergence as kl
from scipy import sparse
from tqdm.auto import tqdm

def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ATACDataset(Dataset):
    def __init__(self, ids: np.ndarray):
        self.ids = ids
    def __len__(self) -> int:
        return len(self.ids)
    def __getitem__(self, idx: int) -> int:
        return self.ids[idx]



class ATAC_Encoder(nn.Module):
    def __init__(self,dim_list:List[int],dropout_rate:float=0.2):
        super().__init__()
        self.n_layers = len(dim_list) - 1
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.shortcuts = nn.ModuleList() 
        
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                self.layers.append(GCNConv(dim_list[i],dim_list[i+1]))
                self.shortcuts.append(nn.Identity())
            else:
                self.layers.append(nn.Linear(dim_list[i],dim_list[i+1]))
                self.shortcuts.append(nn.Sequential(
                    nn.Linear(dim_list[i],dim_list[i+1]),
                    nn.BatchNorm1d(dim_list[i+1])
                ))
            
            self.layer_norms.append(nn.LayerNorm(dim_list[i+1]))
        
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self,x:torch.Tensor,edge_index:torch.Tensor)->torch.Tensor:
        for i in range(self.n_layers):
            layer_input = x
            if i == self.n_layers - 1:
                f_x = self.layers[i](layer_input,edge_index)
                f_x = self.layer_norms[i](f_x)
                x = F.leaky_relu(f_x,0.01)
            else:
                f_x = self.dropout(layer_input)
                f_x = self.layers[i](f_x)
                f_x = self.layer_norms[i](f_x)
                shortcut_x = self.shortcuts[i](layer_input)
                x = F.leaky_relu(f_x + shortcut_x,0.01)
        return x
    
class ATAC_Decoder(nn.Module):
    def __init__(self,dim_list:List[int],dropout_rate:float=0.2):
        super().__init__()
        self.n_layers = len(dim_list) - 1
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for i in range(self.n_layers):
            self.layers.append(nn.Linear(dim_list[i],dim_list[i+1]))

            if i != self.n_layers - 1: 
                self.layer_norms.append(nn.LayerNorm(dim_list[i+1]))
                self.shortcuts.append(nn.Sequential(
                    nn.Linear(dim_list[i],dim_list[i+1]),
                    nn.BatchNorm1d(dim_list[i+1])
                ))
            else:
                self.shortcuts.append(nn.Identity())

        self.dropout = nn.Dropout(dropout_rate)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        for i in range(self.n_layers):
            layer_input = x
            f_x = self.dropout(layer_input)
            f_x = self.layers[i](f_x)
            if i != self.n_layers - 1:
                f_x = self.layer_norms[i](f_x)
                shortcut_x = self.shortcuts[i](layer_input)
                x = F.leaky_relu(f_x + shortcut_x,0.01)
            else:
                x = f_x
        return x

class Spatial_VAE(nn.Module):
    def __init__(self,
                  input_dim:int,
                  encoder_dims:List[int],
                  latent_dim:int,
                  decoder_dims:List[int],
                  reconstruction_loss:str='nb',
                  dropout_rate:float=0.2
                  ):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        
        self.encoder = ATAC_Encoder(dim_list=[input_dim]+encoder_dims,dropout_rate = dropout_rate)
        self.fc_mu = nn.Linear(encoder_dims[-1],latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1],latent_dim)
        
        decoder_output_dim = input_dim*2 if reconstruction_loss == 'nb' else input_dim
        self.decoder = ATAC_Decoder(
            dim_list=[latent_dim] + decoder_dims + [decoder_output_dim],
            dropout_rate = dropout_rate
        )
        
        if self.reconstruction_loss == 'nb':
            self.theta_act = nn.Softplus() 
        
    def reparameterize(self,mu: torch.Tensor,logvar: torch.Tensor)-> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu+eps*std
    def forward(self,
                x:torch.Tensor,
                edge_index: torch.Tensor,
                deterministic:bool=False
                ) -> dict:
        encoded_features = self.encoder(x,edge_index)
        mu = self.fc_mu(encoded_features)
        logvar = self.fc_logvar(encoded_features)

        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu,logvar)
        decoder_output = self.decoder(z)

        if self.reconstruction_loss == 'bce':
            recon_logits = decoder_output
            return {"recon_logits":recon_logits,"mu":mu,"logvar":logvar}
        elif self.reconstruction_loss == 'nb':
            recon_mu_log,recon_theta_log = torch.chunk(decoder_output,2,dim=-1)
            recon_mu = torch.exp(recon_mu_log)
            recon_theta = self.theta_act(recon_theta_log)
            return {"recon_mu":recon_mu,"recon_theta":recon_theta,"mu":mu,"logvar":logvar}
        
    def get_latent_representation(self,
                                  x:torch.Tensor,
                                  edge_index:torch.Tensor
                                  ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            encoded_features = self.encoder(x,edge_index)
            mu = self.fc_mu(encoded_features)
        return mu.cpu().numpy()
    

class VAE_Trainer:

    def __init__(self, adata: sc.AnnData, model: nn.Module, use_gpu: bool = True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"device: {self.device}")
        
        self.model = model.to(self.device)
        self.adata = adata
        
        self.full_tensor = self._get_tensor_from_adata(adata)
        self.edge_index = self._get_edge_index_from_adata(adata)

    def _get_tensor_from_adata(self, adata: sc.AnnData) -> torch.Tensor:
        if isinstance(adata.X, sparse.csr_matrix) or isinstance(adata.X, sparse.csc_matrix):
            return torch.tensor(adata.X.toarray(), dtype=torch.float32).to(self.device)
        else:
            return torch.tensor(adata.X, dtype=torch.float32).to(self.device)

    def _get_edge_index_from_adata(self, adata: sc.AnnData) -> torch.Tensor:
        if 'connectivities' not in adata.obsp:
            raise ValueError("Please run sc.pp.neighbors(adata, use_rep='spatial') ")
        adj_matrix = adata.obsp['connectivities']
        edge_index, _ = add_self_loops(from_scipy_sparse_matrix(adj_matrix)[0])
        return edge_index.to(self.device)

    def _vae_loss_function(self, model_output: dict, x_batch: torch.Tensor, kl_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mu = model_output['mu']
        logvar = model_output['logvar']
        
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.model.reconstruction_loss == 'bce':
            recon_logits = model_output['recon_logits']
            recon_loss = F.binary_cross_entropy_with_logits(recon_logits, (x_batch > 0).float(), reduction="mean")
        elif self.model.reconstruction_loss == 'nb':
            recon_mu = model_output['recon_mu']
            recon_theta = model_output['recon_theta']
            nb_dist = NegativeBinomial(total_count=recon_theta, logits=torch.log(recon_mu) - torch.log(recon_theta))
            recon_loss = -torch.mean(nb_dist.log_prob(x_batch))
        else:
            raise ValueError("Unsupported reconstruction loss type. Please choose 'bce' or 'nb'.")

        total_loss = recon_loss + kl_weight * kl_div
        return total_loss, recon_loss, kl_div
        
    def train(self, epochs: int = 300, batch_size: int = 128, lr: float = 0.001, kl_warmup_epochs: int = 50, beta: float = 1):
        train_ids = np.arange(self.adata.shape[0]) 
        train_dataset = ATACDataset(train_ids)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
        print("--- VAE Training Start ---")
        self.history = {"train_loss": [], "recon_loss": [], "kl_loss": []}

        with tqdm(total=epochs, desc="Training", dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                self.model.train()
                kl_weight = beta * min(1.0, epoch / kl_warmup_epochs)
            
                epoch_total_loss = []
                epoch_recon_loss = []
                epoch_kl_loss = []


                for batch_node_ids in train_dataloader:
                    model_output_full = self.model(self.full_tensor, self.edge_index)
                    batch_input = self.full_tensor[batch_node_ids]
                    batch_output = {key: val[batch_node_ids] for key, val in model_output_full.items()}
                
                    loss, recon_loss, kl_div = self._vae_loss_function(batch_output, batch_input, kl_weight)
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    epoch_total_loss.append(loss.item())
                    epoch_recon_loss.append(recon_loss.item())
                    epoch_kl_loss.append(kl_div.item())
            
                avg_total_loss = np.mean(epoch_total_loss)
                avg_recon_loss = np.mean(epoch_recon_loss)
                avg_kl_loss = np.mean(epoch_kl_loss)

                self.history["train_loss"].append(avg_total_loss)
                self.history["recon_loss"].append(avg_recon_loss)
                self.history["kl_loss"].append(avg_kl_loss)

                pbar.set_postfix({
                    'Loss': f'{avg_total_loss:.4f}',
                    'Recon': f'{avg_recon_loss:.4f}',
                    'KL': f'{avg_kl_loss:.4f}',
                    'KL_wt': f'{kl_weight:.2f}'
                })
            
                pbar.update(1)

        print("--- Done ---\n")
    def get_reconstruction(self) -> np.ndarray:
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.full_tensor, self.edge_index, deterministic=True)
            if self.model.reconstruction_loss == 'bce':
                recon_probs = torch.sigmoid(output['recon_logits'])
                return recon_probs.cpu().numpy()
            elif self.model.reconstruction_loss == 'nb':
                
                return output['recon_mu'].cpu().numpy()