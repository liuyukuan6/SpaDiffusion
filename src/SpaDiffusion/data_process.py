import os
import numpy as np
import scanpy as sc
from torch_geometric.utils import add_self_loops, from_scipy_sparse_matrix
from scipy import sparse
import torch

import scanpy as sc
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops

def build_spatial_graph(adata, n_neighbors=3, device='cpu'):
    
    if "connectivities" not in adata.obsp:
        print(f"Computing spatial neighbors (n={n_neighbors})...")
        if "spatial" not in adata.obsm:
             raise ValueError("Key 'spatial' not found in adata.obsm. Cannot build spatial graph.")
        
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="spatial")

    adj = adata.obsp["connectivities"]

    edge_index = from_scipy_sparse_matrix(adj)[0]
    
    edge_index, _ = add_self_loops(edge_index)
    
    return edge_index.to(device)
def data_to_tensor(adata, device):

    if isinstance(adata.X, sparse.spmatrix):
        x_tensor = torch.tensor(adata.X.toarray(), dtype=torch.float32, device=device)
    else:
        x_tensor = torch.tensor(adata.X, dtype=torch.float32, device=device)
    return x_tensor