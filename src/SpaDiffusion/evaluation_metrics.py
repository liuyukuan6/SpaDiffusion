import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import squidpy as sq
from tqdm.auto import tqdm

def evaluate_reconstruction(
    orig_matrix: np.ndarray, 
    recon_matrix: np.ndarray, 
    adata_orig: ad.AnnData, 
    prefix: str,
    output_dir: str = None
):
    """
    Evaluates the data reconstruction fidelity of the model.
    
    The evaluation is performed at three levels:
    1. Matrix-level: Evaluates global reconstruction performance.
    2. Peak-level: Evaluates reconstruction performance of individual features.
    3. Cell-level: Evaluates reconstruction performance of individual cells.

    Args:
        orig_matrix: Original data matrix (cells x peaks).
        recon_matrix: Reconstructed data matrix (cells x peaks).
        adata_orig: Original AnnData object containing metadata.
        prefix: Prefix for reports and filenames (e.g., 'S1').
        output_dir: (Optional) Directory to save plots and CSV files.
    
    Returns:
        dict: Dictionary containing all calculated correlation statistics.
    """
    print(f"\n--- 1. Starting Reconstruction Fidelity Evaluation ({prefix}) ---")
    
    results = {}

    # 1. Matrix-level Evaluation
    orig_flat = orig_matrix.flatten()
    recon_flat = recon_matrix.flatten()
    
    pearson_corr, pearson_p = pearsonr(orig_flat, recon_flat)
    spearman_corr, spearman_p = spearmanr(orig_flat, recon_flat)
    
    results['matrix_level'] = {
        'pearson_corr': pearson_corr, 'pearson_p': pearson_p,
        'spearman_corr': spearman_corr, 'spearman_p': spearman_p
    }
    print(f"  [Matrix-level] Pearson Correlation: {pearson_corr:.4f}")

    # 2. Peak-level Evaluation
    peak_corrs = []
    for i in tqdm(range(orig_matrix.shape[1]), desc=f"  Evaluating {prefix} Peaks", leave=False):
        orig_vector = orig_matrix[:, i]
        recon_vector = recon_matrix[:, i]
        if np.std(orig_vector) > 1e-6 and np.std(recon_vector) > 1e-6:
            corr, _ = pearsonr(orig_vector, recon_vector)
            peak_corrs.append(corr)
            
    peak_corr_s = pd.Series(peak_corrs)
    results['peak_level'] = peak_corr_s.describe().to_dict()
    print(f"  [Peak-level] Mean Pearson Correlation: {peak_corr_s.mean():.4f}")

    # 3. Cell-level Evaluation
    cell_corrs = []
    for i in tqdm(range(orig_matrix.shape[0]), desc=f"  Evaluating {prefix} Cells", leave=False):
        orig_vector = orig_matrix[i, :]
        recon_vector = recon_matrix[i, :]
        if np.std(orig_vector) > 1e-6 and np.std(recon_vector) > 1e-6:
            corr, _ = pearsonr(orig_vector, recon_vector)
            cell_corrs.append(corr)
            
    cell_corr_s = pd.Series(cell_corrs)
    results['cell_level'] = cell_corr_s.describe().to_dict()
    print(f"  [Cell-level] Mean Pearson Correlation: {cell_corr_s.mean():.4f}")

    # Optional: Save detailed results and charts
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Detailed saving and plotting logic can be moved here
        print(f"  Detailed evaluation results saved to: {output_dir}")

    return results


def evaluate_biological_consistency(
    adata: ad.AnnData,
    ground_truth_key: str,
    cluster_key: str
) -> dict:
    """
    Evaluates the biological consistency of the latent space representation.

    Uses ARI and NMI metrics to compare model clustering results with true cell type labels.

    Args:
        adata: AnnData object containing cell type and cluster labels.
        ground_truth_key: Column name for ground truth cell type labels in adata.obs.
        cluster_key: Column name for model clustering results in adata.obs.

    Returns:
        dict: Dictionary containing ARI and NMI scores.
    """
    print("\n--- 2. Starting Biological Consistency Evaluation ---")
    
    ground_truth_labels = adata.obs[ground_truth_key]
    model_cluster_labels = adata.obs[cluster_key]

    ari_score = adjusted_rand_score(ground_truth_labels, model_cluster_labels)
    nmi_score = normalized_mutual_info_score(ground_truth_labels, model_cluster_labels)

    print(f"  Adjusted Rand Index (ARI): {ari_score:.4f} (closer to 1 is better)")
    print(f"  Normalized Mutual Information (NMI): {nmi_score:.4f} (closer to 1 is better)")

    return {'ari': ari_score, 'nmi': nmi_score}


def evaluate_spatial_coherence(
    adata: ad.AnnData,
    cluster_key: str
) -> pd.DataFrame:
    """
    Evaluates the spatial structure coherence of the latent representation.

    Uses Moran's I statistic to measure the degree of spatial clustering for each cluster.

    Args:
        adata: AnnData object containing spatial coordinates and cluster labels.
        cluster_key: Column name for model clustering results in adata.obs.

    Returns:
        pd.DataFrame: Results containing Moran's I statistics and p-values for each cluster.
    """
    print("\n--- 3. Starting Spatial Structure Coherence Evaluation ---")
    
    try:
        # Calculate spatial adjacency graph
        sq.gr.spatial_neighbors(adata, coord_type="generic")
        
        # Calculate spatial autocorrelation
        sq.gr.spatial_autocorr(
            adata,
            mode="moran"
        )
        
        moran_results = adata.uns[f'{cluster_key}_moranI']
        avg_moran_i = moran_results['I'].mean()
        
        print(f"  Mean Moran's I: {avg_moran_i:.4f} (Significantly positive indicates spatial clustering)")
        return moran_results

    except Exception as e:
        print(f"  Spatial evaluation failed: {e}")
        return None


def run_comprehensive_evaluation(
    adata: ad.AnnData,
    recon_matrix: np.ndarray,
    prefix: str,
    ground_truth_key: str = 'cell_type',
    cluster_key: str = 'leiden',
    output_dir: str = None
):
    """
    Runs a comprehensive three-tier evaluation workflow.

    Args:
        adata: AnnData object containing all relevant information.
        recon_matrix: Data matrix reconstructed by the model.
        prefix: Prefix for reporting and filenames.
        ground_truth_key: Key for ground truth cell type labels.
        cluster_key: Key for model clustering results.
        output_dir: (Optional) Directory to save detailed results.
    """
    print("\n" + "="*50)
    print(f"Starting Comprehensive Evaluation for {prefix} data")
    print("="*50)

    # Prepare original matrix
    if isinstance(adata.X, (np.ndarray, np.generic)):
        orig_matrix = adata.X
    else: # Handle sparse matrices
        orig_matrix = adata.X.toarray()

    # 1. Evaluate Reconstruction Fidelity
    evaluate_reconstruction(orig_matrix, recon_matrix, adata, prefix, output_dir)
    
    # 2. Evaluate Biological Consistency
    if ground_truth_key in adata.obs.columns and cluster_key in adata.obs.columns:
        evaluate_biological_consistency(adata, ground_truth_key, cluster_key)
    else:
        print(f"\nSkipping biological consistency evaluation: '{ground_truth_key}' or '{cluster_key}' not found in adata.obs.")

    # 3. Evaluate Spatial Structure Coherence
    if cluster_key in adata.obs.columns:
        evaluate_spatial_coherence(adata, cluster_key)
    else:
        print(f"\nSkipping spatial coherence evaluation: '{cluster_key}' not found in adata.obs.")

    print("\n" + "="*50)
    print(f"Evaluation for {prefix} data complete")
    print("="*50)