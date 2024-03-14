from ..utils import regionparser

import numpy as np
import cupy as cp
import cupyx as cx
import scipy
from pynndescent import NNDescent
from tqdm.auto import tqdm, trange
from typing import Union
import cupy as cp
import logging
from tqdm.auto import tqdm
import scipy.sparse as sparse
from anndata import AnnData

from scipy.sparse import csr_matrix as scipy_csr_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

def GC_content(dna_string):
    a,c,g,t = 0,0,0,0
    # Iterate over each character in the string and increment the corresponding count
    for nucleotide in dna_string:
        if nucleotide == 'G':
            g += 1
        elif nucleotide == 'C':
            c += 1
        elif nucleotide == 'A':
            a += 1
        elif nucleotide == 'T':
            t += 1

    return a,c,g,t


def get_bias(adata, genome):


    peaks=  regionparser(list(adata.var_names))
    gc_contents = []
    overall_freq = [0,0,0,0]
    for chrom, start, end in tqdm(peaks.iloc[:, 0:3].values, desc="Fetching GC content"):
        seq = genome.fetch_seq(chrom, start, end).upper()
        a,c,g,t = GC_content(seq)
        overall_freq[0] += a
        overall_freq[1] += c
        overall_freq[2] += g
        overall_freq[3] += t
        gc_contents.append((g+c) / (a+c+g+t))
    gc_contents = np.asarray(gc_contents)
    gc_contents[np.isnan(gc_contents)] = 0.5
    adata.var['gc_content'] = gc_contents
    overall_freq = np.array(overall_freq)
    overall_freq = overall_freq / np.sum(overall_freq)
    adata.uns['bg_freq'] = overall_freq
    return None


def sample_bg_peaks(adata,
                    method='nndescent',
                    niterations=50,
                    w = 0.1,
                    bs = 50,
                    n_jobs=-1):
    assert method in ['nndescent', 'chromvar'], "Method not supported"
    reads_per_peak = adata.X.sum(axis=0)
    assert np.min(reads_per_peak) > 0, "Some peaks have no reads"
    reads_per_peak = np.log10(reads_per_peak)
    reads_per_peak = np.array(reads_per_peak).reshape((-1))
    mat = np.array([reads_per_peak, adata.var['gc_content'].values])
    chol_cov_mat = np.linalg.cholesky(np.cov(mat))
    trans_norm_mat = scipy.linalg.solve_triangular(
        a=chol_cov_mat, b=mat, lower=True).transpose()
    print ("Sampling nearest neighbors")
    if method == "nndescent":
        index = NNDescent(trans_norm_mat, metric="euclidean",
                          n_neighbors=niterations, n_jobs=n_jobs)
        knn_idx, _ = index.query(trans_norm_mat, niterations)

        adata.varm['bg_peaks'] = knn_idx
    elif method == 'chromvar':
        knn_idx = create_bins_and_sample_background(trans_norm_mat,
                                                    bs=bs,
                                                    w=w,
                                                    niterations=niterations)
        adata.varm['bg_peaks'] = knn_idx

    return knn_idx



def create_bins_and_sample_background(trans_norm_mat, bs, w, niterations):
    # Create bins
    bins1 = np.linspace(np.min(trans_norm_mat[:, 0]),
                        np.max(trans_norm_mat[:, 0]), bs)
    bins2 = np.linspace(np.min(trans_norm_mat[:, 1]),
                        np.max(trans_norm_mat[:, 1]), bs)

    # Create bin_data
    bin_data = np.array(np.meshgrid(bins1, bins2)).T.reshape(-1, 2)

    # Calculate Euclidean distances
    bin_dist = scipy.spatial.distance.cdist(bin_data, bin_data, 'euclidean')

    # Calculate probabilities
    bin_p = scipy.stats.norm.pdf(bin_dist, 0, w)
    # Find nearest bin membership for each point in trans_norm_mat
    index = NNDescent(bin_data, metric="euclidean",
                          n_neighbors=1)
    indices, _ = index.query(trans_norm_mat, 1)
    bin_membership = indices.flatten()
    # Calculate bin density
    unique, counts = np.unique(bin_membership, return_counts=True)
    bin_density = np.zeros(bs ** 2)
    bin_density[unique] = counts

    # Sample background peaks
    # This assumes bg_sample_helper is defined as per previous instruction
    background_peaks = bg_sample_helper(bin_membership,
                                        bin_p,
                                        bin_density,
                                        niterations)

    return background_peaks


def bg_sample_helper(bin_membership, bin_p, bin_density, niterations):
    n = len(bin_membership)
    out = np.zeros((n, niterations), dtype=int)

    for i in trange(len(bin_density), desc='Sampling background peaks'):
        ix = np.where(bin_membership == i)[0]
        if len(ix) == 0:  # Skip if no members in bin
            continue
        p_tmp = bin_p[i, ]
        p = (p_tmp/ bin_density)[bin_membership]
        p /= p.sum()
        # Sampling with replacement according to probabilities
        sampled_indices = np.random.choice(np.arange(len(p)),
                                           size=niterations * len(ix),
                                           replace=True,
                                           p=p)
        out[ix, :] = sampled_indices.reshape((len(ix), niterations))

    return out



def scipy_to_cupy_sparse(sparse_matrix):
    if not isinstance(sparse_matrix, scipy_csr_matrix):
        raise ValueError("Input matrix must be a SciPy CSR matrix")

    # Get the CSR components of the SciPy sparse matrix
    data = sparse_matrix.data.astype('float')
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    shape = sparse_matrix.shape

    # Convert the components to CuPy arrays
    data_cp = cp.array(data)
    indices_cp = cp.array(indices)
    indptr_cp = cp.array(indptr)

    # Create a CuPy CSR matrix with these components
    cupy_sparse_matrix = cupy_csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
    return cupy_sparse_matrix


def compute_deviations_gpu(adata, chunk_size: int = 10000):
    assert "bg_peaks" in adata.varm_keys(), "Cannot find background peaks in the input object, please first run get_bg_peaks!"

    logging.info('Computing expectation reads per cell and peak...')
    expectation_var = cp.asarray(adata.X.sum(0), dtype=np.float32).reshape((1, adata.X.shape[1]))
    expectation_var /= expectation_var.sum()
    expectation_obs = cp.asarray(adata.X.sum(1), dtype=np.float32).reshape((adata.X.shape[0], 1))
    motif_match = cp.array(adata.varm['motif_match'])
    obs_dev = cp.zeros((adata.n_obs, motif_match.shape[1]), dtype=cp.float32)

    n_bg_peaks = adata.varm['bg_peaks'].shape[1]
    bg_dev = cp.zeros((n_bg_peaks, adata.n_obs, motif_match.shape[1]), dtype=cp.float32)

    for start in tqdm(range(0, adata.n_obs, chunk_size), desc="Processing chunks"):
        end = min(start + chunk_size, adata.n_obs)
        X_chunk = adata.X[start:end]

        if sparse.isspmatrix(X_chunk):
            X_chunk = scipy_to_cupy_sparse(X_chunk)
        else:
            X_chunk = cp.array(X_chunk)

        obs_dev[start:end, :] = _compute_deviations_gpu(motif_match, X_chunk, expectation_obs[start:end],
                                                        expectation_var)

        for i in trange(n_bg_peaks):
            bg_peak_idx = cp.array(adata.varm['bg_peaks'][:, i]).flatten()
            bg_motif_match = motif_match[bg_peak_idx, :]
            bg_dev[i, start:end, :] = _compute_deviations_gpu(bg_motif_match, X_chunk, expectation_obs[start:end],
                                                              expectation_var)

    mean_bg_dev = cp.mean(bg_dev, axis=0)
    std_bg_dev = cp.std(bg_dev, axis=0)
    dev = (obs_dev - mean_bg_dev) / std_bg_dev
    dev = cp.nan_to_num(dev, nan=0.0)

    dev = AnnData(dev.get(), dtype='float32')  # Convert back to CPU for AnnData compatibility
    dev.obs_names = adata.obs_names
    dev.var_names = adata.uns['motif_name']
    return dev


def _compute_deviations_gpu(motif_match, count, expectation_obs, expectation_var):
    observed = count.dot(motif_match)
    expected = expectation_obs.dot(expectation_var.dot(motif_match))
    out = cp.zeros_like(expected)
    cp.divide(observed - expected, expected, out=out)
    out[expected == 0] = 0
    return out