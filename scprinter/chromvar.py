import logging
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Literal, Union

import anndata
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sparse
from anndata import AnnData
from pynndescent import NNDescent
from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse import issparse
from tqdm.auto import tqdm, trange

from .utils import get_peak_bias, get_peak_CpG_density


def sample_bg_peaks(
    adata,
    genome,
    method: Literal["nndescent", "chromvar"] = "nndescent",
    bg_set=None,
    niterations=50,
    w=0.1,
    bs=50,
    n_jobs=1,
    bg_columns=["gc_content", "reads_per_peak"],
):
    """
    This function samples background peaks for chromVAR analysis in single-cell ATAC-seq data.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell ATAC-seq data.
    genome : str or scp.genome.Genome
        scp.genome.Genome object or string specifying the genome.
    method : str, optional
        Method to use for sampling background peaks. Can be either "nndescent" or "chromvar".
        Defaults to "nndescent".
    bg_set : array-like, optional
        Custom background peak set to use. The background peaks would only be selected from
        this set of peaks, instead of all peaks present in the dataset. Only applicable when
        method is "nndescent". Defaults to None.
    niterations : int, optional
        Number of nearest neighbors to sample for each peak. Defaults to 50.
    w : float, optional
        Width parameter for Gaussian kernel density estimation. Defaults to 0.1. Only used in
        "chromvar" mode.
    bs : int, optional
        Bin size for creating bins and sampling background peaks. Defaults to 50. Only used in
        "chromvar" mode.
    n_jobs : int, optional
        Number of jobs to run in parallel for nearest neighbor calculation. Defaults to 1.
    bg_columns: list of str, optional
        List of columns in adata.var to use for background sampling. Defaults to ["gc_content"].
        If "gc_content" is in the list, but not in the adata.var already, it will compute GC bias using get_peak_bias function.

    Returns
    -------
    array-like
        Array containing indices of sampled background peaks for each peak.
    """
    if "gc_content" in bg_columns:
        if "gc_bias" not in adata.var.columns:
            get_peak_bias(adata, genome)
    if "cpg_density" in bg_columns:
        if "cpg_density" not in adata.var.columns:
            get_peak_CpG_density(adata, genome)

    assert method in ["nndescent", "chromvar"], "Method not supported"

    if "reads_per_peak" in bg_columns:
        if hasattr(adata.X, "compute"):
            reads_per_peak = 0
            for b in range(len(adata.X.chunks[0])):
                block = adata.X.blocks[b, 0].compute()
                reads_per_peak += block.sum(axis=0)

            reads_per_peak = np.array(reads_per_peak).reshape((-1))
        else:
            reads_per_peak = np.array(adata.X.sum(axis=0)).flatten()

        assert np.min(reads_per_peak) > 0, "Some peaks have no reads"
        reads_per_peak = np.log10(reads_per_peak)
        reads_per_peak = np.array(reads_per_peak).reshape((-1))
        adata.var["reads_per_peak"] = reads_per_peak
    if len(bg_columns) > 1:
        mat = np.array(adata.var[list(bg_columns)].values).T
        chol_cov_mat = np.linalg.cholesky(np.cov(mat))
        trans_norm_mat = scipy.linalg.solve_triangular(
            a=chol_cov_mat, b=mat, lower=True
        ).transpose()
    else:
        mat = np.array([reads_per_peak])
        trans_norm_mat = mat.transpose()
        # print (trans_norm_mat.shape)

    print("Sampling nearest neighbors")
    if bg_set is not None:
        assert (
            method == "nndescent"
        ), "Only nndescent method is supported for custom background peaks"
    if method == "nndescent":
        index = NNDescent(
            trans_norm_mat[bg_set if bg_set is not None else slice(None)],
            metric="euclidean",
            n_neighbors=niterations + 1,
            n_jobs=n_jobs,
        )
        knn_idx, _ = index.query(trans_norm_mat, niterations + 1)
        knn_idx = knn_idx[:, 1:]
        if bg_set is not None:
            knn_idx = bg_set[knn_idx.reshape((-1))].reshape((-1, niterations))
        adata.varm["bg_peaks"] = knn_idx
    elif method == "chromvar":
        knn_idx = create_bins_and_sample_background(
            trans_norm_mat, bs=bs, w=w, niterations=niterations
        )
        adata.varm["bg_peaks"] = knn_idx

    return knn_idx


def create_bins_and_sample_background(trans_norm_mat, bs, w, niterations):
    """
    Translated from the chromVAR R package.
    Parameters
    ----------
    trans_norm_mat
    bs
    w
    niterations

    Returns
    -------

    """
    # Create bins
    bins1 = np.linspace(np.min(trans_norm_mat[:, 0]), np.max(trans_norm_mat[:, 0]), bs)
    bins2 = np.linspace(np.min(trans_norm_mat[:, 1]), np.max(trans_norm_mat[:, 1]), bs)

    # Create bin_data
    bin_data = np.array(np.meshgrid(bins1, bins2)).T.reshape(-1, 2)

    # Calculate Euclidean distances
    bin_dist = scipy.spatial.distance.cdist(bin_data, bin_data, "euclidean")

    # Calculate probabilities
    bin_p = scipy.stats.norm.pdf(bin_dist, 0, w)
    # Find nearest bin membership for each point in trans_norm_mat
    print("NNDescent", bin_data.shape)
    index = NNDescent(bin_data, metric="euclidean", n_neighbors=1, n_jobs=1)
    indices, _ = index.query(trans_norm_mat, 1)
    # distance = scipy.spatial.distance.cdist(trans_norm_mat, bin_data)
    # indices = np.argmin(distance, axis=1)
    bin_membership = indices.flatten()
    # Calculate bin density
    unique, counts = np.unique(bin_membership, return_counts=True)
    bin_density = np.zeros(bs**2)
    bin_density[unique] = counts

    # Sample background peaks
    # This assumes bg_sample_helper is defined as per previous instruction
    background_peaks = bg_sample_helper(bin_membership, bin_p, bin_density, niterations)

    return background_peaks


def bg_sample_helper(bin_membership, bin_p, bin_density, niterations):
    n = len(bin_membership)
    out = np.zeros((n, niterations), dtype=int)

    for i in trange(len(bin_density), desc="Sampling background peaks"):
        ix = np.where(bin_membership == i)[0]
        if len(ix) == 0:  # Skip if no members in bin
            continue
        p_tmp = bin_p[i,]
        p = (p_tmp / bin_density)[bin_membership]
        p /= p.sum()
        # Sampling with replacement according to probabilities
        sampled_indices = np.random.choice(
            np.arange(len(p)), size=niterations * len(ix), replace=True, p=p
        )
        out[ix, :] = sampled_indices.reshape((len(ix), niterations))

    return out


def scipy_to_cupy_sparse(sparse_matrix, csr_matrix=None):
    """
    A function that converts a SciPy sparse matrix to a CuPy sparse matrix. Only supports CSR matrices now.
    Parameters
    ----------
    sparse_matrix

    Returns
    -------

    """
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

    if not isinstance(sparse_matrix, scipy_csr_matrix):
        raise ValueError("Input matrix must be a SciPy CSR matrix")

    # Get the CSR components of the SciPy sparse matrix
    data = sparse_matrix.data.astype("float")
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


def compute_deviations(
    adata,
    chunk_size: int = 10000,
    device="cuda",
    tf_chunk_size: int = -1,
    expectation_var: Union[np.ndarray, None] = None,
    expectation_obs: Union[np.ndarray, None] = None,
    bg_peaks=None,
    motif_match=None,
    motif_name=None,
    verbose=True,
):
    """
    Computes the deviation of motif matches from the background for each cell.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing the count matrix, background peaks, and motif match information.
    chunk_size : int, optional
        The size of chunks to process the data in. Default is 10000. It is recommended to set this
        such that there's no GPU memory overflow. Although our implementation would tolerate that,
        it will be much slower if there's overflow.
    device : str, optional
        The device to use for computation. Can be either "cuda" for GPU or "cpu" for CPU. Default is "cuda".

    Returns
    -------
    dev : AnnData
        The AnnData object containing the deviation values for each cell and motif match.
    """

    if bg_peaks is None:
        assert (
            "bg_peaks" in adata.varm_keys()
        ), "Cannot find background peaks in the input object, please first run get_bg_peaks!"
        bg_peaks = adata.varm["bg_peaks"]

    if device == "cuda":
        import cupy as backend
    else:
        import numpy as backend

    print("Computing expectation reads per cell and peak...")
    if expectation_var is None:
        if "expectation_var" not in adata.var:
            expectation_var = backend.asarray(adata.X.sum(0), dtype=backend.float32).reshape(
                (1, adata.X.shape[1])
            )
        else:
            print("precomputed expectation_var")
            expectation_var = backend.asarray(
                adata.var["expectation_var"], dtype=backend.float32
            ).reshape((1, adata.X.shape[1]))
    else:
        print("Using provided expectation_var")
        expectation_var = backend.asarray(expectation_var, dtype=backend.float32).reshape(
            (1, adata.X.shape[1])
        )
    expectation_var /= expectation_var.sum()

    if expectation_obs is None:
        if "expectation_obs" not in adata.obs:
            expectation_obs = np.asarray(adata.X.sum(1), dtype=np.float32).reshape(
                (adata.X.shape[0], 1)
            )
        else:
            print("precomputed expectation_obs")
            expectation_obs = np.asarray(adata.obs["expectation_obs"], dtype=np.float32).reshape(
                (adata.X.shape[0], 1)
            )
    else:
        print("Using provided expectation_obs")
        expectation_obs = np.asarray(expectation_obs, dtype=np.float32).reshape(
            (adata.X.shape[0], 1)
        )

    if motif_match is None:
        assert (
            "motif_match" in adata.varm_keys()
        ), "Cannot find motif match in the input object, please first run get_motif_match!"
        motif_match = adata.varm["motif_match"]
    if motif_name is None:
        motif_name = adata.uns["motif_name"]
    if tf_chunk_size < 0:
        tf_chunk_size = motif_match.shape[1]
    dev_all = []
    for motif_match_start in tqdm(
        list(range(0, motif_match.shape[1], tf_chunk_size)),
        desc="Processing TF chunks",
        disable=not verbose,
    ):
        motif_match_tp = motif_match[:, motif_match_start : motif_match_start + tf_chunk_size]

        if sparse.isspmatrix(motif_match_tp):
            if device == "cuda":
                motif_match_tp = scipy_to_cupy_sparse(motif_match_tp)
            else:
                motif_match_tp = motif_match_tp.tocsr()
        else:
            motif_match_tp = backend.asarray(motif_match_tp, dtype=backend.float32)

        obs_dev = np.zeros((adata.n_obs, motif_match_tp.shape[1]), dtype=np.float32)
        n_bg_peaks = bg_peaks.shape[1]
        # bg_dev = np.zeros((n_bg_peaks, adata.n_obs, motif_match.shape[1]), dtype=np.float32)
        mean_bg_dev = np.zeros_like(obs_dev)
        std_bg_dev = np.zeros_like(obs_dev)

        for start in tqdm(
            range(0, adata.n_obs, chunk_size), desc="Processing chunks", disable=not verbose
        ):
            end = min(start + chunk_size, adata.n_obs)
            temp_adata = adata[start:end].copy()
            X_chunk = temp_adata.X
            expectation_obs_chunk = backend.asarray(expectation_obs[start:end])
            if sparse.isspmatrix(X_chunk):
                if device == "cuda":
                    X_chunk = scipy_to_cupy_sparse(X_chunk)
                else:
                    X_chunk = X_chunk.tocsr()
            else:
                X_chunk = backend.array(X_chunk)
            res = _compute_deviations(
                motif_match_tp,
                X_chunk,
                expectation_obs_chunk,
                expectation_var,
                device=device,
            )
            obs_dev[start:end, :] = res.get() if device == "cuda" else res
            bg_dev_chunk = np.zeros(
                (n_bg_peaks, end - start, motif_match_tp.shape[1]), dtype=np.float32
            )
            for i in trange(n_bg_peaks, desc="Processing background peaks", disable=not verbose):
                bg_peak_idx = backend.array(bg_peaks[:, i]).flatten()
                bg_motif_match = motif_match_tp[bg_peak_idx, :]
                res = _compute_deviations(
                    bg_motif_match,
                    X_chunk,
                    expectation_obs_chunk,
                    expectation_var,
                    device=device,
                )
                bg_dev_chunk[i, :, :] = res.get() if device == "cuda" else res
            mean_bg_dev[start:end, :] = np.mean(bg_dev_chunk, axis=0)
            std_bg_dev[start:end, :] = np.std(bg_dev_chunk, axis=0)
            del temp_adata, X_chunk

        # mean_bg_dev = np.mean(bg_dev, axis=0)
        # std_bg_dev = np.std(bg_dev, axis=0)
        dev = (obs_dev - mean_bg_dev) / std_bg_dev
        dev = np.nan_to_num(dev, nan=0.0)
        dev_all.append(dev)

    dev = np.concatenate(dev_all, axis=1) if len(dev_all) > 1 else dev_all[0]
    dev = AnnData(
        dev, dtype="float32", obs=adata.obs.copy()
    )  # Convert back to CPU for AnnData compatibility
    dev.var_names = motif_name
    return dev


def _compute_deviations(motif_match, count, expectation_obs, expectation_var, device):
    if device == "cuda":
        import cupy as backend
        from cupyx.scipy.sparse import issparse as cupyx_issparse
    else:
        import numpy as backend

    observed = count @ motif_match
    if device == "cuda":
        if cupyx_issparse(observed):
            observed = observed.toarray()
    else:
        if sparse.issparse(observed):
            observed = observed.toarray()
    expected = expectation_obs @ (expectation_var @ motif_match)
    out = backend.zeros_like(expected)
    backend.divide(observed - expected, expected, out=out)
    out[expected == 0] = 0
    return out


def compute_deviations_batch(
    adata_paths,
    bg_peaks,
    motif_match,
    motif_name,
    save_path,
    gpus=[],
    chunk_size: int = 10000,
    tf_chunk_size: int = -1,
    save_key="",
    launch=True,
    overwrite=False,
    clean=True,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    final_result_path = os.path.join(save_path, f"{save_key}_chromvar_deviations.h5ad")
    if os.path.exists(final_result_path) and (not overwrite):
        print(f"✅ Final result {final_result_path} already exists, skipping computation.")
        return anndata.read_h5ad(final_result_path)

    clean_files = []
    if type(adata_paths) is AnnData:
        # Now split it into multiple parts and save them
        adata = adata_paths
        batch_size = (adata.n_obs // len(gpus) + 1) if len(gpus) > 0 else adata.n_obs
        adata_paths = []
        for i in range(0, adata.n_obs, batch_size):
            part_path = os.path.join(save_path, f"{save_key}_input_part_{len(adata_paths)}.h5ad")
            tmp = adata[i : i + batch_size].copy()
            index_name = tmp.obs_names.copy()
            del tmp.obs
            del tmp.obsm
            del tmp.varm
            tmp.obs_names = index_name
            tmp.write_h5ad(part_path)
            adata_paths.append(part_path)
        print(f"Split input AnnData into {len(adata_paths)} parts and saved to {save_path}")
        clean_files = adata_paths.copy()

    tmp_file_path = os.path.join(save_path, f"{save_key}_tmp_info.pkl")
    clean_files.append(tmp_file_path)
    if (not os.path.exists(tmp_file_path)) or overwrite:
        # Compute expectations and save intermediate
        expectation_var = 0

        for adata_path in tqdm(adata_paths, desc="calculating expectations"):
            adata = anndata.read_h5ad(adata_path)

            expectation_var += np.asarray(adata.X.sum(0), dtype=np.float32).reshape(
                (1, adata.X.shape[1])
            )
        expectation_var /= expectation_var.sum()

        data_to_dump = [expectation_var, bg_peaks, motif_match, motif_name]

        with open(tmp_file_path, "wb") as f:
            pickle.dump(data_to_dump, f)
        print(f"✅ Saved expectation values and other data to {tmp_file_path}")
    else:
        print(f"✅ Using existing expectation values and other data from {tmp_file_path}")

    if not gpus:
        print("⚠️ No GPUs specified, using CPU for computation.")
        gpus = ["cpu"]

    # Prepare available GPU queue
    available_gpus = Queue()
    for gpu in gpus:
        available_gpus.put(gpu)

    # Prepare job queue
    job_queue = Queue()
    all_finished = True
    for idx, path in enumerate(adata_paths):
        part_path = os.path.join(save_path, f"{save_key}_part_{idx}.h5ad")
        if not os.path.exists(part_path):
            job_queue.put((path, idx))
            all_finished = False

    def run_job(path, part, gpu, launch=True, overwrite=False):
        device = "cuda" if gpu != "cpu" else "cpu"
        env = os.environ.copy()
        if gpu != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        out_path = os.path.join(save_path, f"{save_key}_part_{part}.h5ad")

        command = (
            [
                "methyl_chromvar_entrance",
                "--adata",
                str(path),
                "--chunk_size",
                str(chunk_size),
                "--tf_chunk_size",
                str(tf_chunk_size),
                "--device",
                str(device),
                "--info_pickle",
                str(tmp_file_path),
                "--save_path",
                str(out_path),
            ]
            + (["--overwrite"] if overwrite else [])
            + (["--verbose"] if not launch else [])
        )
        if launch:
            print(f"🚀 Starting job part {part} on GPU {gpu}")
            subprocess.run(command, env=env, check=True)
            print(f"✅ Finished job part {part} on GPU {gpu}")
        else:
            print("Please run the following command to compute deviations:")

            print(f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(command))
        return gpu  # return freed GPU

    # Parallel execution
    futures = set()
    print(job_queue.qsize(), "jobs to run on", len(gpus), "GPUs")
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        # Initial fill: one job per GPU
        while (not job_queue.empty()) and (not available_gpus.empty()):
            path, part = job_queue.get()
            gpu = available_gpus.get()
            futures.add(executor.submit(run_job, path, part, gpu, launch, overwrite))

        while not job_queue.empty():
            for done in as_completed(futures):
                freed_gpu = done.result()
                path, part = job_queue.get()
                futures.add(executor.submit(run_job, path, part, freed_gpu, launch, overwrite))
                futures.remove(done)  # Remove the completed future
                break
        # wait till all jobs are done
        executor.shutdown(wait=True)

    # all finishes, concat results
    if launch or all_finished:
        print("All jobs completed. Concatenating results...")
        adata_list = []
        for part in range(len(adata_paths)):
            part_path = os.path.join(save_path, f"{save_key}_part_{part}.h5ad")
            adata_part = anndata.read_h5ad(part_path)
            os.remove(part_path)  # Clean up part files
            adata_list.append(adata_part)
        if clean:
            for f in clean_files:
                print("Cleaning up", f)
                os.remove(f)
        adata_combined = anndata.concat(adata_list, axis=0)
        adata_combined.write_h5ad(os.path.join(save_path, f"{save_key}_chromvar_deviations.h5ad"))

        return adata_combined


def bag_deviations(
    adata=None, ranked_df=None, cutoff=0.7, motif_motif_matrix=None, collapse_greater=True
):
    """
    This function performs a bagging operation on transcription factors (TFs) based on their correlation with each other.
    It selects a representative TF (sentinel TF) for each group of TFs that have a correlation coefficient greater than or equal to the specified threshold.

    Parameters
    ----------
    adata : AnnData, optional
        An AnnData object containing the count matrix and TF information. If provided, TF variability will be computed from this object.
    ranked_df : DataFrame, optional
        A DataFrame containing TF names and their variability scores. If provided, TF variability will be computed from this object.
    cutoff : float, optional
        The coefficient threshold. TFs with a correlation coefficient greater (if collapse_higher=True, otherwise smaller) than or equal to this threshold will be grouped together.
    motif_motif_matrix : str | pd.DataFrame
        The path to the motif motif matrix file or a dataframe. The common options are `scp.datasets.FigR_motifs_bagging_mouse` and `scp.datasets.FigR_motifs_bagging_human`.
    collapse_greater : bool, optional
        Whether to collapse TFs with coefficient greater than or equal to the threshold.
        If False, TFs with coefficient smaller than or equal to the threshold will be collapsed.
        For instance, if it's correlation based method, set as True, if it's tomtom or testing based metod, set as False
    Returns
    -------
    DataFrame
        If an AnnData object is provided, a DataFrame containing the selected sentinel TFs and their corresponding TF groups.
    list
        If a DataFrame object is provided, a list containing the selected sentinel TFs.
    list
        If a DataFrame object is provided, a list containing the TF groups.
    """

    assert motif_motif_matrix is not None, "Motif correlation matrix must be provided"
    assert adata is not None or ranked_df is not None, "Either adata or ranking_df must be provided"

    # Compute variability and get transcription factors (TFs)
    if adata is not None:
        x = adata.X
        if type(adata.X) not in [np.ndarray]:
            x = x.get()
        vb = np.nanstd(x, axis=0)
        vb = pd.DataFrame({"variability": vb}, index=adata.var.index)
        vb = vb.sort_values(by=["variability"], ascending=False)
        TFnames = vb.index.tolist()
    else:
        TFnames = ranked_df.index.tolist()
    TFnames_to_rank = {tf: i for i, tf in enumerate(TFnames)}
    # Import correlation based on PWMs for the organism
    if type(motif_motif_matrix) is pd.DataFrame:
        cormat = motif_motif_matrix
    else:
        cormat = pd.read_csv(
            motif_motif_matrix, sep="\t"
        )  # Assuming the RDS file contains one object

    tf1 = cormat.iloc[:, 0]
    tf2 = cormat.iloc[:, 1]

    assert set(TFnames).issubset(
        set(tf1).union(set(tf2))
    ), "All TF names must be in the correlation matrix"
    cormat = cormat[(tf1.isin(TFnames)) & (tf2.isin(TFnames))]
    tf1 = cormat.iloc[:, 0]
    tf2 = cormat.iloc[:, 1]

    if not collapse_greater:
        cormat.iloc[:, 2] = cormat.iloc[:, 2] * -1
        cutoff = -1 * cutoff
    factor = np.array(cormat.iloc[:, 2])
    i = 1
    TFgroups = []
    while len(TFnames) != 0:
        tfcur = TFnames[0]
        boo = np.array(((tf1 == tfcur) | (tf2 == tfcur)) & (factor >= cutoff))
        hits = cormat[boo]
        tfhits = list(set(list(np.unique(hits.iloc[:, :2])) + [tfcur]))

        # Update lists
        TFnames = [tf for tf in TFnames if tf not in tfhits]
        TFgroups.append(tfhits)
        cormat = cormat[tf1.isin(TFnames) & tf2.isin(TFnames)]
        tf1 = cormat.iloc[:, 0]
        tf2 = cormat.iloc[:, 1]
        factor = np.array(cormat.iloc[:, 2])
        i += 1

    sentinalTFs = []
    for group in TFgroups:
        ranks = [TFnames_to_rank[tf] if tf in TFnames_to_rank else 1e9 for tf in group]
        sentinalTFs.append(group[np.argmin(ranks)])
    if adata is not None:
        objReduced = adata.var.loc[sentinalTFs]
        objReduced["groups"] = TFgroups
        return objReduced
    else:
        return sentinalTFs, TFgroups


def permutation_test(
    adata: AnnData,
    groupby: str,
    group1: str,
    group2=None,
    n_iterations: int = 1000,
):

    group = np.copy(adata.obs[groupby].values)
    fg = adata.X[group == group1]
    bg = adata.X[group == group2] if group2 is not None else adata.X[group != group1]

    obs_stat = np.array(fg.mean(axis=0) - bg.mean(axis=0))
    bg_stat = []
    for i in trange(n_iterations):
        np.random.shuffle(group)
        fg = adata.X[group == group1]
        bg = adata.X[group == group2] if group2 is not None else adata.X[group != group1]
        bg_stat.append(np.array(fg.mean(axis=0) - bg.mean(axis=0)))
    bg_stat_mean = np.mean(bg_stat, axis=0)
    bg_stat_std = np.std(bg_stat, axis=0)

    zscore = (obs_stat - bg_stat_mean) / bg_stat_std

    pval = scipy.stats.norm.cdf(zscore, 0, 1)
    pval = np.minimum(pval, 1 - pval)

    res = pd.DataFrame(
        {
            "stat": obs_stat,
            "zscore": zscore,
            "pval": pval,
        },
        index=adata.var_names,
    )

    return res


def motif_peak_ztest(
    adata=None, peak_set: np.ndarray = None, bg_peaks=None, motif_match=None, motif_names=None
):
    """
    Peak-level motif enrichment Z-test (chromVAR style)

    Parameters
    ----------
    peak_set : array-like
        Vector of peak indices (reference peak space)
    bg_peaks : ndarray
        [n_peaks x n_bg_iterations] integer matrix of sampled background peak indices
    motif_match : sparse.csr_matrix
        [n_peaks x n_motifs] binary peak-by-motif matrix
    motif_names : list (optional)
        Column names for motifs (default: use positions 0..n_motifs-1)
    Returns
    -------
    pandas.DataFrame
        Motif enrichment results, sorted by p-value then motif frequency.
    """
    if adata is not None:
        print("Using adata to extract motif_match, motif_names and bg_peaks")
        motif_match = adata.varm["motif_match"]
        motif_names = adata.uns["motif_name"]
        bg_peaks = adata.varm["bg_peaks"]

    n_peaks, n_motifs = motif_match.shape
    assert n_peaks == bg_peaks.shape[0], "TF matrix and background peak matrix must match in #peaks"
    assert np.all(
        np.isin(peak_set, np.arange(n_peaks))
    ), "peak_set contains indices beyond available peaks"

    if motif_names is None:
        motif_names = np.array([f"motif_{i}" for i in range(n_motifs)])

    # ---- Observed motif frequency in selected peaks ----
    print("Getting selected peak motif frequencies ...")
    if issparse(motif_match):
        obs_freq = motif_match[peak_set].mean(axis=0)
        obs_freq = np.asarray(obs_freq).flatten()
    else:
        obs_freq = motif_match[peak_set].mean(axis=0)

    # ---- Background motif frequencies ----
    print("Getting background peak motif frequencies ...")
    bg_idx_for_peaks = bg_peaks[peak_set]  # subset rows

    # bg_freq_tab: shape [n_motifs, n_iterations]
    bg_freq_tab = np.zeros((n_motifs, bg_idx_for_peaks.shape[1]), dtype=np.float32)
    for j in range(bg_idx_for_peaks.shape[1]):
        bgp = bg_idx_for_peaks[:, j]
        if issparse(motif_match):
            fr = motif_match[bgp].mean(axis=0)
            bg_freq_tab[:, j] = np.asarray(fr).flatten()
        else:
            bg_freq_tab[:, j] = motif_match[bgp].mean(axis=0)

    print("Calculating empirical & Z-test p-values ...")
    # background stats
    bg_mean = bg_freq_tab.mean(axis=1)
    bg_std = bg_freq_tab.std(axis=1, ddof=1)
    z = (obs_freq - bg_mean) / bg_std
    z[np.isnan(z)] = 0.0
    z[np.isinf(z)] = 0.0

    pval = scipy.stats.norm.cdf(z, 0, 1)
    pval = np.minimum(pval, 1 - pval)
    res = pd.DataFrame(
        {
            "motif_obs_freq": obs_freq,
            "zscore": z,
            "pval": pval,
        },
        index=motif_names,
    )

    # sort: p-val then observed freq
    res = res.sort_values(by=["pval", "motif_obs_freq"], ascending=[True, False])

    return res
