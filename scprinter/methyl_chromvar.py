import os
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import anndata as ad
import numpy as np
from anndata import AnnData

from .chromvar import *


def compute_deviations_methyl_batch(
    adata_paths,
    bg_peaks,
    motif_match,
    motif_name,
    save_path,
    peak_mask=None,
    fg_layer="mc",
    bg_layer="cov",
    gpus=[],
    chunk_size: int = 10000,
    tf_chunk_size: int = -1,
    save_key="",
    launch=True,
    overwrite=False,
    flip_sign=True,
    clean=True,
):
    """
    Compute methylation deviations in batch mode using multiple GPUs.
    Parameters
    ----------
    adata_paths
    bg_peaks: needs to be calculated after filtering peaks
    motif_match: after filtering peaks
    motif_name
    save_path
    peak_mask
    fg_layer
    bg_layer
    gpus
    chunk_size
    tf_chunk_size
    save_key
    launch
    overwrite
    flip_sign
    clean

    Returns
    -------

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    final_result_path = os.path.join(save_path, f"{save_key}_methyl_chromvar_deviations.h5ad")
    if os.path.exists(final_result_path) and (not overwrite):
        print(f"✅ Final result already exists at {final_result_path}, loading it.")
        return ad.read_h5ad(final_result_path)

    clean_files = []
    if type(adata_paths) is AnnData:
        # Now split it into multiple parts and save them
        adata = adata_paths
        batch_size = adata.n_obs // len(gpus) if len(gpus) > 0 else adata.n_obs
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
        expectation_var_fg = 0
        expectation_var_bg = 0
        for adata_path in tqdm(adata_paths, desc="calculating expectations"):
            adata = (
                ad.read_h5ad(adata_path)
                if adata_path.endswith(".h5ad")
                else ad.read_zarr(adata_path)
            )
            adata = adata[:, peak_mask] if peak_mask is not None else adata
            expectation_var_fg += np.asarray(
                adata.layers[fg_layer].sum(0), dtype=np.float32
            ).reshape((1, adata.X.shape[1]))
            expectation_var_bg += np.asarray(
                adata.layers[bg_layer].sum(0), dtype=np.float32
            ).reshape((1, adata.X.shape[1]))

        if np.min(expectation_var_bg) == 0:
            print(
                "⚠️ Warning: some peaks have zero coverage across all cells, "
                "which may lead to NaN deviations. "
                "Consider removing those peaks before computing deviations."
            )

        expectation_var_fg /= expectation_var_fg.sum()
        expectation_var_bg /= expectation_var_bg.sum()
        data_to_dump = [
            expectation_var_fg,
            expectation_var_bg,
            bg_peaks,
            peak_mask,
            motif_match,
            motif_name,
        ]

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
                "--fg_layer",
                fg_layer,
                "--bg_layer",
                bg_layer,
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
            + (["--verbose"] if (not launch) else [])
            + (["--flip_sign"] if flip_sign else [])
        )
        if launch:
            print(f"🚀 Starting job part {part} on GPU {gpu}")
            subprocess.run(command, env=env, check=True)
            print(f"✅ Finished job part {part} on GPU {gpu}")
        else:
            print("Please run the following command to compute deviations:")
            print(f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(command))
            time.sleep(1)

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
            adata_part = ad.read_h5ad(part_path)
            os.remove(part_path)  # Clean up part files
            adata_list.append(adata_part)
        if clean:
            for f in clean_files:
                print("Cleaning up", f)
                os.remove(f)
        adata_combined = ad.concat(adata_list, axis=0)
        adata_combined.write_h5ad(
            os.path.join(save_path, f"{save_key}_methyl_chromvar_deviations.h5ad")
        )

        return adata_combined


def compute_deviations_methyl(
    adata,
    fg_layer="mc",
    bg_layer="cov",
    chunk_size: int = 10000,
    device="cuda",
    tf_chunk_size: int = -1,
    expectation_var_fg=None,
    expectation_var_bg=None,
    bg_peaks=None,
    motif_match=None,
    motif_name=None,
    verbose=True,
    flip_sign=True,
):
    """
    Computes the deviation of motif matches from the background for each cell.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing the count matrix of the methylated reads and coverage in .layers['mc'] and .layers['cov'] respectively.
        It should also contain motif matches
        and background peaks in the varm. If not provided, the function will use the same object as adata_fg assuming it's already the ratio
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

    print("Computing expectation methylation ratio per cell and peak...")

    if expectation_var_fg is not None:
        print("Using provided expectation_var_fg")
        expectation_var_fg = backend.asarray(expectation_var_fg, dtype=backend.float32).reshape(
            (1, adata.X.shape[1])
        )
    else:
        print("Computing expectation_var_fg")
        expectation_var_fg = backend.asarray(
            adata.layers[fg_layer].sum(0), dtype=backend.float32
        ).reshape((1, adata.X.shape[1]))
    expectation_var_fg /= expectation_var_fg.sum()

    if expectation_var_bg is not None:
        print("Using provided expectation_var_bg")
        expectation_var_bg = backend.asarray(expectation_var_bg, dtype=backend.float32).reshape(
            (1, adata.X.shape[1])
        )
    else:
        print("Computing expectation_var_bg")
        expectation_var_bg = backend.asarray(
            adata.layers[bg_layer].sum(0), dtype=backend.float32
        ).reshape((1, adata.X.shape[1]))
    expectation_var_bg /= expectation_var_bg.sum()

    print("Computing expectation_obs_fg and bg")
    expectation_obs_fg = np.asarray(adata.layers[fg_layer].sum(1), dtype=np.float32).reshape(
        (adata.X.shape[0], 1)
    )
    expectation_obs_bg = np.asarray(adata.layers[bg_layer].sum(1), dtype=np.float32).reshape(
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
        mean_bg_dev = np.zeros_like(obs_dev)
        std_bg_dev = np.zeros_like(obs_dev)

        for start in tqdm(
            range(0, adata.n_obs, chunk_size), desc="Processing chunks", disable=not verbose
        ):
            end = min(start + chunk_size, adata.n_obs)
            X_chunk_fg = adata.layers[fg_layer][start:end].copy()
            X_chunk_bg = adata.layers[bg_layer][start:end].copy()
            expectation_obs_chunk_fg = backend.asarray(expectation_obs_fg[start:end])
            expectation_obs_chunk_bg = backend.asarray(expectation_obs_bg[start:end])
            if sparse.isspmatrix(X_chunk_fg):
                if device == "cuda":
                    X_chunk_fg = scipy_to_cupy_sparse(X_chunk_fg)
                    X_chunk_bg = scipy_to_cupy_sparse(X_chunk_bg)
                else:
                    X_chunk_fg = X_chunk_fg.tocsr()
                    X_chunk_bg = X_chunk_bg.tocsr()
            else:
                X_chunk_fg = backend.array(X_chunk_fg)
                X_chunk_bg = backend.array(X_chunk_bg)

            res = _compute_deviations_methyl(
                motif_match_tp,
                X_chunk_fg,
                X_chunk_bg,
                expectation_obs_chunk_fg,
                expectation_obs_chunk_bg,
                expectation_var_fg,
                expectation_var_bg,
                device=device,
            )
            obs_dev[start:end, :] = res.get() if device == "cuda" else res
            bg_dev_chunk = np.zeros(
                (n_bg_peaks, end - start, motif_match_tp.shape[1]), dtype=np.float32
            )
            for i in trange(n_bg_peaks, desc="Processing background peaks", disable=not verbose):
                bg_peak_idx = backend.array(bg_peaks[:, i]).flatten()
                bg_motif_match = motif_match_tp[bg_peak_idx, :]
                res = _compute_deviations_methyl(
                    bg_motif_match,
                    X_chunk_fg,
                    X_chunk_bg,
                    expectation_obs_chunk_fg,
                    expectation_obs_chunk_bg,
                    expectation_var_fg,
                    expectation_var_bg,
                    device=device,
                )
                bg_dev_chunk[i, :, :] = res.get() if device == "cuda" else res
            mean_bg_dev[start:end, :] = np.mean(bg_dev_chunk, axis=0)
            std_bg_dev[start:end, :] = np.std(bg_dev_chunk, axis=0)
            del X_chunk_fg, X_chunk_bg

        # mean_bg_dev = np.mean(bg_dev, axis=0)
        # std_bg_dev = np.std(bg_dev, axis=0)
        dev = (obs_dev - mean_bg_dev) / std_bg_dev
        dev = np.nan_to_num(dev, nan=0.0)
        dev_all.append(dev)

    dev = np.concatenate(dev_all, axis=1) if len(dev_all) > 1 else dev_all[0]
    if flip_sign:
        print("Flipping the sign of the deviations, so positive -> active TF")
        dev = -dev  # Flip the sign of the deviations, so positive -> active TF
    dev = AnnData(
        dev, dtype="float32", obs=adata.obs.copy()
    )  # Convert back to CPU for AnnData compatibility
    dev.var_names = motif_name
    return dev


def _compute_deviations_methyl(
    motif_match,
    count_fg,
    count_bg,
    expectation_obs_fg,
    expectation_obs_bg,
    expectation_var_fg,
    expectation_var_bg,
    device,
):
    if device == "cuda":
        import cupy as backend
        from cupyx.scipy.sparse import issparse as cupyx_issparse
    else:
        import numpy as backend

    observed = (count_fg @ motif_match) / (count_bg @ motif_match)

    if device == "cuda":
        if cupyx_issparse(observed):
            observed = observed.toarray()
    else:
        if sparse.issparse(observed):
            observed = observed.toarray()

    expected_fg = expectation_obs_fg @ (expectation_var_fg @ motif_match)
    expected_bg = expectation_obs_bg @ (expectation_var_bg @ motif_match)
    expected = expected_fg / expected_bg
    out = backend.zeros_like(expected)
    backend.divide(observed - expected, expected, out=out)
    out[expected == 0] = 0
    return out
