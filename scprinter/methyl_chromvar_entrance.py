import argparse
import gc
import os.path
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import anndata
import cupy as cp
import h5py
import numpy as np
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

import scprinter as scp

rmm.reinitialize(
    managed_memory=True,  # Allows oversubscription
    pool_allocator=True,  # default is False
    devices=0,  # GPU device IDs to register. By default registers only GPU 0.
)

cp.cuda.set_allocator(rmm_cupy_allocator)

parser = argparse.ArgumentParser(description="Entrance for Methylation ChromVAR analysis")
parser.add_argument("--adata", type=str, required=True, help="Path to anndata file")
parser.add_argument(
    "--fg_layer", type=str, default=None, help="layer that stores the methylation data"
)
parser.add_argument(
    "--bg_layer", type=str, default=None, help="layer that stores the coverage data"
)
parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
parser.add_argument(
    "--tf_chunk_size", type=int, default=100, help="Chunk size for transcription factors"
)
parser.add_argument("--device", type=str, default="cuda", help="Device to run the analysis on")
parser.add_argument(
    "--info_pickle", type=str, required=True, help="Path to read the info pickle file"
)
parser.add_argument("--save_path", type=str, required=True, help="Path to save the output results")
parser.add_argument(
    "--overwrite", action="store_true", help="Overwrite existing results if they exist"
)
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
parser.add_argument("--flip_sign", action="store_true", help="Flip the sign of the deviations")


def main():
    args = parser.parse_args()
    with open(args.info_pickle, "rb") as f:
        info = pickle.load(f)

    adata = (
        anndata.read_h5ad(args.adata)
        if args.adata.endswith(".h5ad")
        else anndata.read_zarr(args.adata)
    )

    if (not os.path.exists(args.save_path)) or args.overwrite:
        if os.path.exists(args.save_path):
            print(f"Overwriting existing file: {args.save_path}")
        else:
            print(f"Creating new file: {args.save_path}")
        if (args.fg_layer is not None) and (args.bg_layer is not None):
            print("Computing methylation deviations...")
            expectation_var_fg, expectation_var_bg, bg_peaks, peak_mask, motif_match, motif_name = (
                info
            )
            if peak_mask is not None:
                adata = adata[:, peak_mask].copy()
            dev = scp.methyl_chromvar.compute_deviations_methyl(
                adata,
                fg_layer=args.fg_layer,
                bg_layer=args.bg_layer,
                expectation_var_fg=expectation_var_fg,
                expectation_var_bg=expectation_var_bg,
                bg_peaks=bg_peaks,
                motif_match=motif_match,
                motif_name=motif_name,
                chunk_size=args.chunk_size,
                tf_chunk_size=args.tf_chunk_size,
                device=args.device,
                verbose=args.verbose,
                flip_sign=args.flip_sign,
            )
            # Save the results
            dev.write(args.save_path)
        else:
            print("Compute atac deviations...")
            expectation_var, bg_peaks, motif_match, motif_name = info
            dev = scp.chromvar.compute_deviations(
                adata,
                expectation_var=expectation_var,
                bg_peaks=bg_peaks,
                motif_match=motif_match,
                motif_name=motif_name,
                chunk_size=args.chunk_size,
                tf_chunk_size=args.tf_chunk_size,
                device=args.device,
                verbose=args.verbose,
            )
            # Save the results
            dev.write(args.save_path)


if __name__ == "__main__":
    main()
    gc.collect()
    cp._default_memory_pool.free_all_blocks()
    rmm.finalize()  # Finalize RMM to release resources
