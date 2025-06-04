from __future__ import annotations

import gc
import os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
import copy
import itertools
import json
import math
import pickle
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Literal

import bioframe
import pandas as pd
import pyranges
import pysam
import torch
from scipy.sparse import csc_matrix, vstack
from scipy.stats import zscore
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

from . import TFBS, footprint, motifs
from .datasets import (
    FigR_motifs_human_meme,
    FigR_motifs_mouse_meme,
    pretrained_seq_TFBS_model0,
    pretrained_seq_TFBS_model1,
)
from .genome import Genome
from .io import _get_group_atac, _get_group_atac_bw, get_bias_insertions, scPrinter
from .motifs import Motifs
from .preprocessing import export_bigwigs
from .seq import dataloader, interpretation
from .utils import *

verbose_template = "Please copy the following command in your terminal and run it to start the job"
launch_template = "Launching the following command now (no action needed from your side)"


# Set global variables so all child processes have access to the same printer dispertion model and insertion profiles
def set_global_disp_model(printer):
    globals()[printer.unique_string + "_dispModels"] = printer.dispersionModel


def set_global_insertion_profile(printer):
    globals()[printer.unique_string + "insertion_profile"] = printer.fetch_insertion_profile()


class BindingScoreAnnData:
    """
    A wrapper class for anndata.AnnData | snap.AnnData to allow for lazy loading of sites
    """

    def __init__(self, backed=False, *args, **kwargs):
        wrapper = snap.AnnData if backed else anndata.AnnData
        self.adata = wrapper(*args, **kwargs)

    def __getattr__(self, attr):
        # Delegate attribute access to self.adata, unless it's the 'adata' attribute
        return getattr(self.adata, attr)

    def __setattr__(self, attr, value):
        # Use the object's __dict__ to set 'adata' attribute directly to avoid recursion
        if attr == "adata":
            super().__setattr__(attr, value)
        else:
            setattr(self.adata, attr, value)

    def __repr__(self):
        return self.adata.__repr__()

    def fetch_site(self, identifier):
        """
        Fetch the site information from the underlying hdf5 file
        Parameters
        ----------
        identifier: str
            the identifier of the footprint region, e.g. "chr1:1294726-1295726"

        Returns
        -------
        pd.DataFrame
            a dataframe containing the site information
        """
        site_index = pd.read_hdf(str(self.uns["sites"]), "site_id").loc[identifier]

        return pd.read_hdf(str(self.uns["sites"]), "chunk_%d" % site_index["chunk"]).iloc[
            site_index["start"] : site_index["end"]
        ]


def _get_binding_score_batch(
    region_identifiers,
    cell_grouping,
    Tn5Bias,  # Numeric vector of predicted Tn5 bias for the current region
    regions,  # pyRanges object for the current region
    sites=None,  # pyRanges object. Genomic ranges of motif matches in the current region
    # Must have $score attribute which indicates how well the motif is matched (between 0 and 1)
    # If NULL, the region will be divided into equally sized tiles for TF binding scoring
    tileSize=10,  # Size of tiles if sites is NULL
    contextRadius=100,
    unique_string=None,
    downsample=1,
    strand="*",
):
    """
    Underlying function for calculating batch of binding scores. Parallelization is not done here.

    Parameters:
    region_identifiers (list[str]): List of region identifiers.
    cell_grouping (list[str] | list[list[str]] | np.ndarray): Cell groupings.
    Tn5Bias (np.ndarray): Numeric vector of predicted Tn5 bias for the current region.
    regions (pd.DataFrame): pyRanges object for the current region.
    sites (pd.DataFrame | None): pyRanges object. Genomic ranges of motif matches in the current region.
    tileSize (int): Size of tiles if sites is None. Default is 10.
    contextRadius (int): Context radius. Default is 100.
    unique_string (str | None): Unique string for accessing global variables.
    downsample (float): Downsampling factor for ATAC-seq data. Default is 1 (no downsampling).
    strand (str): Strand for motif scanning. Default is "*".

    Returns:
    list[dict]: List of dictionaries containing region identifier and corresponding binding scores.
    """

    result = []
    # Create a fake scPrinter object to pass on
    # Why we need a fake one?
    # Because these functions are created to be compatible with scPrinter centered objects
    # But it would be slow to pass the real printer project, because python would pickle everything
    # So we need a fake one to only keep the insertion_profile~
    insertion_profile = globals()[unique_string + "insertion_profile"]
    dispModels = globals()[unique_string + "_dispModels"]
    bindingscoremodel = globals()[unique_string + "_bindingscoremodel"]
    printer = scPrinter(insertion_profile=insertion_profile)
    for i, region_identifier in enumerate(region_identifiers):
        if sites is None:
            site = None
        else:
            site = sites[i]

        atac = _get_group_atac(printer, cell_grouping, regions.iloc[i])
        if downsample < 1:
            n_remove_insertions = int(np.sum(atac) * (1 - downsample))
            v = atac.data
            reduce = np.random.choice(len(v), n_remove_insertions, replace=True, p=v / np.sum(v))
            reduce, reduce_ct = np.unique(reduce, return_counts=True)
            v[reduce] -= reduce_ct
            v[v < 0] = 0

        result.append(
            TFBS.getRegionBindingScore(
                region_identifier,
                atac,
                Tn5Bias[i,],
                dftopyranges(regions.iloc[i : i + 1]),
                dispModels,
                bindingscoremodel,
                site,
                tileSize,
                contextRadius,
                strand,
            )
        )

        del atac
    return result


def get_binding_score(
    printer: scPrinter,
    cell_grouping: list[str] | list[list[str]] | np.ndarray,
    group_names: list[str] | str,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    region_width: int | None = None,
    model_key: str = "TF",
    n_jobs: int = 16,
    contextRadius: int = 100,
    save_key: str = None,
    backed: bool = True,
    overwrite: bool = False,
    motif_scanner: motifs.Motifs | None = None,
    manual_sites: list[pd.DataFrame] | None = None,
    tile: int = 10,
    open_anndata=True,
    downsample=1,
    strand="*",
    cache_factor=2,
    verbose=True,
):
    """
    Calculate the binding score for a given region and cell group, stores the result in the anndata object.
    The resulting anndata object will be linked to the printer object with the key 'save_key'.

    Parametersf
    ----------
    printer: scPrinter
        The printer object you generated by `scprinter.pp.import_fragments` | loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    group_names: list[str] | str
        The name of the group you want to visualize.
        If you want to visualize multiple groups, you can provide a list of names, e.g. `['group1', 'group2']`
    regions: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end],
        a pyranges object of the regions,
        | a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        | a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center
    region_width: int | None
        The width of the region.
        The regions you provided would be resized to this width with fixed center point of each region
    model_key: str
        The key of the model you want to use for the binding score calculation, which you specified when loading the model.
    n_jobs: int
        Number of parallel process to use
    contextRadius: int
        The radius of the context region used for the binding score calculation, default 100
    save_key: str
        The name of the anndata object you want to save the binding score result to.
    backed: bool
        Whether to save the result to a backed anndata object.
    overwrite: bool
        Whether to overwrite if there is an existing result.
    motif_scanner: motifs.Motifs | None
        The motif scanner you want to use for the binding score calculation. When passing one, the binding score will be
        calculated only on sites that have a motif matching. When passing None,
        the binding score will be calculated on tiled regions with step size 10
    manual_sites: list[pd.DataFrame] | None
        The manually specified sites you want to use for the binding score calculation.
        This is to be compatible with the R version, to reproduce the results when using motif scores as well.
    tile: int
        The tile size for the tiled region used for the binding score calculation. Default 10. For instance, a peak region
        of 1000bp, will be divided into 10bp tiles, and the binding score will be calculated on each tile. (flanking regions
        will be ignored)
    open_anndata: bool
        Whether to open the anndata object when finish calculating the binding score result. Default True. When False, it will
        save memory by lazy loading, so good for Terra and gcloud.
    downsample: float
        The fraction of the ATAC signal to be downsampled. Default 1, meaning no downsampling.
    strand: str
        The strand of the regions, default "*". When providing a motif scanner, strand is ignored, it will always based on whether the motifs
        matched on +/- strand. For tiled window calculation, "+" | "*" means only on + strand and "-" means only on - strand.
    cache_factor: int
        After how many cache_factor x n_jobs chunks calculated, we fetch the results and save them to disk is backed. Default 2.
    verbose: bool
        Whether to print the progress bar. Default True.

    Returns
    -------
    AnnData
        This AnnData can be accessed through `printer.bindingscoreadata[save_key]`
        The Anndata object stores `n_obs x n_var` matrix, where `n_obs` is the number of groups, `n_vars` is the number of regions.
        The `obs` stores the group associated properties, the `var` stores the region associated properties.
        The `obsm` stores the binding score matrix, with the key of region_identifiers such as `chr1:1000-2000`.
        The value of each `obsm` has the shape of `n_obs x n_sites`, n_sites is the number of sites in the region.
        The `uns` stores the model parameters.

    """

    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
            cell_grouping = [cell_grouping]

        regions = regionparser(regions, printer, region_width)
        if "region_name" in regions:
            region_identifiers = list(regions["region_name"])
        else:
            region_identifiers = df2regionidentifier(regions)

        if "binding score" not in printer.uns:
            printer.uns["binding score"] = {}

        if save_key is None:
            save_key = "%s_BindingScore" % model_key

        save_path = os.path.join(printer.file_path, "%s.h5ad" % save_key) if backed else "None"

        if (save_key in printer.uns["binding score"]) | os.path.exists(save_path):
            # Check duplicates...
            if not overwrite:
                print("detected %s, not allowing overwrite" % save_key)
                return
            else:
                try:
                    os.remove(
                        os.path.join(
                            printer.file_path,
                            "%s_sites.h5ad" % save_key,
                        )
                    )
                except:
                    pass
                try:
                    os.remove(save_path)
                except:
                    pass

        # results will be saved in a new key
        print("Creating %s in printer.bindingscoreadata" % save_key)
        print("obs=groups, var=regions")

        df_obs = pd.DataFrame(
            {"name": group_names, "id": np.arange(len(group_names))}, index=group_names
        )
        df_var = regions
        df_var["identifier"] = region_identifiers
        df_var.index = region_identifiers

        adata_params = {
            "backed": backed,
            "X": csr_matrix((len(group_names), len(regions))),
            "obs": df_obs,
            "var": df_var,
        }

        if backed:
            adata_params["filename"] = save_path

        adata = BindingScoreAnnData(**adata_params)
        if backed:
            adata.obs_names = [str(xx) for xx in group_names]
            adata.var_names = region_identifiers

        adata.uns["sites"] = str(os.path.join(printer.file_path, f"{save_key}_sites.h5ad"))

        # has to read in, add key and put back
        a = printer.uns["binding score"]
        a[save_key] = str(save_path)
        printer.uns["binding score"] = a
        assert model_key in printer.bindingScoreModel, (
            "%s not loaded, run printer.load_bindingscore_model first" % model_key
        )
        assert (
            printer.dispersionModel is not None
        ), "dispersion Model not loaded, run printer.load_disp_model first"

        dispersionModels = printer.dispersionModel
        bindingscoremodel = printer.bindingScoreModel[model_key]

        scales = bindingscoremodel["scales"]
        model_version = bindingscoremodel["version"]
        # Add model version to the anndata
        adata.uns["version"] = model_version
        # Only keep the related scales to avoid sending unnecessary stuff back and forth
        dispModels = [dispersionModels[str(scale)] for scale in scales]

        # global insertion_profile, dispModels, bindingscoremodel
        unique_string = printer.unique_string
        globals()[unique_string + "insertion_profile"] = printer.fetch_insertion_profile()
        globals()[unique_string + "_dispModels"] = dispModels
        globals()[unique_string + "_bindingscoremodel"] = printer.bindingScoreModel[model_key]

        cell_grouping = cell_grouping2cell_grouping_idx(printer, cell_grouping)

        # Usually when you call this funciton, you are working with a large batch of regions, so h5 mode is faster
        print("Start to get biases")
        seqBias = get_bias_insertions(printer, regions, bias_mode="h5")
        print("Finish getting biases")
        small_chunk_size = min(int(math.ceil(len(regions) / n_jobs) + 1), 100)

        pool = ProcessPoolExecutor(max_workers=n_jobs)
        p_list = []
        sites = None
        width = regions.iloc[0]["End"] - regions.iloc[0]["Start"]

        chunk_ct = 0
        index_df = []

        bar1 = trange(
            0, len(regions), small_chunk_size, desc="submitting jobs", disable=not verbose
        )
        bar = trange(
            len(region_identifiers), desc="collecting Binding prediction", disable=not verbose
        )
        time = 0
        if backed:
            adata.close()
            adata = h5py.File(
                save_path, "r+", locking=False
            )  # H5 make it faster than snapatac anndata here.
            adata_obsm = adata["obsm"]
        for i in bar1:
            if motif_scanner is not None:
                sites = motif_scanner.scan_motif(
                    regions.iloc[i : i + small_chunk_size,], clean=False, concat=False
                )
                new_sites = []
                for j, site in enumerate(sites):
                    if len(site) == 0:
                        new_sites.append(site)
                        continue
                    # site information:
                    # [bed_coord['chr'], bed_coord['start'],
                    #   bed_coord['end'], bed_coord['index'],
                    #   name, score,
                    #   strand, motif_start, motif_end
                    site = pd.DataFrame(
                        {
                            "Chromosome": site[:, 0],
                            "Start": site[:, 1].astype("int") + site[:, -2].astype("int"),
                            "End": site[:, 1].astype("int") + site[:, -1].astype("int"),
                            "Strand": site[:, -3],
                            "TF": site[:, 4],
                            "Score": site[:, 5].astype("float"),
                        }
                    )

                    relativePos = (
                        np.array(site["Start"] + site["End"]) * 0.5
                        - int(regions["Start"].iloc[i + j])
                    ).astype("int")
                    # Only keep sites with distance to CRE edge >= contextRadius
                    siteFilter = (relativePos > contextRadius) & (
                        relativePos <= (width - contextRadius)
                    )
                    site = site.iloc[siteFilter]
                    new_sites.append(site)

                sites = new_sites
                if any([len(xx) > 0 for xx in sites]):
                    pd.concat([xx for xx in sites if len(xx) > 0], axis=0).to_hdf(
                        os.path.join(
                            printer.file_path,
                            "%s_sites.h5ad" % save_key,
                        ),
                        "chunk_%d" % chunk_ct,
                        complevel=4,
                    )
                    site_index = np.cumsum([0] + [len(xx) for xx in sites])
                    index_df.append(
                        pd.DataFrame(
                            {
                                "start": site_index[:-1],
                                "end": site_index[1:],
                                "chunk": [chunk_ct] * (len(site_index) - 1),
                            },
                            index=region_identifiers[i : i + small_chunk_size],
                        )
                    )

                    chunk_ct += 1

            if manual_sites is not None:
                sites = manual_sites[i : i + small_chunk_size]
            p_list.append(
                pool.submit(
                    _get_binding_score_batch,
                    region_identifiers[i : i + small_chunk_size],
                    cell_grouping,
                    seqBias[
                        i : i + small_chunk_size,
                    ],  # Numeric vector of predicted Tn5 bias for the current region
                    regions.iloc[
                        i : i + small_chunk_size,
                    ],  # pyRanges object for the current region
                    sites,  # pyRanges object. Genomic ranges of motif matches in the current region
                    # Must have $score attribute which indicates how well the motif is matched (between 0 and 1)
                    # If NULL, the region will be divided into equally sized tiles for TF binding scoring
                    tile,  # Size of tiles if sites is NULL
                    contextRadius,
                    printer.unique_string,
                    downsample,
                    strand,
                )
            )

            if len(p_list) > n_jobs * cache_factor:
                for p in as_completed(p_list):
                    rs = p.result()
                    for r in rs:
                        if "BindingScore" not in r:
                            print(r, "error?")
                            continue
                        time += r["time"]
                        identifier = r["region_identifier"]
                        if backed:
                            adata_obsm.create_dataset(identifier, data=r["BindingScore"])
                            adata_obsm[identifier].attrs["encoding-type"] = "array"
                            adata_obsm[identifier].attrs["encoding-version"] = "0.2.0"
                        else:
                            adata.obsm[identifier] = r["BindingScore"]
                        # if motif_scanner is not None:
                        #     sites = r['sites']
                        #     sites['Chromosome'] = sites['Chromosome'].astype('str')
                        #     sites['Strand'] = sites['Strand'].astype('str')
                        #     sites['Score'] = sites['Score'].astype('float')
                        #
                    sys.stdout.flush()
                    gc.collect()
                    p_list.remove(p)
                    bar.update(len(rs))
                    del rs, p
                    if len(p_list) <= n_jobs:
                        break

        if motif_scanner is not None:
            pd.concat(index_df, axis=0).to_hdf(
                os.path.join(printer.file_path, "%s_sites.h5ad" % save_key),
                "site_id",
                mode="a",
                complevel=4,
                append=True,
            )

        for p in as_completed(p_list):
            rs = p.result()
            for r in rs:
                if "BindingScore" not in r:
                    print(r, "error?")
                    continue
                time += r["time"]
                identifier = r["region_identifier"]
                if backed:
                    adata_obsm.create_dataset(identifier, data=r["BindingScore"])
                    adata_obsm[identifier].attrs["encoding-type"] = "array"
                    adata_obsm[identifier].attrs["encoding-version"] = "0.2.0"
                    # adata.obsm[identifier] = r['BindingScore']
                else:
                    adata.obsm[identifier] = r["BindingScore"]
                # if motif_scanner is not None:
                #     sites = r['sites']
                #     sites['Chromosome'] = sites['Chromosome'].astype('str')
                #     sites['Strand'] = sites['Strand'].astype('str')
                #     sites['Score'] = sites['Score'].astype('float')
                #
            sys.stdout.flush()
            gc.collect()
            bar.update(len(rs))
            del rs
        bar.close()
        pool.shutdown(wait=True)
        print("finishes")
        if backed:
            adata.close()
            if open_anndata:
                adata = snap.read(save_path)
        printer.bindingscoreadata[save_key] = adata
        # return adata


def _get_footprint_score_onescale(
    region_identifiers,
    cell_grouping,
    Tn5Bias,  # Numeric vector of predicted Tn5 bias for the current region
    regions,  # pyRanges object for the current region
    footprintRadius,
    flankRadius,
    smoothRadius,
    mode,
    id_,
    unique_string,
    return_pval,
    bigwigmode=False,
):
    """
    Calculate the footprint score for a single scale, some regions, and all pseudobulks.

    Parameters:
    region_identifiers (list): List of region identifiers.
    cell_grouping (list): List of cell barcodes belonging to the group.
    Tn5Bias (numpy.ndarray): Numeric vector of predicted Tn5 bias for the current region.
    regions (pyranges.PyRanges): PyRanges object for the current region.
    footprintRadius (int): Radius of the footprint region.
    flankRadius (int): Radius of the flanking region (not including the footprint region).
    smoothRadius (int): Radius of the smoothing region (not including the footprint region).
    mode (int): Mode for retrieving the correct dispersion model.
    id_ (str): Identifier for the current calculation.
    unique_string (str): Unique string to access global variables.
    return_pval (bool): Whether to return the p-value for the footprint score | the z-scores.
    bigwigmode (bool): Whether to calculate footprint score from bigwig files.

    Returns:
    tuple: A tuple containing the footprint score (numpy.ndarray), the mode (int), and the id (str).
    """
    result = None
    if not bigwigmode:
        insertion_profile = globals()[unique_string + "insertion_profile"]
        printer = scPrinter(insertion_profile=insertion_profile)
    else:
        dict1 = globals()[unique_string + "group_bigwig"]

    dispModels = globals()[unique_string + "_dispModels"]

    # Create a fake scPrinter object to pass on
    # Why we need a fake one?
    # Because these functions are created to be compatible with scPrinter centered objects
    # But it would be slow to pass the real printer project, because python would pickle everything

    for i, region_identifier in enumerate(region_identifiers):
        if i % 1000 == 0 & i > 0:
            print(i)
        # assumes the cell_grouping
        if bigwigmode:
            atac = _get_group_atac_bw(dict1, cell_grouping, regions.iloc[i])
        else:
            atac = _get_group_atac(printer, cell_grouping, regions.iloc[i])
        r, mode = footprint.regionFootprintScore(
            atac,
            Tn5Bias[i],
            dispModels[str(mode)],
            footprintRadius,
            flankRadius,
            mode,
            smoothRadius,
            return_pval,
        )
        if result is None:
            result = np.zeros((len(regions), r.shape[0], r.shape[-1]))
        result[i, :, :] = r
    return result, mode, id_


def _unify_modes(modes, footprintRadius, flankRadius, smoothRadius):
    """
    Unify modes, footprintRadius, flankRadius, and smoothRadius.

    Parameters:
    modes (int | list[int] | np.ndarray | None): Modes for footprint calculation.
    footprintRadius (int | list[int] | np.ndarray | None): Radius of the footprint region.
    flankRadius (int | list[int] | np.ndarray | None): Radius of the flanking region (not including the footprint region).
    smoothRadius (int | list[int] | np.ndarray | None): Radius of the smoothing region (not including the footprint region).
    """

    if modes is None:
        modes = np.arange(2, 101)
    # we want a list of modes, but also compatible with one mode
    if type(modes) is int:
        modes = [modes]

    # If no radius provided, reuse the modes
    if footprintRadius is None:
        footprintRadius = modes
    if flankRadius is None:
        flankRadius = modes

    if smoothRadius is None:
        smoothRadius = [int(x / 2) for x in modes]

    # Again, expct to see a list, if not, create one
    if type(footprintRadius) is int:
        footprintRadius = [footprintRadius]
    if type(flankRadius) is int:
        flankRadius = [flankRadius]
    if type(smoothRadius) is int:
        smoothRadius = [smoothRadius] * len(footprintRadius)

    return modes, footprintRadius, flankRadius, smoothRadius


def get_footprint_score(
    printer: scPrinter,
    cell_grouping: list[str] | list[list[str]] | np.ndarray | "bigwig",
    group_names: list[str] | str,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    region_width: int | None = None,
    modes: (
        int | list[int] | np.ndarray | None
    ) = None,  # int | list of int. This is used for retrieving the correct dispersion model.
    footprintRadius: int | list[int] | np.ndarray | None = None,  # Radius of the footprint region
    flankRadius: (
        int | list[int] | np.ndarray | None
    ) = None,  # Radius of the flanking region (not including the footprint region)
    smoothRadius: (
        int | list[int] | np.ndarray | None
    ) = 5,  # Radius of the smoothing region (not including the footprint region)
    n_jobs: int = 16,  # Number of cores to use
    save_key: str = None,
    backed: bool = True,
    overwrite: bool = False,
    chunksize: int | None = None,
    buffer_size: int | None = None,
    return_pval: bool = True,
    collapse_barcodes: bool = False,
):
    """
    Get footprint score for a given region and cell group
    , stores the result in the anndata object.
    The resulting anndata object will be linked to the printer object with the key 'save_key'.
    New version updated to use footprint_generator as an underlying structure

    Parameters
    ----------
    printer: scPrinter
        The printer object you generated by `scprinter.pp.import_fragments` | loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    group_names: list[str] | str
        The name of the group you want to visualize.
        If you want to visualize multiple groups, you can provide a list of names, e.g. `['group1', 'group2']`
    regions: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end], a pyranges object of the regions,
        | a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        | a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center
    region_width: int | None
        The width of the region.
        The regions you provided would be resized to this width with fixed center point of each region
    modes: int | list[int] | np.ndarray
        The modes you want to calculate the footprint score for.
    footprintRadius: int | list[int] | np.ndarray | None
        The footprint radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    flankRadius: int | list[int] | np.ndarray | None
        The flanking radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    n_jobs: int
        Number of parallel process to use
    save_key: str
        The name of the anndata object you want to save the binding score result to.
    backed: bool
        Whether to save the result to a backed anndata object.
    overwrite: bool
        Whether to overwrite if there is an existing result.
    chunksize: int | None
        How many regions to be grouped into chunks for parallel processing. None would use a heuristic rule to decide chunksize.
    buffer_size: int | None
        Decides how many chunks to buffer to collect the results and write to disk. None would use a heuristic rule to decide buffer_size.
    return_pval: bool
        Whether to return the p-value for the footprint score | the z-scores. Default is True
    collapse_barcodes: bool
        Whether to collapse the cell grouping into pseudobulk (like actually make the pseudobulk bigwig) before calculating footprint score.
        It might be more efficient when dealing with a small number of groups but each group contains a lot of cells
        (so slicing sparse array is slower than making a new bigwig and read from it).


    Returns
    -------
    AnnData
        This AnnData can be accessed through `printer.footprintsadata[save_key]`
        The Anndata object stores `n_obs x n_var` matrix, where `n_obs` is the number of groups, `n_vars` is the number of regions.
        The `obs` stores the group associated properties, the `var` stores the region associated properties.
        The `obsm` stores the footprint matrix, with the key of region_identifiers such as `chr1:1000-2000`.
        The value of each `obsm` has the shape of `n_obs x n_modes x region_width`.
        The `uns` stores the model parameters.

    """
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        bigwigmode = False
        modes, footprintRadius, flankRadius, smoothRadius = _unify_modes(
            modes, footprintRadius, flankRadius, smoothRadius
        )
        if cell_grouping == "bigwig":
            bigwigmode = True
            print("footprint from bigwig mode")

        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
            if cell_grouping != "bigwig":
                cell_grouping = [cell_grouping]
        if bigwigmode:
            if any(
                [name not in printer.insertion_file.uns["group_bigwig"] for name in group_names]
            ):
                raise ValueError(f"group_names not in bigwig")
            cell_grouping = group_names

        regions = regionparser(regions, printer, region_width)
        if "region_name" in regions:
            region_identifiers = list(regions["region_name"])
        else:
            region_identifiers = df2regionidentifier(regions)
        if save_key is None:
            save_key = "FootPrints"
        save_path = os.path.join(printer.file_path, "%s.h5ad" % save_key) if backed else "None"
        if "footprints" not in printer.uns:
            printer.uns["footprints"] = {}

        if (save_key in printer.uns["footprints"]) | (os.path.exists(save_path)):
            # Check duplicates...
            if not overwrite:
                print("detected %s, not allowing overwrite" % save_key)
                return
            else:
                try:
                    printer.footprintsadata[save_key].close()
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    del printer.footprintsadata[save_key]
                except:
                    pass

        width = np.array(regions.iloc[0])
        width = width[2] - width[1]
        estimated_filesize = width * len(regions) * len(group_names) * len(modes) * 32 * 1.16e-10
        print("estimated file size: %.2f GB" % (estimated_filesize))

        # results will be saved in a new key
        print("Creating %s in printer.footprintsadata" % save_key)

        print("obs=groups, var=regions")
        df_obs = pd.DataFrame({"name": group_names, "id": np.arange(len(group_names))})
        df_obs.index = group_names
        df_var = regions
        region_identifiers = [str(x) for x in region_identifiers]
        df_var["identifier"] = region_identifiers
        df_var.index = region_identifiers
        adata_params = {
            "X": csr_matrix((len(group_names), len(regions))),
            "obs": df_obs,
            "var": df_var,
            "uns": {"scales": np.array(modes)},
        }
        if backed:
            adata_params["filename"] = save_path

        wrapper = snap.AnnData if backed else anndata.AnnData
        adata = wrapper(**adata_params)
        if backed:
            adata.obs_names = [str(xx) for xx in group_names]
            adata.var_names = region_identifiers

        a = printer.uns["footprints"]
        a[save_key] = str(save_path)
        printer.uns["footprints"] = a

        if chunksize is None:
            small_chunk_size = min(max(int(16000 / len(group_names)), 100), 10000)
        else:
            small_chunk_size = chunksize
        if buffer_size is None:
            collect_num = int(16000 / len(group_names) * 100 / small_chunk_size)
        else:
            collect_num = buffer_size
        print(small_chunk_size, collect_num)
        threadpool = ThreadPoolExecutor(max_workers=1)

        modes = list(modes)

        if backed:
            adata.close()
            adata = h5py.File(save_path, "r+", locking=False)
            adata_obsm = adata["obsm"]
        else:
            adata_obsm = adata.obsm
            # return

        def write(multimode_footprints, region_identifiers, obsm, backed):
            if backed:
                for result, id in zip(multimode_footprints, region_identifiers):
                    obsm.create_dataset(id, data=result)
                    obsm[id].attrs["encoding-type"] = "array"
                    obsm[id].attrs["encoding-version"] = "0.2.0"
            else:
                for result, regionID in zip(multimode_footprints, region_identifiers):
                    obsm[regionID] = result
            del multimode_footprints
            sys.stdout.flush()
            gc.collect()

        generator = footprint_generator(
            printer,
            cell_grouping,
            regions,
            region_width,
            modes,
            footprintRadius,
            flankRadius,
            smoothRadius,
            n_jobs,
            small_chunk_size,
            collect_num,
            return_pval=return_pval,
            bigwigmode=bigwigmode,
            collapse_barcodes=collapse_barcodes,
        )

        for multimode_footprints, regionID in generator:
            threadpool.submit(write, multimode_footprints, regionID, adata_obsm, backed)

        sys.stdout.flush()

        threadpool.shutdown(wait=True)
        if backed:
            adata.close()
            adata = snap.read(save_path)
        printer.footprintsadata[save_key] = adata
        # return adata


def resolve_overlapping_region_start_end(regions):
    """
    Resolve the overlapping regions and return the start and end index of each region. So that there is no overlaps

    Parameters
    ----------
    regions

    Returns
    -------
    region_starts, region_ends

    """
    regions = regions.copy()
    regions.columns = ["chrom", "start", "end"] + list(regions.columns[3:])
    regions = bioframe.cluster(regions)  # group regions that are overlapped
    regions["fake_id"] = np.arange(len(regions))
    regions = regions.sort_values(by=["chrom", "start", "end"])
    cluster_stats = regions.drop_duplicates("cluster")
    cluster_stats = cluster_stats.sort_values(by=["chrom", "start", "end"])
    order_of_cluster = np.array(cluster_stats["cluster"])
    aa = regions.groupby("cluster")

    region_starts = np.zeros(len(regions)).astype("int")
    region_ends = np.zeros(len(regions)).astype("int")

    for cluster in tqdm(order_of_cluster, dynamic_ncols=True):
        region_temp = aa.get_group(cluster)
        chroms, starts, ends = (
            np.array(region_temp["chrom"]),
            np.array(region_temp["start"]),
            np.array(region_temp["end"]),
        )
        mask = np.array(region_temp["fake_id"])
        if len(mask) == 1:
            # No overlapping at all
            region_starts[mask[0]] = 0
            region_ends[mask[0]] = ends[0] - starts[0]
        else:
            # Overlapping
            for i, m in enumerate(mask[:-1]):
                region_starts[m] = 0
                region_ends[m] = starts[i + 1] - starts[i]
            region_starts[mask[-1]] = 0
            region_ends[mask[-1]] = ends[-1] - starts[-1]
    return region_starts, region_ends


def export_footprint_bed3(
    printer: scPrinter,
    cell_grouping: list[str] | list[list[str]] | np.ndarray | "bigwig",
    group_names: list[str] | str,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    region_width: int | None = None,
    modes: (
        int | list[int] | np.ndarray | None
    ) = None,  # int | list of int. This is used for retrieving the correct dispersion model.
    footprintRadius: int | list[int] | np.ndarray | None = None,  # Radius of the footprint region
    flankRadius: (
        int | list[int] | np.ndarray | None
    ) = None,  # Radius of the flanking region (not including the footprint region)
    smoothRadius: (
        int | list[int] | np.ndarray | None
    ) = 5,  # Radius of the smoothing region (not including the footprint region)
    n_jobs: int = 16,  # Number of cores to use
    save_path: str = None,
    chunksize: int | None = None,
    buffer_size: int | None = None,
    footprint_save_cutoff: float = 0.1,
):
    """
    Export the multi-scale footprints to bed3 format so it can be visualized in the IGV genome browser (shoebox)

    Parameters
    ----------
    printer: scPrinter
        The printer object you generated by `scprinter.pp.import_fragments` | loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    group_names: list[str] | str
        The name of the group you want to visualize.
        If you want to visualize multiple groups, you can provide a list of names, e.g. `['group1', 'group2']`
    regions: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end], a pyranges object of the regions,
        | a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        | a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center
    region_width: int | None
        The width of the region.
        The regions you provided would be resized to this width with fixed center point of each region
    modes: int | list[int] | np.ndarray
        The modes you want to calculate the footprint score for.
    footprintRadius: int | list[int] | np.ndarray | None
        The footprint radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    flankRadius: int | list[int] | np.ndarray | None
        The flanking radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    n_jobs: int
        Number of parallel process to use
    save_path: str
        The path or the directory you want to save the bed files to.
    chunksize: int | None
        How many regions to be grouped into chunks for parallel processing. None would use a heuristic rule to decide chunksize.
    buffer_size: int | None
        Decides how many chunks to buffer to collect the results and write to disk. None would use a heuristic rule to decide buffer_size.
    footprint_save_cutoff: float
        The cutoff to save the footprint score to the bed file. Default is 0.1, mean that any footprint scores below 0.1 would be set to zero.


    Returns
    -------

    """

    with warnings.catch_warnings(), torch.no_grad():
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        warnings.simplefilter("ignore")
        modes, footprintRadius, flankRadius, smoothRadius = _unify_modes(
            modes, footprintRadius, flankRadius, smoothRadius
        )
        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
            if cell_grouping != "bigwig":
                cell_grouping = [cell_grouping]
        regions = regionparser(regions, printer, region_width)
        regions = regions.sort_values(by=list(regions.columns[:3]))
        region_width = int(regions.iloc[0]["End"] - regions.iloc[0]["Start"])
        regions_select_start, regions_select_end = resolve_overlapping_region_start_end(
            resize_bed_df(regions, region_width - 200)
        )
        regions_select_start += 100

        if chunksize is None:
            small_chunk_size = min(max(int(16000 / len(group_names)), 100), 10000)
        else:
            small_chunk_size = chunksize
        if buffer_size is None:
            collect_num = int(16000 / len(group_names) * 100 / small_chunk_size)
        else:
            collect_num = buffer_size
        print(small_chunk_size, collect_num)

        modes = list(modes)

        generator = footprint_generator(
            printer,
            cell_grouping,
            regions,
            region_width,
            modes,
            footprintRadius,
            flankRadius,
            smoothRadius,
            n_jobs,
            small_chunk_size,
            collect_num,
            async_generator=False,
            return_pval=True,
            bigwigmode=False,
            collapse_barcodes=False,
        )

    filehandle_dict = {}
    # Iterate over the groups based on provided cell_grouping and group_names
    for group_name in group_names:
        # Open the file for this pseudobulk using the name from group_names
        pseudobulk_filename = os.path.join(save_path, f"{group_name}.bed")
        f = open(pseudobulk_filename, "w")
        # Write the custom header line with the pseudobulk name
        track_line = f'#track type=shoebox name="{group_name}" viewLimits=0:3 color=0,0,255\n'
        f.write(track_line)
        sys.stdout.flush()
        filehandle_dict[group_name] = f

    # Loop through data generated from multimode footprints and region IDs
    bar = None

    threadpool = {n: ThreadPoolExecutor(max_workers=1) for n in group_names}

    def write_footprint(f, footprint_data, chrom, start, region_select_start, region_select_end):
        strs = []
        for bp_idx in range(
            region_select_start, region_select_end
        ):  # Only processing positions 100 to 899
            rounded_values = footprint_data[:, bp_idx]  # Extract values for each scale
            if any(rounded_values):  # Check if any non-zero values exist
                pos_start = start + bp_idx
                pos_end = pos_start + 1  # BED format single base pair entry
                bed_row = [chrom, pos_start, pos_end] + list(rounded_values)
                # Write row to the pseudobulk file
                strs.append("\t".join(map(str, bed_row)))
        if len(strs) > 0:
            strs = "\n".join(strs) + "\n"
            f.write(strs)
            sys.stdout.flush()

    ct = 0
    for multimode_footprints, regionIDs in generator:
        rounded_multimode_footprints = np.where(
            multimode_footprints >= footprint_save_cutoff, np.round(multimode_footprints, 3), 0
        )  # Set values below footprint_save_cutoff to zero
        for multimode_footprint, regionID in zip(rounded_multimode_footprints, regionIDs):
            if bar is None:
                bar = trange(
                    len(regions_select_start), desc="Exporting Footprints", dynamic_ncols=True
                )
            bar.update(1)
            regionID = str(regionID)
            # Process each region for chromosomal coordinates and footprints
            chrom, positions = regionID.split(":")
            start, end = map(int, positions.split("-"))
            region_select_start = regions_select_start[ct]
            region_select_end = regions_select_end[ct]
            ct += 1

            for i, group_name in enumerate(group_names):
                footprint_data = multimode_footprint[i]
                f = filehandle_dict[group_name]
                threadpool[group_name].submit(
                    write_footprint,
                    f,
                    footprint_data,
                    chrom,
                    start,
                    region_select_start,
                    region_select_end,
                )
        del multimode_footprints, regionIDs
        bar.refresh()
        gc.collect()

    func1 = partial(pysam.tabix_index, preset="bed", force=True, line_skip=1, meta_char="#")
    paths = []
    for i, group_name in enumerate(group_names):
        filehandle_dict[group_name].close()
        # Sort, compress, and index the BED file
        paths.append(os.path.join(save_path, f"{group_name}.bed"))
    process_map(func1, paths, max_workers=n_jobs, desc="Sorting Bed Files", dynamic_ncols=True)


def footprint_generator(
    printer: scPrinter,
    cell_grouping: list[str] | list[list[str]] | np.ndarray,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    region_width: int | None = None,
    modes: (
        int | list[int] | np.ndarray | None
    ) = None,  # int | list of int. This is used for retrieving the correct dispersion model.
    footprintRadius: int | list[int] | np.ndarray | None = None,  # Radius of the footprint region
    flankRadius: (
        int | list[int] | np.ndarray | None
    ) = None,  # Radius of the flanking region (not including the footprint region)
    smoothRadius: (
        int | list[int] | np.ndarray | None
    ) = 5,  # Radius of the smoothing region (not including the footprint region)
    n_jobs: int = 16,  # Number of cores to use
    chunk_size: int = 8000,  # Number of regions to process in each chunk
    buffer_size: int = 100,  # Buffer Size for processed chunks in memory
    async_generator: bool = True,  # Whether to use asyncronous processing
    verbose=True,
    return_pval: bool = True,
    bigwigmode=False,
    collapse_barcodes=False,
):
    """
    The underlying function for generating footprints. This function will return a generator that yields footprints for each region.
    The calculation of footprints will be on-going even when the calculated footprints are not fetched | downstrea procssed
    with a buffer size (collect_num) to avoid memory overflow.

    Parameters
    ----------
    printer: scPrinter
        The printer object you generated by `scprinter.pp.import_fragments` | loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    regions: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end], a pyranges object of the regions,
        | a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        | a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`,
        the transcript start site will be used as the region center
    region_width: int | None
        The width of the region.
        The regions you provided would be resized to this width with fixed center point of each region
    modes: int | list[int] | np.ndarray
        The modes you want to calculate the footprint score for.
    footprintRadius: int | list[int] | np.ndarray | None
        The footprint radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    flankRadius: int | list[int] | np.ndarray | None
        The flanking radius you want to use for the footprint score calculation.
        When passing None, it will use the same value as the modes. Default is None
    n_jobs: int
        Number of parallel process to use
    chunk_size: int
        Number of regions to process in each chunk
    buffer_size: int
        Buffer Size for the results of scales x chunks in memory. (So if buffer_size=99,
        we store one chunk of 99 scales)
    async_generator: bool
        Whether to use asyncronous processing, when True,
        the returned footprints will not keep the same order as the peaks_iter. when False,
        the returned footprints will keep the same order as the peaks_iter
    verbose: bool
        Whether to print progress information during the calculation
    return_pval: bool
        Whether to return the p-values of the footprint scores | the z-score. Default is True
    bigwigmode: bool
        Whether to read the insertion profile from a bigwig file. Default is False
    collapse_barcodes: bool
        Whether to collapse the cell grouping into pseudobulk (like actually make the pseudobulk bigwig) before calculating footprint score.
        It might be more efficient when dealing with a small number of groups but each group contains a lot of cells
        (so slicing sparse array is slower than making a new bigwig and read from it).

    Returns
    -------
    generator
        A generator that yields footprints for each chunked regions (order of the chunks can be shuffled depending on
        the param `async_generator`).


    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        unique_string = printer.unique_string

        # global insertion_profile, dispModels

        modes, footprintRadius, flankRadius, smoothRadius = _unify_modes(
            modes, footprintRadius, flankRadius, smoothRadius
        )

        if type(cell_grouping) not in [np.ndarray, list]:
            cell_grouping = [cell_grouping]

        regions = regionparser(regions, printer, region_width)
        if "region_name" in regions:
            region_identifiers = list(regions["region_name"])
        else:
            region_identifiers = df2regionidentifier(regions)

        assert (
            printer.dispersionModel is not None
        ), "dispersion Model not loaded, run printer.load_disp_model first"

        dispModels = printer.dispersionModel
        insertion_profile = printer.fetch_insertion_profile()

        if not bigwigmode:
            cell_grouping = cell_grouping2cell_grouping_idx(printer, cell_grouping)

        if collapse_barcodes:
            new_insertion_profile = {}
            new_cell_grouping = [[i] for i in range(len(cell_grouping))]
            for key in tqdm(insertion_profile, desc="collapsing_barcode"):
                xx = [
                    csc_matrix(insertion_profile[key][barcodes].sum(axis=0))
                    for barcodes in cell_grouping
                ]
                new_insertion_profile[key] = vstack(xx)
            insertion_profile = new_insertion_profile
            cell_grouping = new_cell_grouping
        if not bigwigmode:
            globals()[unique_string + "insertion_profile"] = insertion_profile
        else:
            globals()[unique_string + "group_bigwig"] = printer.uns["group_bigwig"]

        globals()[unique_string + "_dispModels"] = dispModels

        Tn5Bias = get_bias_insertions(printer, regions, bias_mode="h5")

        pool = ProcessPoolExecutor(max_workers=n_jobs)
        modes = list(modes)
        multimode_footprints_all = {}
        p_list = []

        Task_generator = itertools.product(
            range(0, len(regions), chunk_size),  # region iterator,
            zip(modes, footprintRadius, flankRadius, smoothRadius),  # footprint scale iterator)
        )

        return_order = list(range(0, len(regions), chunk_size))
        # print (return_order)
        next_return = return_order.pop(0)

        bar1 = trange(len(regions) * len(modes), desc="Submitting jobs", disable=not verbose)
        bar2 = trange(
            len(regions) * len(modes),
            desc="collecting multi-scale footprints",
            disable=not verbose,
        )

        submitted_job_num = 0
        for i, (mode, r1, r2, r3) in Task_generator:
            slice_ = slice(i, i + chunk_size)
            if i not in multimode_footprints_all:
                multimode_footprints_all[i] = [None] * len(modes)
            p_list.append(
                pool.submit(
                    _get_footprint_score_onescale,
                    region_identifiers[slice_],
                    cell_grouping,
                    Tn5Bias[slice_],
                    regions.iloc[slice_],
                    r1,
                    r2,
                    r3,
                    mode,
                    i,
                    unique_string,
                    return_pval,
                    bigwigmode,
                )
            )
            if verbose:
                bar1.update(len(Tn5Bias[slice_]))
            submitted_job_num += len(Tn5Bias[slice_])
            final_collect = submitted_job_num == len(regions) * len(modes)

            # When there are more than buffer_size jobs in the queue, | we are at the final collect (all jobs submitted)
            if (len(p_list) > buffer_size) | final_collect:
                # print ("start collecting")
                for p in as_completed(p_list):
                    results, mode, id = p.result()
                    p_list.remove(p)
                    multimode_footprints_all[id][modes.index(mode)] = results
                    if verbose:
                        bar2.update(len(results))

                    # Which chunk of region to return, if it's async, it will say whoever is ready to return we return
                    # if it's not async, we return the next chunk of region decided by next_return
                    id_for_return = id if async_generator else next_return
                    # When we fetched all scales for a region, yield the stacked results
                    if not any([xx is None for xx in multimode_footprints_all[id_for_return]]):
                        multimode_footprints = np.stack(
                            multimode_footprints_all[id_for_return], axis=2
                        )
                        slice_ = slice(id_for_return, id_for_return + chunk_size)
                        yield multimode_footprints, region_identifiers[slice_]
                        del multimode_footprints_all[id_for_return]
                        if not async_generator:
                            try:
                                next_return = return_order.pop(0)
                            except:
                                next_return = -1

                    # When half of the buffer_size jobs are done and we still have jobs to submit,
                    # break the loop and submit more jobs
                    if len(p_list) <= (buffer_size / 2) and not final_collect:
                        break
        if verbose:
            bar1.close()
            bar2.close()


def get_insertions(
    printer: scPrinter,
    cell_grouping: list[str] | list[list[str]] | np.ndarray,
    group_names: list[str] | str,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    region_width: int | None = None,
    save_key: str = None,
    backed: bool = True,
    overwrite: bool = False,
    summarize_func: callable | str = None,
):
    """
    Get the insertion profile for given regions and cell group, and potentially pass the insertions into a summarize_func.
    For instance, you can use it to compute the mean insertion profile for each cell group (e.g., summarize_func=partial(np.mean, axis=-1)).

    Parameters
    ----------
    printer : scPrinter
        The scPrinter object containing the ATAC-seq data.
    cell_grouping : list[str] | list[list[str]] | np.ndarray
        The cell group(s) for which to retrieve insertion profiles.
    group_names : list[str] | str
        The name(s) of the cell group(s).
    regions : str | Path | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic region(s) for which to retrieve insertion profiles.
    region_width : int | None, optional
        The width of the genomic regions (default is None).
    save_key : str, optional
        The key to save the insertion profiles in the AnnData object (default is None).
    backed : bool, optional
        Whether to save the AnnData object in a backed HDF5 file (default is True).
    overwrite : bool, optional
        Whether to overwrite the existing file if it already exists (default is False).
    summarize_func : callable | str, optional
        A function to summarize the insertion profiles (default is None). Can also be a string.
        Supported functions are "mean", "sum", "max", and "min".

    Returns
    -------
    This AnnData can be accessed through `printer.insertionadata[save_key]`.
    """

    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")

        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
            cell_grouping = [cell_grouping]

        if type(summarize_func) is str:
            if summarize_func == "mean":
                summarize_func = partial(np.mean, axis=-1)
            elif summarize_func == "sum":
                summarize_func = partial(np.sum, axis=-1)
            elif summarize_func == "max":
                summarize_func = partial(np.max, axis=-1)
            elif summarize_func == "min":
                summarize_func = partial(np.min, axis=-1)
            else:
                raise ValueError("summarize_func must be either mean, sum, max | min")

        regions = regionparser(regions, printer, region_width)
        if "region_name" in regions:
            region_identifiers = list(regions["region_name"])
        else:
            region_identifiers = df2regionidentifier(regions)
        save_path = os.path.join(printer.file_path, "%s.h5ad" % save_key) if backed else "None"
        if "insertions" not in printer.uns:
            printer.uns["insertions"] = {}

        if save_key is None:
            save_key = "Insertions"
        if (save_key in printer.uns["insertions"]) | (os.path.exists(save_path)):
            # Check duplicates...
            if not overwrite:
                print("detected %s, not allowing overwrite" % save_key)
                return
            else:
                try:
                    printer.insertionadata[save_key].close()
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    del printer.insertionadata[save_key]
                except:
                    pass

        width = np.array(regions.iloc[0])
        width = width[2] - width[1]

        # results will be saved in a new key
        print("Creating %s in printer.insertionadata" % save_key)

        print("obs=groups, var=regions")
        df_obs = pd.DataFrame({"name": group_names, "id": np.arange(len(group_names))})
        df_obs.index = group_names
        df_var = regions
        df_var["identifier"] = region_identifiers
        df_var.index = region_identifiers

        adata_params = {
            "X": csr_matrix((len(group_names), len(regions))),
            "obs": df_obs,
            "var": df_var,
        }
        if backed:
            adata_params["filename"] = save_path
        wrapper = snap.AnnData if backed else anndata.AnnData
        adata = wrapper(**adata_params)
        if backed:
            adata.obs_names = [str(xx) for xx in group_names]
            adata.var_names = region_identifiers

        a = printer.uns["insertions"]
        a[save_key] = str(save_path)
        printer.uns["insertions"] = a
        cell_grouping = cell_grouping2cell_grouping_idx(printer, cell_grouping)
        if backed:
            adata.close()
            adata = h5py.File(save_path, "r+", locking=False)
            adata_obsm = adata["obsm"]
        else:
            adata_obsm = adata.obsm

        for i, region_identifier in enumerate(tqdm(region_identifiers)):
            atac = _get_group_atac(printer, cell_grouping, regions.iloc[i])
            if summarize_func is not None:
                atac = summarize_func(atac)

            if backed:
                adata_obsm.create_dataset(region_identifier, data=atac)
                adata_obsm[region_identifier].attrs["encoding-type"] = "array"
                adata_obsm[region_identifier].attrs["encoding-version"] = "0.2.0"
            else:
                adata_obsm[region_identifier] = atac

        if backed:
            adata.close()
            adata = snap.read(save_path)
        printer.insertionadata[save_key] = adata


def seq_model_config(
    printer: scPrinter,
    region_path: str | Path,
    cell_grouping: list[list[str]] | list[str] | np.ndarray,
    group_names: list[str] | str,
    genome: Genome,
    fold: int,
    model_name=None,
    additional_config={},
    path_swap=None,
    config_save_path: str | Path = None,
    overwrite_bigwig=False,
):
    """
    Generate a configuration dictionary for the seq2PRINT model.

    Parameters
    ----------
    printer : scPrinter
        The scPrinter object containing the insertion profile and other relevant information.
    region_path : str | Path
        The file path to the peak / region file (from scp.peak.clean_macs2 and saved as a bed text file).
    cell_grouping : list[list[str]] | list[str] | np.ndarray
        The cell grouping for creating the model configuration.
    group_names : list[str] | str
        The names of the groups.
    genome : Genome
        The genome object.
    fold : int
        The cross-validation fold for splitting the data into train/valid/test.
    model_name : str | None, optional
        The name of the model, used to save the model weights and also collect experiments, the model weights would be saved as {model_name}_fold{fold}.pt. Default is None.
    additional_config : dict, optional
        Additional configurations for the model. Default is an empty dictionary.
    path_swap: tuple, optional
        A tuple of two strings, the first string is the string to be replaced in the path, the second string is the string to replace the first string.
        \bFor instance, if you input the region_path as "/a/b/c/peak.bed", and provide path_swap as ('/a/b/', '../)
        \ball path contains '/a/b/' would be replaced with '../'. This is useful when you generate config in one machine and launch it somewhere else.
        \bDefault is None.
    config_save_path: str | Path, optional
        The file path to save the configuration dictionary. Default is None.
    overwrite_bigwig : bool, optional
        Whether to overwrite existing insertion bigwig files. Default is False. If you create new cell_grouping but reuse the old group names, please overwrite.


    Returns
    -------
    template_json : dict
        The configuration dictionary for the sequence model.
    """

    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
        cell_grouping = [cell_grouping]
    if len(group_names) > 1:
        raise NotImplementedError("Currently only support one group at a time")
    if model_name is None and "savename" not in additional_config:
        raise ValueError("Please provide a model name | a savename in model_config")
    if "group_bigwig" not in printer.insertion_file.uns:
        printer.insertion_file.uns["group_bigwig"] = {}
    if path_swap is not None:
        before, after = path_swap
        if not before.endswith(os.sep):
            before += os.sep
    else:
        before, after = "", ""
    for name, grouping in zip(group_names, cell_grouping):
        if name in printer.insertion_file.uns["group_bigwig"]:
            if not overwrite_bigwig:
                print("bigwig for %s already exists, skip" % name)
                continue
        export_bigwigs(printer, grouping, name)

    template_json = {
        "peaks": region_path.replace(before, after),
        "signals": printer.insertion_file.uns["group_bigwig"][name].replace(before, after),
        "genome": genome.name,
        "split": genome.splits[fold],
        "max_jitter": 128,
        "reverse_compliment": True,
        "n_filters": 1024,
        "bottleneck_factor": 1,
        "no_inception": False,
        "groups": 8,
        "n_layers": 8,
        "n_inception_layers": 8,
        "inception_layers_after": True,
        "inception_version": 2,
        "activation": "gelu",
        "batch_norm_momentum": 0.1,
        "dilation_base": 1,
        "rezero": False,
        "batch_norm": True,
        "batch_size": 64,
        "head_kernel_size": 1,
        "kernel_size": 3,
        "weight_decay": 1e-3,
        "lr": 3e-3,
        "scheduler": False,
        "savename": f"{model_name}_fold{fold}",
        "amp": True,
        "ema": True,
    }
    for key in additional_config:
        template_json[key] = additional_config[key]
    if config_save_path is not None:
        with open(config_save_path, "w") as f:
            json.dump(template_json, f, indent=4, cls=NumpyEncoder)

    return template_json


def seq_lora_model_config(
    printer: scPrinter,
    region_path: str | Path,
    cell_grouping: list[list[str]] | list[str] | np.ndarray,
    group_names: list[str] | str,
    embeddings: np.ndarray | pd.DataFrame,
    genome: Genome,
    pretrain_model: str | Path,
    overwrite_barcode=False,
    model_name=None,
    fold=0,
    model_config: dict | str = {},
    additional_lora_config={},
    additional_group_attr: np.ndarray = None,
    additional_group_bias_attr: np.ndarray = None,
    path_swap=None,
    config_save_path: str | Path = None,
):
    """
    Generate a configuration dictionary for the seq2PRINT LoRA model (adapt a pretrained seq2PRINT model to various pseudobulks | single cells).

    Parameters
    ----------
    printer : scPrinter
        The scPrinter object containing the insertion profile and other relevant information.
    region_path : str | Path
        The file path to the peak file (from scp.peak.clean_macs2 and saved as a bed text file).
    cell_grouping : list[list[str]] | list[str] | np.ndarray
        The cell grouping for creating the model configuration.
    group_names : list[str] | str
        The names of the groups.
    embeddings : np.ndarray | pd.DataFrame
        The single cell embeddings for the LoRA model. If it is a numpy array, it should have the shape of (n_cells, n_features).
        \bIf it is a pandas DataFrame, the index should be the cell barcodes and the columns should be the features.
    genome : Genome
        The genome object.
    pretrain_model : str | Path
        The file path to the pretrain seq2PRINT model. Should be the xxx.pt file
    overwrite_barcode : bool, optional
        Whether to re-export and overwrite existing barcode files. Default is False. If you change the cell_grouping but keep the model_name the same, set this as True.
    model_name: str
        The name of the model. Default is None.
    fold : int, optional
        The fold for splitting the data. Default is 0.
    model_config : dict | str | Path
        The model_config you used to for the pretrained seq2PRINT model
    additional_group_attr : np.ndarray
        The additional group attributes for the LoRA model. If it is a numpy array, it should have the shape of (n_groups, n_features).
    additional_group_bias_attr: np.ndarray
        The additional group bias attributes for the LoRA model. If it is a numpy array, it should have the shape of (n_groups, n_features).
    additional_lora_config : dict, optional
        Additional configurations for the LoRA model. Default is an empty dictionary.
    path_swap: tuple, optional
        A tuple of two strings, the first string is the string to be replaced in the path, the second string is the string to replace the first string.
        \bFor instance, if you input the region_path as "/a/b/c/peak.bed", and provide path_swap as ('/a/b/', '../)
        \ball path contains '/a/b/' would be replaced with '../'. This is useful when you generate config in one machine and launch it somewhere else.
        \bDefault is None.
    config_save_path: str | Path, optional
        The file path to save the LoRA configuration dictionary. If None (default), the configuration dictionary will not be saved.

    Returns
    -------
    template_json : dict
        The configuration dictionary for the LoRA seq2PRINT model.
    """

    # load the model config if it's not a dictionary
    if type(model_config) is not dict:
        model_config = json.load(open(model_config, "r"))
    if type(cell_grouping) not in [np.ndarray, list]:
        cell_grouping = [cell_grouping]
    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
    assert len(group_names) == len(
        cell_grouping
    ), "group_names and cell_grouping must have the same length"
    if additional_group_attr is not None:
        assert len(additional_group_attr) == len(
            group_names
        ), "additional_group_attr must have the same length as group_names"
    if additional_group_bias_attr is not None:
        assert len(additional_group_bias_attr) == len(
            group_names
        ), "additional_group_bias_attr must have the same length as group_names"

    if type(embeddings) is pd.DataFrame:
        cell_all = np.concatenate([np.array(x) for x in cell_grouping])
        cell_all = set(cell_all)
        # make sure cell_all is a subset of embeddings.index
        assert (
            len(cell_all - set(embeddings.index)) == 0
        ), "cell grouping must all appear in embeddings.index"
    elif type(embeddings) is np.ndarray:
        assert len(embeddings) == len(
            printer.insertion_file.obs
        ), "embeddings must have the same length as the number of cells in the scprinter object, consider"
    if model_name is None and "savename" not in additional_lora_config:
        raise ValueError("Please provide a model name | a savename in additional_lora_config")

    # dump the insertions and genome wide coverages for single cells
    insertion_path = os.path.join(printer.file_path, "insertions.pkl")
    if not os.path.exists(insertion_path):
        insertion = {}
        for chrom in genome.chrom_sizes:
            insertion[chrom] = printer.obsm[f"insertion_{chrom}"]
        coverage_gw = [insertion[k].sum(axis=-1) for k in insertion]
        coverage_gw = np.sum(coverage_gw, axis=0)
        pickle.dump(insertion, open(insertion_path, "wb"))
        pickle.dump(coverage_gw, open(os.path.join(printer.file_path, "coverage_gw.pkl"), "wb"))
    else:
        coverage_gw = pickle.load(open(os.path.join(printer.file_path, "coverage_gw.pkl"), "rb"))

    model_name = model_name if model_name is not None else additional_lora_config["savename"]
    bc_path = os.path.join(printer.file_path, f"{model_name}_grp2barcodes.pkl")
    embed_path = os.path.join(printer.file_path, f"{model_name}_grp2embeddings.pkl")
    cov_path = os.path.join(printer.file_path, f"{model_name}_grp2covs.pkl")
    template_json = copy.deepcopy(model_config)

    default_lora_json = {
        "lora_rank": 32,
        "lora_hidden_dim": 256,
        "lora_dna_cnn": True,
        "lora_dilated_cnn": True,
        "lora_pff_cnn": True,
        "lora_profile_cnn": True,
        "lora_count_cnn": True,
        "n_lora_layers": 0,
        "mask_cov": False,
        "accumulate_grad_batches": 8,
        "dataloader_mode": "uniform",
        "cell_sample": 10,
        "coverage_in_lora": True,
        "lr": 3e-5,
        "coverage_warming": 5,
    }
    # Once upon a time, I used lr=3e-4....
    for key in default_lora_json:
        template_json[key] = default_lora_json[key]

    for key in additional_lora_config:
        template_json[key] = additional_lora_config[key]
    template_json["savename"] = model_name + f"_fold{fold}"
    if path_swap is not None:
        before, after = path_swap
        if not before.endswith(os.sep):
            before += os.sep
    else:
        before, after = "", ""
    template_json["peaks"] = region_path.replace(before, after)
    template_json["insertion"] = insertion_path.replace(before, after)
    template_json["grp2barcodes"] = bc_path.replace(before, after)
    template_json["grp2embeddings"] = embed_path.replace(before, after)
    template_json["grp2covs"] = cov_path.replace(before, after)
    template_json["pretrain_model"] = pretrain_model.replace(before, after)
    template_json["group_names"] = list(group_names)
    if (not os.path.exists(bc_path)) | overwrite_barcode:
        bc, embed, cov = [], [], []
        for barcode in cell_grouping:
            ids = printer.insertion_file.obs_ix(np.array(barcode))
            bc.append(np.sort(ids))
            if type(embeddings) is pd.DataFrame:
                embed.append(embeddings.loc[barcode].mean(axis=0))
            elif type(embeddings) is np.ndarray:
                embed.append(embeddings[ids].mean(axis=0))
            cov.append(np.log10(np.array(coverage_gw[ids]).sum()))
        bc, embed, cov = (np.array(bc, dtype="object"), np.array(embed), np.array(cov)[:, None])
        if additional_group_attr is not None:
            embed = np.concatenate([embed, additional_group_attr], axis=-1)
        if additional_group_bias_attr is not None:
            cov = np.concatenate([cov, additional_group_bias_attr], axis=-1)
        # embed = np.concatenate([embed, cov[:, None]], axis=-1)
        embed = zscore(embed, axis=0)
        cov = zscore(cov, axis=0)
        pickle.dump(bc, open(bc_path, "wb"))
        pickle.dump(embed, open(embed_path, "wb"))
        pickle.dump(cov, open(cov_path, "wb"))

    if config_save_path is not None:
        with open(config_save_path, "w") as f:
            json.dump(template_json, f, indent=4, cls=NumpyEncoder)
    return template_json


def seq_lora_slice_model_config(
    region_path: str | Path,
    lora_config: dict | str,
    pretrained_lora_model,
    model_name=None,
    finetune_group_names: list[str] | np.ndarray = [],
    fold=0,
    additional_lora_config={},
    path_swap=None,
    config_save_path: str | Path = None,
):
    """
    Generate a configuration dictionary that will slice only a subset of the cells from a lora model and further finetune the model on the subset of cells.

    Parameters
    ----------
    region_path: str | Path
        The path to the peak file
    lora_config: dict | str
        The lora configuration dictionary or the path to the configuration dictionary
    pretrained_lora_model: str | Path
        The path to the pretrained lora model
    model_name: str | None
        The name of the model, if None, it will be the same as the pretrained model
    finetune_group_names: list[str] | np.ndarray
        The group names you want to finetune on (note that these group names have to be in the group names used to train the lora model)
    fold: int
        The fold number
    additional_lora_config: dict
        Additional configurations for the lora model
    path_swap: tuple
        A tuple of two strings, the first string is the string to be replaced in the path, the second string is the string to replace the first string.
    config_save_path: str | Path
        The path to save the configuration dictionary


    Returns
    -------
    lora_config: dict
        The configuration dictionary for the finetuned lora model

    """
    if type(lora_config) is not dict:
        lora_config = json.load(open(lora_config, "r"))
    if model_name is None:
        if "savename" in additional_lora_config:
            model_name = additional_lora_config["savename"]
        else:
            model_name = lora_config["savename"]
    if path_swap is not None:
        before, after = path_swap
        if not before.endswith(os.sep):
            before += os.sep
    else:
        before, after = "", ""

    lora_config = copy.deepcopy(lora_config)
    lora_config["pretrain_lora_model"] = pretrained_lora_model.replace(before, after)
    group_names = lora_config["group_names"]
    group_names_query = {g: i for i, g in enumerate(group_names)}
    # Now query the index of finetune_cell_grouping in cell_grouping
    cells = [group_names_query[group] for group in finetune_group_names]
    lora_config["cells"] = cells
    lora_config["coverage_warming"] = 0
    for key in additional_lora_config:
        lora_config[key] = additional_lora_config[key]
    lora_config["group_names"] = finetune_group_names
    lora_config["savename"] = f"{model_name}_fold{fold}"
    lora_config["peaks"] = region_path.replace(before, after)

    if config_save_path is not None:
        with open(config_save_path, "w") as f:
            json.dump(lora_config, f, indent=4, cls=NumpyEncoder)
    return lora_config


def launch_seq2print(
    model_config_path,
    temp_dir,
    model_dir,
    data_dir,
    gpus=0,
    wandb_project=None,
    verbose=False,
    launch=False,
):
    """
    Launch the seq2print training script

    Parameters
    ----------
    model_config_path: str | Path
        The path to the model configuration dictionary, the same as the `config_save_path` when you generate the configs.
    temp_dir: str | Path
        The temporary directory to store the intermediate files
    model_dir: str | Path
         The directory to store the trained model weights
    data_dir: str | Path
        The directory that the peaks, insertions etc are saved. This will be appended to all the paths in the model configuration dictionary
    gpus: int | list[int]
        The gpus you want to use
    wandb_project: str | None
        The wandb project name to log the training process, if None, wandb will not be enabled
    verbose: bool
        The command strings will always be printed, this controls whether to print additional information.
    launch: bool
        Whether to launch the training script or just print the command string (and you can copy and paste to run it yourself)

    Returns
    -------

    """
    # file_map = {
    #     "seq2print": "seq2print_train",
    #     "lora": "seq2print_lora_train",
    #     "lora_slice": "seq2print_lora_slice_train",
    # }
    command = [
        "seq2print_train",
        "--config",
        model_config_path,
        "--temp_dir",
        temp_dir,
        "--model_dir",
        model_dir,
        "--data_dir",
        data_dir,
        "--project",
        str(wandb_project),
    ]

    if wandb_project is not None:
        command.append("--enable_wandb")

    if verbose:
        if launch:
            print(launch_template)
        else:
            print(verbose_template)
    command_str = "CUDA_VISIBLE_DEVICES=" + str(gpus) + " " + " ".join(command)
    print(command_str)
    if launch:
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpus)
            subprocess.run(command, env=env, check=True)  # Directly streams output to terminal
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")


def ohe_from_region(
    region_path,
    genome,
    model=None,
    dna_len=1840,
    signal_len=1000,
    save_path=None,
    return_summits=False,
):
    """
    Generate the ohe DNA matrices for modisco purpose

    Parameters
    ----------
    region_path: str | Path
        The path to the peak file, note that this peak file should be your CREs not the ones used to train seq2PRINT model
    genome: Genome
        The genome object
    model: str | Path | torch.nn.Module
        The pretrained model path. When provided, the dna_len and signal_len will be automatically fetched from the model
    dna_len: int
        The length of the DNA sequence
    signal_len: int
        The length of the signal sequence
    save_path: str | Path
        The path to save the ohe matrix

    Returns
    -------
    ohe: np.ndarray
        The one-hot encoded DNA matrices
    """

    summits = pd.read_table(region_path, sep="\t", header=None)
    summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
    summits["summits"] = (summits[1] + summits[2]) // 2
    summits = summits[[0, "summits"]]
    summits["summits"] = np.array(summits["summits"], dtype=int)
    if model is not None:
        if type(model) is not torch.nn.Module:
            acc_model = torch.load(model, map_location="cpu", weights_only=False)
        else:
            acc_model = model
        acc_model.eval()
        dna_len = acc_model.dna_len
        signal_len = acc_model.output_len + 200
    signal_window = signal_len
    print("signal_window", signal_window, "dna_len", dna_len)
    bias = str(genome.fetch_bias())[:-3] + ".bw"
    signals = [bias, bias]

    dataset = dataloader.seq2PRINTDataset(
        signals=signals,
        ref_seq=genome.fetch_fa(),
        summits=summits,
        DNA_window=dna_len,
        signal_window=signal_window,
        max_jitter=0,
        min_counts=None,
        max_counts=None,
        cached=False,
        reverse_compliment=False,
        device="cpu",
    )
    dataset.cache()
    ohe = dataset.cache_seqs.detach().cpu().numpy()
    if save_path is not None:
        np.savez(save_path, ohe)
    if return_summits:
        return ohe, summits
    return ohe


ohe_for_modisco = ohe_from_region


def seq_attr_seq2print(
    genome: Genome,
    region_path: str | Path,
    model_type: Literal["seq2print", "lora"],
    model_path: str | Path | list[str | Path],
    gpus: list[int] | int,
    preset: Literal["footprint", "count", None] = None,
    method="shap_hypo",
    wrapper="just_sum",
    nth_output="0-30",
    decay=0.85,
    sample=None,
    save_norm=False,
    overwrite=False,
    # lora_ids: list | list[list] = None,
    lora_config: dict | str | None = None,
    group_names: list | list[list] | str | None = None,
    save_group_names: str | list = None,
    verbose=True,
    launch=False,
    numpy_mode=False,
    save_key="deepshap",
):
    """
    Launch the sequence based attribution score calculation.

    Parameters
    ----------
    genome: Genome
        The genome object
    region_path: str | Path
        The path to the peak file
    model_type: Literal['seq2print', 'lora']
        The model type
    model_path: str | Path
        The path to the model(s)
    gpus: list[int] | int
        The gpus you want to use
    preset: Literal['footprint', 'count', None]
        The preset for the sequence attributions. Will overwrite the following parameters if provided: wrapper, nth_output, decay
    method: str
        The method for the sequence attributions (advanced usage)
    wrapper: str
        The wrapper for the sequence attributions (advanced usage)
    nth_output: str
        The nth_output for the sequence attributions (advanced usage)
    decay: float
        The decay for the sequence attributions (advanced usage)
    sample: int | None
        The number of peaks to use for the sequence attributions (downsampling the peaks to get a quick idea)
    save_norm: bool
        Whether to just save the normalization factor from the sampled summits
    overwrite: bool
        Whether to overwrite the existing files
    lora_config: dict | str | None
        The lora configuration dictionary or the path to the configuration dictionary
    group_names: list | list[list] | str | None
        The group names, will be used to save the TF binding scores and will be used to query the lora_ids
    save_group_names: str | list | None
        The save names for each group (specified by lora_ids), same length as lora_ids, if None, it will be infered from lora_ids with '-' connecting all ids. This should be the same as the one you provided to seq_tfbs_seq2print However, if you are collapsing many groups of lora id into one model, please provide a name or a list of names, or you might run into filename too long error.
    verbose: bool
        Verbosity of the command string
    launch: bool
        Whether to launch the command or just print the command string
    numpy_mode: bool
        Whether to return the numpy array of the sequence attributions instead of creating a bigwig file. This is efficient when you only have a few peaks to calculate the sequence attributions for.
    save_key: str
        A keyword for this set of sequence attributions, it should be unique to the set of regions you are using.
    Returns
    -------

    """
    if type(model_path) is list:
        for m in model_path:
            seq_attr_seq2print(
                genome=genome,
                region_path=region_path,
                model_type=model_type,
                model_path=m,
                gpus=gpus,
                preset=preset,
                method=method,
                wrapper=wrapper,
                nth_output=nth_output,
                decay=decay,
                sample=sample,
                save_norm=save_norm,
                overwrite=overwrite,
                save_group_names=save_group_names,
                lora_config=lora_config,
                group_names=group_names,
                verbose=verbose,
                launch=launch,
                numpy_mode=numpy_mode,
                save_key=save_key,
            )
        return
    genome = genome.name

    if preset is not None:
        if preset == "footprint":
            wrapper = "just_sum"
            nth_output = "0-30"
            decay = 0.85
        elif preset == "count":
            wrapper = "count"
            nth_output = "0"
            decay = 0.85
        if verbose:
            print("Using preset, the following parameters would be overwritten")
            print("using wrapper:", wrapper)
            print("using nth_output:", nth_output)
            print("using decay:", decay)

    if type(gpus) is not list:
        gpus = [gpus]
    gpus = [str(x) for x in gpus]
    gpu_num = len(gpus)

    entrance_script = "seq2print_attr"
    command = (
        [
            entrance_script,
            "--pt",
            model_path,
            "--peaks",
            region_path,
            "--method",
            method,
            "--wrapper",
            wrapper,
            "--nth_output",
            str(nth_output),
            "--gpus",
        ]
        + gpus
        + [
            "--genome",
            genome,
            "--decay",
            str(decay),
            "--save_key",
            save_key,
        ]
    )
    if overwrite:
        command.append("--overwrite")
    if numpy_mode:
        command.append("--write_numpy")
    if model_type == "lora":
        assert lora_config is not None, "lora_config must be provided when model_type is lora"
        if type(lora_config) is not dict:
            lora_config = json.load(open(lora_config, "r"))
        group_names_all = lora_config["group_names"]
        group_names_query = {str(g): i for i, g in enumerate(group_names_all)}
        # Now query the index of finetune_cell_grouping in cell_grouping
        if type(group_names) not in [np.ndarray, list]:
            group_names = [group_names]
        if type(group_names[0]) not in [np.ndarray, list]:
            group_names = [[x] for x in group_names]
        group_names = [[str(xx) for xx in x] for x in group_names]
        lora_ids = [[group_names_query[xx] for xx in group] for group in group_names]
        if type(lora_ids) is int:
            lora_ids = [[lora_ids]]
        if type(lora_ids[0]) is int:
            lora_ids = [[x] for x in lora_ids]
        lora_ids_str = ",".join(["-".join([str(y) for y in x]) for x in lora_ids])
        if save_group_names is None:
            save_group_names = ["-".join([str(y) for y in x]) for x in group_names]
        else:
            assert len(group_names) == len(
                save_group_names
            ), "group_names and save_group_names must have the same length"
        save_group_names = ",".join([str(x) for x in save_group_names])
        command.extend(["--models", lora_ids_str])
        command.extend(["--save_names", save_group_names])
    if preset is not None:
        command.extend(["--model_norm", preset])
    if sample is not None:
        command.extend(["--sample", str(sample)])
    if not verbose:
        command.append("--silent")
    if save_norm:
        command.append("--save_norm")

    if verbose:
        if launch:
            print(launch_template)
        else:
            print(verbose_template)
    print(" ".join(command))
    if launch:
        if gpu_num == 1:
            from .seq.scripts.evaluation_model import main as evaluation_model

            evaluation_model(
                pt=model_path,
                models=lora_ids_str if model_type == "lora" else None,
                genome=genome,
                peaks=region_path,
                start=0,
                end=-1,
                method=method,
                wrapper=wrapper,
                nth_output=nth_output,
                write_numpy=numpy_mode,
                gpus=gpus,
                overwrite=overwrite,
                decay=decay,
                extra="",
                model_norm=preset,
                sample=sample,
                silent=not verbose,
                save_norm=save_norm,
                save_key=save_key,
                save_names=save_group_names if model_type == "lora" else None,
            )
        else:
            try:
                subprocess.run(command, check=True)  # Directly streams output to terminal
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")

    if not save_norm:
        if model_type == "seq2print":
            return os.path.join(
                f"{model_path}_{save_key}" + (f"_sample{sample}" if sample is not None else ""),
                f"hypo.{wrapper}.{method}_{nth_output}_.{decay}.npz",
            )
        else:
            return [
                os.path.join(
                    f"{model_path}_{save_key}" + (f"_sample{sample}" if sample is not None else ""),
                    f"model_{id}.hypo.{wrapper}.{method}_{nth_output}_.{decay}.npz",
                )
                for id in save_group_names
            ]
    else:
        return os.path.join(
            f"{model_path}_{save_key}" + (f"_sample{sample}" if sample is not None else ""),
            f"norm.{wrapper}.{method}_{nth_output}_.{decay}.npy",
        )


def seq_tfbs_seq2print(
    seq_attr_count: str | list[str] | Path | list[Path] | None,
    seq_attr_footprint: str | list[str] | Path | list[Path] | None,
    genome: Genome,
    region_path: str | Path,
    gpus: list[int] | int,
    model_type: Literal["seq2print", "lora", None] = None,
    model_path: str | Path | None | list = None,
    lora_config: dict | str | None = None,
    group_names: list | str | list[list] | None = None,
    save_group_names: list | str | None = None,
    save_path: str | Path = None,
    overwrite_seqattr=False,
    post_normalize=False,
    verbose: bool = True,
    launch: bool = False,
    return_adata=False,
    save_key=None,
):
    """
    Launch the sequence based TF binding score calculation, will automatically locate the seq_attr_count and seq_attr_footprint if not provided
    Will calculate those sequence attribution scores if not provided

    Parameters
    ----------
    seq_attr_count: str | list[str] | Path | list[Path] | None
        The path to the sequence attribution scores from the count head, if None, will automatically locate the file but you need to provide the model_type and model_path
    seq_attr_footprint: str | list[str] | Path | list[Path] | None
        The path to the sequence attribution scores from the footprint head, if None, will automatically locate the file but you need to provide the model_type and model_path
    genome: Genome
        The genome object
    region_path: str | Path
        The path to the peak file
    gpus: list[int] | int
        The gpus you want to use
    model_type: Literal['seq2print', 'lora', None]
        The model type
    model_path: str | Path | None | list
        The path or list of paths to the model. When provided as a list, the sequence attribution scores are averaged across the models
    lora_config: dict | str | None
        The lora configuration dictionary or the path to the configuration dictionary, must be provided when model_type is lora
    group_names: list | str | list[list] | None
        The group names, be consistent with the ones you input when creating lora config. Note that seq2PRINT supports merging multiple groups into one model.
        If it's not a LoRA model, this should be None or a single group name, and that single TFBS file for that bulk seq2PRINT model will be saved accordingly.
    save_group_names: list | str | None
        The final group name to save the model, it wouldn't matter if you are not merging multiple group. If you are merging, for instance group 1,2,3,4,5 into a new group, perhaps naming it as CD4 etc would be easier to interpret. When set as None, we will infer one by adding '-' in between the group names, but notice that it might trigger filename too long error.
        If it's not a LoRA model, this should be None (inferred from the group_names) or a single group name, and that single TFBS file for that bulk seq2PRINT model will be saved accordingly.
    save_path: str | Path
        The path to save the TF binding scores
    overwrite_seqattr: bool
        Whether to overwrite the existing seqattr files (If you change the regions but reuse the same grouping, set this as True)
    post_normalize: bool
        Whether to normalize the seq_attr for each lora model (to further control coverage bias)
    verbose: bool
        The command strings will always be printed, this controls whether to print additional information.
    launch: bool
        Whether to launch the training script or just print the command string (and you can copy and paste to run it yourself)
    return_adata: bool
        Summarize the TFBS in an adata format that's similar to the footprints / tfbs scores
    save_key: str | None
        The keyword of the collection (used for LoRA, name the whole set). It should be unique to the set of regions you are using.

    Returns
    -------

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if lora_config is not None:
        if type(lora_config) is not dict:
            lora_config = json.load(open(lora_config, "r"))
    if type(gpus) is not list:
        gpus = [gpus]
    if type(group_names) is np.ndarray:
        group_names = list(group_names)
    if type(save_group_names) is np.ndarray:
        save_group_names = list(save_group_names)

    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
    if type(group_names[0]) not in [np.ndarray, list]:
        group_names = [[str(x)] for x in group_names]
    if save_group_names is None:
        save_group_names = [None] * len(group_names)
    if type(save_group_names) not in [np.ndarray, list]:
        save_group_names = [save_group_names]

    for i in range(len(group_names)):
        if save_group_names[i] is None:
            save_group_names[i] = "-".join(group_names[i])
    save_group_names = [str(x) for x in save_group_names]
    if return_adata:
        assert save_key is not None, "Please provide the save_key if you are using return_adata"
        assert launch is True, "Please set launch as True if you are using return_adata"
    else:
        save_key = "deepshap" if save_key is None else save_key
    if model_type == "lora":
        assert group_names is not None, "Please provide the group_names if you are using lora model"
        assert (seq_attr_count is None) and (
            seq_attr_footprint is None
        ), "Both seq_attr_count and seq_attr_footprint should be None if you are using lora model"

        # group_names_all = lora_config["group_names"]
        # group_names_query = {g: i for i, g in enumerate(group_names_all)}
        # Now query the index of finetune_cell_grouping in cell_grouping
        # lora_ids = [[group_names_query[xx] for xx in group] for group in group_names]
    else:
        # lora_ids = [[None]]
        group_names = [[None]]
    if type(seq_attr_count) in [str, Path]:
        seq_attr_count = [seq_attr_count]
    if type(seq_attr_footprint) in [str, Path]:
        seq_attr_footprint = [seq_attr_footprint]
    # todo: don't print all commands for the user to run, there are duplicated commands
    seq_attrs_all = []
    all_pass = True
    read_numpy = False
    for seq_attr, template, kind in zip(
        [seq_attr_count, seq_attr_footprint],
        ["attr.count.shap_hypo_0_.0.85.bigwig", "attr.just_sum.shap_hypo_0-30_.0.85.bigwig"],
        ["count", "footprint"],
    ):

        if seq_attr is None:
            if verbose:
                print(f"Automatic locating seq_attr_{kind}")
            assert model_type is not None, "Please provide the model_type"
            assert model_path is not None, "Please provide the model_path"
            if type(model_path) is not list:
                model_path = [model_path]
            seq_attr = []

            if overwrite_seqattr:
                for id, id_str in zip(group_names, save_group_names):
                    for model in model_path:
                        seq_attr_path = os.path.join(
                            f"{model}_{save_key}",
                            f"model_{id_str}." + template if id[0] is not None else template,
                        )
                        seq_attr_path_npz = seq_attr_path.replace(".bigwig", ".npz")
                        if os.path.exists(seq_attr_path_npz):
                            os.remove(seq_attr_path_npz)
                        if os.path.exists(seq_attr_path):
                            os.remove(seq_attr_path)

            for model in model_path:
                generate_seq_attr = []
                generate_seq_attr_save_name = []
                for id, id_str in zip(group_names, save_group_names):
                    # id_str = "-".join([str(x) for x in id])
                    seq_attr_path = os.path.join(
                        f"{model}_{save_key}",
                        f"model_{id_str}." + template if id[0] is not None else template,
                    )
                    seq_attr_path_npz = seq_attr_path.replace(".bigwig", ".npz")
                    # print (seq_attr_path, seq_attr_path_npz, "exists?")
                    if os.path.exists(seq_attr_path_npz):
                        read_numpy = True
                    if return_adata and (not os.path.exists(seq_attr_path)):
                        read_numpy = True

                    if (
                        (not os.path.exists(seq_attr_path))
                        and (not os.path.exists(seq_attr_path_npz))
                        or (read_numpy and (not os.path.exists(seq_attr_path_npz)))
                    ):
                        all_pass = False
                        generate_seq_attr.append(id)
                        generate_seq_attr_save_name.append(id_str)

                if verbose and (len(generate_seq_attr) > 0):
                    print(f"{model} seq attr file does not exist")

                if len(generate_seq_attr) > 0:
                    seq_attr_seq2print(
                        region_path=region_path,
                        model_type=model_type,
                        model_path=model,
                        genome=genome,
                        gpus=gpus,
                        preset=kind,
                        lora_config=lora_config,
                        overwrite=overwrite_seqattr,
                        verbose=verbose,
                        group_names=generate_seq_attr,
                        save_group_names=generate_seq_attr_save_name,
                        launch=launch,
                        numpy_mode=return_adata,
                        save_key=save_key,
                    )
                seq_attr.append(
                    os.path.join(
                        f"{model}_{save_key}",
                        "model_{lora_id}." + template if id[0] is not None else template,
                    )
                )
        else:
            if ".npz" in seq_attr[0]:
                read_numpy = True

        if read_numpy:
            seq_attr = [x.replace(".bigwig", ".npz") for x in seq_attr]
        seq_attrs_all.append(seq_attr)

    if launch:
        all_pass = True  # if we are launching, we assume all the files are there
    if not all_pass:
        return

    count, foot = seq_attrs_all
    # count = " ".join(count)
    # foot = " ".join(foot)
    count_pt = pretrained_seq_TFBS_model0
    foot_pt = pretrained_seq_TFBS_model1
    save_name = ",".join([os.path.join(save_path, str(x)) for x in save_group_names])
    gpus = [str(x) for x in gpus]

    command = (
        ["seq2print_tfbs", "--count_pt", count_pt, "--foot_pt", foot_pt, "--seq_count"]
        + count
        + ["--seq_foot"]
        + foot
        + [
            "--genome",
            genome.name,
            "--peaks",
            region_path,
            "--save_name",
            save_name,
            "--gpus",
        ]
        + gpus
    )
    if group_names[0][0] is not None:
        lora_ids_str = ",".join(save_group_names)
        command.extend(["--lora_ids", lora_ids_str])

    if return_adata:
        command.append("--write_numpy")

    if save_key is not None:
        save_key = os.path.join(save_path, save_key)
        command.extend(["--collection_name", save_key])

    if read_numpy:
        command.append("--read_numpy")
    if post_normalize:
        command.append("--post_normalize")
    if not verbose:
        command.append("--silent")
    if verbose:
        if launch:
            print(launch_template)
        else:
            print(verbose_template)
    print(" ".join(command))
    if launch:
        try:
            subprocess.run(command, check=True)  # Directly streams output to terminal
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")

    if return_adata:
        regions = regionparser(region_path, printer=None, width=800)

        if "region_name" in regions:
            region_identifiers = list(regions["region_name"])
        else:
            region_identifiers = df2regionidentifier(regions)
        results = np.load(f"{save_key}_TFBS.npz")["tfbs"]

        print("obs=groups, var=regions")
        lora_ids_str = save_group_names
        df_obs = pd.DataFrame(
            {"save_group_name": save_group_names, "id": lora_ids_str, "group_name": group_names}
        )
        df_obs.index = save_group_names

        df_var = regions
        df_var["identifier"] = region_identifiers
        df_var.index = region_identifiers

        adata_params = {
            "X": csr_matrix((len(group_names), len(regions))),
            "obs": df_obs,
            "var": df_var,
        }

        adata = anndata.AnnData(**adata_params)
        for i, region_identifier in enumerate(region_identifiers):
            adata.obsm[region_identifier] = results[:, i]
        return adata


def seq_denovo_seq2print(
    model_path: str | Path | list[str] | list[Path],
    region_path: str | Path,
    genome: Genome,
    gpus: list[int] | int,
    preset: Literal["footprint", "count"] = "footprint",
    n_seqlets=100000,
    modisco_window=1000,
    leiden_resolution=1,
    leiden_solver="modularity",
    save_path: str | Path = None,
    overwrite=False,
    verbose=False,
    launch=False,
    attr_key="deepshap",
):
    """
    Launch the sequence based denovo motif discovery using modisco

    Parameters
    ----------
    model_path: str | Path
        The path to the model, can be a list of paths, the sequence attribution scores will be averaged across the models
    region_path: str | Path
        The path to the peak file
    genome: Genome
        The genome object
    gpus: list[int] | int
        The gpus you want to use
    preset: Literal['footprint', 'count']
        The preset for the sequence attributions.
    n_seqlets: int
        The number of seqlets to use for modisco
    modisco_window: int
        The window size for modisco
    leiden_resolution: float
        The resolution for the leiden clustering
    leiden_solver: str
        The solver for the leiden clustering
    save_path: str | Path
        The path to save the modisco results
    overwrite: bool
        Whether to overwrite the existing files
    verbose: bool
        The command strings will always be printed, this controls whether to print additional information.
    launch: bool
        Whether to launch the training script or just print the command string (and you can copy and paste to run it yourself)
    attr_key: str
        A keyword for the set of sequence attributions to calculate modisco on, it should be unique to the set of regions you are using.

    Returns
    -------

    """
    if type(gpus) is not list:
        gpus = [gpus]
    if type(model_path) is not list:
        model_path = [model_path]

    ohe_path = f"{region_path}_ohe.npz"
    if (not os.path.exists(ohe_path)) | overwrite:
        ohe_for_modisco(region_path, genome, model=model_path[0], save_path=ohe_path)
    all_pass = True
    seq_attrs = []
    for model in model_path:
        seq_attr_path = os.path.join(
            f"{model}_{attr_key}",
            (
                "hypo.count.shap_hypo_0_.0.85.npz"
                if preset == "count"
                else "hypo.just_sum.shap_hypo_0-30_.0.85.npz"
            ),
        )
        seq_attrs.append(seq_attr_path)
        if os.path.exists(seq_attr_path):
            continue
        all_pass = False
        if verbose:
            print("Generating seq attr first")
        seq_attr_seq2print(
            region_path=region_path,
            model_type="seq2print",
            model_path=model,
            genome=genome,
            gpus=gpus,
            preset=preset,
            overwrite=overwrite,
            verbose=verbose,
            launch=launch,
            save_key=attr_key,
        )
    if launch:
        all_pass = True
    if not all_pass:
        return
    if (not os.path.exists(save_path)) | overwrite:
        modisco_helper(
            ohe=ohe_path,
            hypo=seq_attrs,
            output=save_path,
            verbose=verbose,
            n=n_seqlets,
            w=modisco_window,
            resolution=leiden_resolution,
            solver=leiden_solver,
            launch=launch,
        )


def seq_denovo_callhits(
    modisco_output: str | Path,
    model_path: str | Path | list[str] | list[Path],
    region_path: str | Path,
    region_width: int = 1000,
    device="cuda:0",
    attr_key="deepshap",
    preset: Literal["footprint", "count"] = "footprint",
    save_path: str | Path = None,
    overwrite=False,
    verbose=False,
    launch=False,
    return_hits=True,
    genome=None,
):
    """
    Call the hits from the modisco output using finemo-gpu

    Parameters
    ----------
    modisco_output: str | Path
        The path to the modisco output .h5 file
    model_path: str | Path | list[str] | list[Path]
        The path to the seq2PRINT model(s)
    region_path: str | Path
        The path to the peak file
    region_width: int
        The width of the region to use for the hits, this is used for finemo-gpu to extract the regions
    device: str
        The GPU device to use. e.g. 'cuda:0'
    attr_key: str
        The keyword for the sequence attributions. Should be the same as the one used in seq_denovo_seq2print
    preset: Literal['footprint', 'count']
        The preset for the sequence attributions. Can be 'footprint' or 'count'
    save_path: str | Path
        The path to save the hits
    overwrite: bool
        Whether to overwrite the existing files
    verbose: bool
        The command strings will always be printed, this controls whether to print additional information.
    launch: bool
        Whether to launch the training script or just print the command string (and you can copy and paste to run it yourself)
    return_hits: bool
        Whether to read the hits as a pandas dataframe. If True, the `launch` must be True
    genome: Genome
        The genome object

    Returns
    -------

    """
    ohe_path = f"{region_path}_ohe.npz"
    if not os.path.exists(ohe_path):
        ohe_for_modisco(region_path, genome, model=model_path[0], save_path=ohe_path)

    if type(model_path) is not list:
        model_path = [model_path]
    seq_attrs = []
    save_path_dir = os.path.dirname(save_path)
    print(save_path_dir)
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    for model in model_path:
        seq_attr_path = os.path.join(
            f"{model}_{attr_key}",
            (
                "hypo.count.shap_hypo_0_.0.85.npz"
                if preset == "count"
                else "hypo.just_sum.shap_hypo_0-30_.0.85.npz"
            ),
        )
        seq_attrs.append(seq_attr_path)
    seq_attr = seq_attrs[0] if len(seq_attrs) == 1 else modisco_output + ".avg.hypo.npz"
    if (not os.path.exists(f"{save_path}/hits.tsv")) | overwrite:
        command1 = [
            "finemo",
            "extract-regions-modisco-fmt",
            "-s",
            ohe_path,
            "-a",
            seq_attr,
            "-w",
            str(region_width),
            "-o",
            f"{save_path}.finemo.npz",
        ]
        command2 = [
            "finemo",
            "call-hits",
            "-M",
            "pp",
            "-r",
            f"{save_path}.finemo.npz",
            "-m",
            modisco_output,
            "-o",
            save_path,
            "-d",
            device,
        ]

        if verbose:
            if launch:
                print(launch_template)
            else:
                print(verbose_template)
        print(" ".join(command1))
        if launch:
            try:
                subprocess.run(command1, check=True)  # Directly streams output to terminal
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
        print(" ".join(command2))
        if launch:
            try:
                subprocess.run(command2, check=True)  # Directly streams output to terminal
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
    if return_hits:
        # chr     start   end     start_untrimmed end_untrimmed   motif_name      hit_coefficient hit_correlation hit_importance  strand
        #   peak_name       peak_id
        hits = pd.read_csv(f"{save_path}/hits.tsv", sep="\t")
        regions = regionparser(region_path, printer=None, width=region_width)
        region_used = regions.iloc[hits["peak_id"]]
        hits["chr"] = np.array(region_used.iloc[:, 0])
        hits["distance_to_center"] = np.abs((hits["start"] + hits["end"]) // 2 - region_width // 2)
        hits["start"] = np.array(region_used.iloc[:, 1]) + np.array(hits["start"])
        hits["end"] = np.array(region_used.iloc[:, 1]) + np.array(hits["end"])
        hits["start_untrimmed"] = np.array(region_used.iloc[:, 1]) + np.array(
            hits["start_untrimmed"]
        )
        hits["end_untrimmed"] = np.array(region_used.iloc[:, 1]) + np.array(hits["end_untrimmed"])

        return hits


def modisco_helper(
    ohe,
    hypo,
    output,
    overwrite_avg_hypo=False,
    n=1000000,
    w=1000,
    resolution=1.0,
    solver="modularity",
    verbose=False,
    launch=False,
):
    if output[-3:] != ".h5":
        output += ".h5"
    if type(hypo) is not list:
        hypo = [hypo]
    if len(hypo) > 1:
        hypo_path = output + ".avg.hypo.npz"
        if (not os.path.exists(hypo_path)) | overwrite_avg_hypo:
            new_hypo = [np.load(h)["arr_0"] for h in hypo]
            new_hypo = np.array(new_hypo).mean(axis=0)
            np.savez(hypo_path, new_hypo)
        hypo = hypo_path
    else:
        hypo = hypo[0]

    command = [
        "seq2print_modisco",
        "motifs",
        "-s",
        ohe,
        "-a",
        hypo,
        "-n",
        str(n),
        "-o",
        output,
        "-w",
        str(w),
        "-r",
        str(resolution),
        "--solver",
        solver,
    ]
    if verbose:
        if launch:
            print(launch_template)
        else:
            print(verbose_template)
    print(" ".join(command))
    if launch:
        try:
            subprocess.run(command, check=True)  # Directly streams output to terminal
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")


def delta_effects_seq2print(
    model_path: str | Path | list[str | Path],
    genome: Genome = None,
    region_path: str = None,
    motifs: Motifs | str = None,
    motif_sample_mode: Literal["argmax", "multinomial"] = "argmax",
    group_names: list | list[list] | str | None = None,
    lora_config: dict | str | None = None,
    prefix: str = "",
    sample_num=25000,
    gpus: int | list[int] | None = None,
    collapse_footprint_across_bins=False,
    random_seq=False,
    overwrite=False,
    verbose=False,
    launch=False,
    flank=200,
    vmin=-0.3,
    vmax=0.3,
    save_path=None,
    plot=True,
):
    """
    Calculate the delta effects for the seq2PRINT model by marginalizing the provided motifs on the peaks

    Parameters
    ----------
    models: str | list[str] | Path | list[Path]
        The path to the model or a list of paths to the models. If it is a list, the delta effects will be averaged across the models
    genome: Genome
        The genome object
    region_path: str
        The path to the peak file
    motifs: Motifs | str
        The motifs to marginalize, can be a scp.motifs.Motifs object or the path to the de novo motifs (.h5)
    motif_sample_mode: Literal['argmax', 'multinomial']
        The sampling mode for the motifs to get the consensus DNA sequence to plant into the peaks. Can be 'argmax' or 'multinomial'
    group_names: list | str | list[list] | None
        The group names, be consistent with the ones you input when creating lora config. Note that seq2PRINT supports merging multiple groups into one model.
    lora_config: dict | str | None
        The lora configuration dictionary or the path to the configuration dictionary, must be provided when model_type is lora
    prefix: str
        Add a prefix string to the motif names
    sample_num: int
        The number of randomly sampled peaks to use for marginalization
    gpus: int | list[int] | None
        The gpus to use for the marginalization
    collapse_footprint_across_bins: bool
        Whether to collapse the footprint across genome to get the delta effects across scales (save memory)
    random_seq: bool
        Whether to use random sequences for the marginalization instead of the cCRE background sequences
    overwrite: bool
        Whether to overwrite the existing delta effects
    verbose: bool
        The command strings will always be printed, this controls whether to print additional information.
    launch: bool
        Whether to launch the training script or just print the command string (and you can copy and paste to run it yourself)
    flank: int
        The flank size of the peaks to kept for the marginalization
    vmin: float | 'auto'
        The minimum value for the delta effects plot. When 'auto' is provided, the 5th percentile of the delta effects will be used
    vmax: float | 'auto'
        The maximum value for the delta effects plot. When 'auto' is provided, the 95th percentile of the delta effects will be used
    save_path: str | Path
        The folder to save the visualized delta effects
    plot: bool
        Whether to plot the delta effects

    Returns
    -------

    """
    names = []
    if isinstance(motifs, str):
        with h5py.File(motifs, "r") as f:
            for group in ["pos_patterns", "neg_patterns"]:
                for key in f[group].keys():
                    nn = f"{prefix}{group}.{key}"
                    names.append(nn)
    elif isinstance(motifs, Motifs):
        for motif in motifs.all_motifs:
            names.append(motif.name)
    else:
        names = list(motifs.keys())
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.exists(os.path.join(save_path, "delta_effects.npy")) and not overwrite:
        delta_effects_fp, delta_effects_count = np.load(
            os.path.join(save_path, "delta_effects.npy"), allow_pickle=True
        )
    else:
        if lora_config is not None:
            if type(lora_config) is not dict:
                lora_config = json.load(open(lora_config, "r"))
            group_names_all = lora_config["group_names"]
            group_names_query = {str(g): i for i, g in enumerate(group_names_all)}
            # Now query the index of finetune_cell_grouping in cell_grouping
            if type(group_names) is str:
                group_names = [group_names]
            if type(group_names[0]) is str:
                group_names = [[x] for x in group_names]
            lora_ids = [[group_names_query[xx] for xx in group] for group in group_names]
            if type(lora_ids) is int:
                lora_ids = [[lora_ids]]
            if type(lora_ids[0]) is int:
                lora_ids = [[x] for x in lora_ids]
        else:
            lora_ids = [[None]]

        interpretation.delta_effects.get_delta_effects(
            model_path=model_path,
            genome=genome,
            region_path=region_path,
            motifs=motifs,
            motif_sample_mode=motif_sample_mode,
            lora_ids=lora_ids,
            prefix=prefix,
            sample_num=sample_num,
            gpus=gpus,
            collapse_footprint_across_bins=collapse_footprint_across_bins,
            random_seq=random_seq,
            verbose=verbose,
            launch=launch,
            save_path=save_path,
        )
        if launch:
            delta_effects_fp, delta_effects_count = np.load(
                os.path.join(save_path, "delta_effects.npy"), allow_pickle=True
            )
        # shape (5, 1, 330, 99, 800) (5, 1, 330)
        # (#models, #ids, #motifs, #scales, #bins (optional))
        else:
            print("Please launch the delta effects calculation first")
            return
    delta_effects_fp = delta_effects_fp.mean(axis=0)  # average across models
    delta_effects_count = delta_effects_count.mean(axis=0)  # average across models

    de = {}
    for j, name in enumerate(names):
        de[name] = [delta_effects_fp[:, j], delta_effects_count[:, j]]

    if vmin == "auto" or vmax == "auto":
        vs = [de[k][0].reshape((-1)) for k in de]
        vs = np.concatenate(vs)
        cutoff = min(np.abs(np.quantile(vs, 0.05)), np.abs(np.quantile(vs, 0.95)))
        cutoff = int(cutoff * 100) / 100
        vmin = -1 * cutoff
        vmax = cutoff
        print("Using vmin/vmax:", vmin, vmax)
    if plot:
        interpretation.plot_delta_effects(
            {k: de[k][0][0] for k in de}, flank=flank, vmin=vmin, vmax=vmax, save_path=save_path
        )
    return de


def modisco_report(
    modisco_h5: str | Path | list[Path],
    save_path: str | Path,
    meme_motif: os.PathLike | Literal["human", "mouse"],
    motif_prefix: str | list[str] = "",
    delta_effect_path: str | Path | list[Path] = None,
    top_n_matches=3,
    trim_threshold=0.3,
    selected_patterns: list[str] | list[list[str]] | None = None,
):
    """
    Create the modisco report that contains the motif logos, the motif matches, the delta effects.

    Parameters
    ----------
    modisco_h5: str | Path | list[Path]
        The path to the modisco h5 file(s)
    save_path: str | Path
        The path to save the modisco report
    meme_motif: os.PathLike | Literal ['human', 'mouse']
        The path to a motif database in meme format or use the default human/mouse FigR motif database
    motif_prefix: str
        The prefix for the denovo motifs if any. This should be the same as the prefix argument passed to `scp.tl.delta_effects_seq2print`.
        When provided as a list, make it the same length as the modisco_h5
    delta_effect_path: str | Path
        The path to the delta effects file. This should be your save_path argument passed to `scp.tl.delta_effects_seq2print`
        When provided as a list, make it the same length as the modisco_h5
    top_n_matches: int
        The number of top matches to keep
    trim_threshold: float
        The threshold for trimming the motifs
    trim_min_length: int
        The minimum length for trimming the motifs
    selected_patterns: str | list[str] | None
        The selected patterns to plot. If None, all patterns will be plotted. If a list of strings, only the patterns in the list will be plotted. e.g. ['pos_patterns.pattern_60', 'pos_patterns.pattern_4', 'pos_patterns.pattern_2', 'pos_patterns.pattern_27']. When providing a list of modisco_h5, provide a list of lists that is the same length as the modisco_h5
    Returns
    -------


    """
    output_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if type(meme_motif) is str:
        if meme_motif == "human":
            meme_motif = (FigR_motifs_human_meme,)
        elif meme_motif == "mouse":
            meme_motif = FigR_motifs_mouse_meme

    if type(modisco_h5) is not list:
        modisco_h5 = [modisco_h5]
    if type(delta_effect_path) is not list:
        delta_effect_path = [delta_effect_path]
    if type(motif_prefix) is str:
        motif_prefix = [motif_prefix]
    if selected_patterns is None:
        selected_patterns = [None] * len(modisco_h5)
    elif type(selected_patterns[0]) is str:
        selected_patterns = [selected_patterns]
    img_path_suffixs = [os.path.join(output_dir, f"{h5}_figs") for h5 in modisco_h5]
    for img_path_suffix in img_path_suffixs:
        if not os.path.exists(img_path_suffix):
            os.makedirs(img_path_suffix)
    return interpretation.modisco_report.report_motifs(
        modisco_h5,
        output_dir,
        save_name,
        img_path_suffixs,
        meme_motif,
        top_n_matches,
        delta_effect_path,
        motif_prefix,
        trim_threshold,
        selected_patterns,
    )


def seq2print_register_norm_factor(model_path, count_hypo, footprint_hypo):
    """
    Manually register the normalization factor for the seq2PRINT model, this is useful when your training process failed at the last part but can still manually rerun the normalization factor process.

    Parameters
    ----------
    model_path
    count_hypo
    footprint_hypo

    Returns
    -------

    """
    acc_model = torch.load(model_path, map_location="cpu", weights_only=False)
    if type(count_hypo) in [tuple, list]:
        count_hypo = count_hypo
    else:
        count_hypo = np.load(count_hypo)
    if type(footprint_hypo) in [tuple, list]:
        foot_hypo = footprint_hypo
    else:
        foot_hypo = np.load(footprint_hypo)
    low, median, high = count_hypo
    print("count head normalization factor", low, median, high)
    acc_model.count_norm = [low, median, high]
    low, median, high = foot_hypo
    print("footprint head normalization factor", low, median, high)
    acc_model.foot_norm = [low, median, high]
    torch.save(acc_model, model_path)
