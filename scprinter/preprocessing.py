from __future__ import annotations

import multiprocessing
import os.path
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from anndata import AnnData
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, hstack, issparse, vstack
from tqdm.auto import tqdm, trange

from .genome import Genome
from .io import load_printer, scPrinter
from .peak import *
from .shift_detection import detect_shift
from .utils import *


def import_data(
    path,
    barcode,
    genome,
    plus_shift,
    minus_shift,
    tempname,
    close_after_import=True,
    to_csc=True,
    **kwargs,
):
    """
    Imports data from a specified path, applies whitelist,
    and shifts the fragments if necessary.

    Parameters:
    -----------
    path: str
        The path to the input fragments file.
    barcode: list[str]
        The whitelist of barcodes to be included in the data.
    genome: Genome
        The scp.genome.Genome object that contains all the necessary information for the genome.
    plus_shift: int
        The extra shift for the left end of the fragment.
    minus_shift: int
        The extra shift for the right end of the fragment.
    tempname: str
        The temporary name for the output file.
    close_after_import: bool
        Whether to close the AnnData object after importing the data. Default is True.
    to_csc: bool
        Whether to convert the data to CSC format. Default is True.
    kwargs: dict
        Additional keyword arguments to be passed to the snapatac2.pp.import_data function.

    Returns:
    -------
    str: The temporary name of the output file.

    """

    if ".gz" not in path:
        if os.path.exists(path + ".gz"):
            print("using gzipped file: %s.gz" % path)
            path = path + ".gz"
        else:
            print("gzipping %s, because currently the backend requires gzipped file" % path)
            os.system("gzip %s" % path)
            path = path + ".gz"

    # this is equivalent to do +4/-4 shift,
    # because python is 0-based, and I'll just use the right end as index
    # meaning that I'll do [end, end+1) as insertion position, not [end-1, end)
    extra_plus_shift = 4 - plus_shift
    extra_minus_shift = -5 - minus_shift

    # Import data from the fragments file.
    # shift_left and shift_right are set all to zeros now, the extra shifting now happens during the frags to insertion part.
    data = snap.pp.import_data(
        path,
        file=tempname,
        whitelist=barcode,
        chrom_sizes=genome.chrom_sizes,
        shift_left=0,
        shift_right=0,
        **kwargs,
    )
    data.obs["frag_path"] = [path] * len(data.obs_names)
    frag_sample_name = "_._".join(tempname.split("_._")[1:])
    data.obs["frag_sample_name"] = [frag_sample_name] * len(data.obs_names)
    snap.metrics.tsse(data, genome.fetch_gff())
    data = frags_to_insertions(
        data,
        split=True,
        extra_plus_shift=extra_plus_shift,
        extra_minus_shift=extra_minus_shift,
        to_csc=to_csc,
    )
    if close_after_import:
        data.close()
        return tempname
    else:
        return data


def import_fragments(
    path_to_frags: str | list[str] | Path | list[Path],
    barcodes: list[str] | list[list[str]] | Path | list[Path],
    savename: str | Path,
    genome: Genome,
    sample_names: str | list[str] | None = None,
    plus_shift: int = 4,
    minus_shift: int = -5,
    auto_detect_shift: bool = True,
    unique_string: str | None = None,
    n_jobs: int = 20,
    **kwargs,
):
    """
    Import ATAC fragments into single cell genome-wide insertion profile backed in anndata format and create the scPrinter obejct.

    Parameters
    ----------
    path_to_frags: str | list[str] | Path | list[Path]
        Path or List of paths to the fragment files. When multiple files are provided,
        they will be separately imported and concatenated.
    barcodes: list[str] | list[list[str]] | Path | list[Path]
        List of barcodes to be whitelisted. If None, all barcodes will be used. When provide must be the same length as path_to_frags.
        If you input one fragments file, the barcodes can either be a list of barcodes, or the path to a barcode file
        where each line is one barcode to include.
        If you input multiple fragments files, the barcodes should be a list of list of barcodes, or a list of paths.
        If you provide sample_names as well, the barcode here should be the one without appending the sample name (the raw barcode presentede in the fragments).
        Note: if you provide a whitelist, each fragment file needs to have at least one barcode that got imported.
    savename: str | Path
        Path to save the scPrinter anndata object
    genome: Genome
        Genome object that contains all the necessary information for the genome
    sample_names: str | list[str] | None
        The name of the samples. Recommend to use when inputting multiple fragments files, and there can be collision of barcode names.
        When provided, the barcode will be appended with the sample name.
    plus_shift: int
        The shift **you have done** for the left end of the fragment. Default is 4,
        which is what the standard pipeline does
    minus_shift: int
        The shift **you have done** for the right end of the fragment. Default is -5,
        which is what the standard pipeline does (terra SHARE-seq pipeline now does -4, but chromap is -5)
    auto_detect_shift: bool
        Whether to automatically detect the shift. Default is True. It will overwrite the plus_shift and minus_shift you provided.
    unique_string: str
        A unique string. If None, a random UUID will be used. It is used to create a unique name for global variables.
        Most of the time you don't need to set it.
    n_jobs: int
        Number of workers for parallel processing. Default is 20. Only used when multiple fragments files are provided.
    kwargs
        Other arguments passed to snapatac2.pp.import_data

    Returns
    -------

    """
    if unique_string is None:
        unique_string = str(uuid.uuid4())
    # hoping for a list, but if it's a string, treated it as a list
    if type(path_to_frags) is str:
        pathsToFrags = [path_to_frags]
        barcodes = [barcodes]
    else:
        pathsToFrags = path_to_frags

    if len(pathsToFrags) > 1:
        if sample_names is None:
            print(
                "Multiple fragments files detected, it is suggested to provide sample names to avoid barcode collision"
            )
        else:
            if len(sample_names) != len(pathsToFrags):
                raise ValueError("sample_names should have the same length as pathsToFrags")

    # No longer needed since now snapATAC2 won't handle the frags -> insertions
    # # this function check is snapATAC2 fix the insertion in the future
    # flag_ = check_snap_insertion()
    # print("snapatac2 shift check", flag_)
    # if not flag_:
    #     flag_ = check_snap_insertion(0, 1)
    #     if not flag_:
    #         print("raise an Issue please")
    #         raise EOFError
    #     else:
    #         extra_minus_shift += 1

    # these are historical_kwargs that snapatac2 takes, but not anymore
    for historical_kwarg in ["min_tsse", "low_memory"]:
        if historical_kwarg in kwargs:
            del kwargs[historical_kwarg]

    if len(pathsToFrags) == 1:
        print("single input")
        path = pathsToFrags[0]

        if auto_detect_shift:
            plus_shift, minus_shift = detect_shift(path, genome)
            print(
                "You are now using the beta auto_detect_shift function, this overwrites the plus_shift and minus_shift you provided"
            )
            print(
                "If you believe the auto_detect_shift is wrong, please set auto_detect_shift=False"
            )
            print("detected plus_shift and minus_shift are", plus_shift, minus_shift, "for", path)
        time.sleep(5)
        data = import_data(
            path,
            barcodes[0],
            genome,
            plus_shift,
            minus_shift,
            savename,
            close_after_import=False,
            **kwargs,
        )
    else:
        # with multiple fragments, store them in memory and concat
        # Should be able to support snapatac2.anndataset in the future, but, let's keep it this way for now
        adatas = []
        p_list = []
        pool = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")
        )

        ct = 0
        if auto_detect_shift:
            print(
                "You are now using the beta auto_detect_shift function, this overwrites the plus_shift and minus_shift you provided"
            )
            print(
                "If you believe the auto_detect_shift is wrong, please set auto_detect_shift=False"
            )
        bar = trange(len(pathsToFrags), desc="Importing fragments")
        for path, barcode in zip(pathsToFrags, barcodes):
            if auto_detect_shift:
                plus_shift, minus_shift = detect_shift(path, genome)
                print(
                    "detected plus_shift and minus_shift are", plus_shift, minus_shift, "for", path
                )
            post_tag = f"_._part{ct}" if sample_names is None else f"_._{sample_names[ct]}"
            p_list.append(
                pool.submit(
                    import_data,
                    path,
                    barcode,
                    genome,
                    plus_shift,
                    minus_shift,
                    savename + post_tag,
                    True,
                    False,
                    **kwargs,
                )
            )
            ct += 1
            bar.update(1)
            bar.refresh()
        temp_files = []
        for p in tqdm(as_completed(p_list), total=len(p_list)):
            savepath = p.result()
            temp_files.append(savepath)
            adatas.append(("_._".join(savepath.split("_._")[1:]), savepath))
            # adatas.append(anndata.read_h5ad(savepath))
            sys.stdout.flush()

        data = snap.AnnDataSet(adatas=adatas, filename=savename + "_temp")
        # Now transfer the insertions to the final AnnData object
        data2 = snap.AnnData(filename=savename)
        data2.obs = data.obs[:]
        data2.obs_names = data.obs_names
        # if provided sample_names, add it to the obs_names
        if sample_names is not None:
            data2.obs_names = [
                f"{sample_name}_{barcode}"
                for sample_name, barcode in zip(data.obs["sample"], data.obs_names)
            ]

        print("start transferring insertions")
        for key in data.adatas.obsm.keys():
            if "insertion" not in key:
                continue
            insertions = data.adatas.obsm[key]
            data2.obsm[key] = insertions.tocsc()
        dfs = data.adatas.obs[:]
        for key in dfs:
            data2.obs[key.name] = dfs[key.name]

        data2.uns["reference_sequences"] = data.uns["reference_sequences"]
        data.close()
        data2.close()

        for i in range(ct):
            os.remove(temp_files[i])
        os.remove(savename + "_temp")
        data = snap.read(savename)
    data.uns["genome"] = genome.name
    data.uns["unique_string"] = unique_string

    data.close()

    return load_printer(savename, genome)


def mean_norm_counts_array(data):
    """
    Normalize the count matrix by dividing each cell by its mean count across all peaks.

    Parameters
    ----------
    data : np.ndarray
        The count matrix to be normalized.

    Returns
    -------
    The normalized count matrix.
    """
    sparse_input = issparse(data)
    data = data.astype("float32")
    _row_mean = data.mean(axis=1)
    row_mean = np.array(_row_mean).ravel().astype(np.float32) + 0.00001
    if sparse_input:
        data.data = data.data / np.repeat(row_mean, data.getnnz(axis=1))
    else:
        data = data / row_mean[:, None]
    return data


def mean_norm_counts(adata):
    """
    Normalize the count matrix by dividing each cell by its mean count across all peaks.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the count matrix to be normalized.

    Returns
    -------
    The normalization is performed in place, modifying the input AnnData object.
    """
    # xx = csr_matrix(adata.X / adata.X.mean(axis=1))
    # adata.X = xx

    adata.X = mean_norm_counts_array(adata.X)

    return adata


def make_gene_matrix(
    printer: scPrinter,
    strategy: Literal["gene", "promoter", "tss"] = "gene",
    cell_grouping: list[list[str]] | list[str] | np.ndarray | None = None,
    group_names: list[str] | str | None = None,
    sparse=True,
):
    """
    Generate a gene matrix based on the given strategy.

    Parameters
    ----------
    printer : scPrinter
        The scPrinter object containing the insertion profile and other relevant information.
    strategy : Literal["gene", "promoter", "tss"], optional
        The strategy to generate the gene matrix. Default is "gene".
    cell_grouping : list[list[str]] | list[str] | np.ndarray | None, optional
        The cell grouping for creating the gene matrix. Default is None.
    group_names : list[str] | str | None, optional
        The names of the groups. Default is None.
    sparse : bool, optional
        Whether to use sparse matrices for storing the gene matrix. Default is True.

    Returns
    -------
    adata : AnnData
        The gene matrix in AnnData format.
    """

    if strategy == "gene":
        # Fetch all genes from the database
        db = printer.gff_db
        info = []
        for gene in db.features_of_type("gene"):
            if gene.chrom not in printer.genome.chrom_sizes:
                continue
            info.append([gene.chrom, gene.start, gene.end, gene.id])
        regions = pd.DataFrame(info, columns=["chrom", "start", "end", "gene_id"])
        adata = make_peak_matrix(
            printer, regions, cell_grouping=cell_grouping, group_names=group_names, sparse=sparse
        )

        adata.var.index = list(regions["gene_id"])
        return adata
    elif strategy == "tss":
        tss_list = []
        db = printer.gff_db
        for transcript in db.features_of_type("transcript"):
            # Get gene name or fallback to gene_id
            gene_name = transcript.attributes.get(
                "gene_name", transcript.attributes.get("gene_id", ["unknown"])
            )[0]
            if transcript.chrom not in printer.genome.chrom_sizes:
                continue
            # Infer TSS position based on strand
            if transcript.strand == "+":
                tss = transcript.start
            else:
                tss = transcript.end
            tss_list.append(
                (
                    transcript.chrom,
                    max(tss - 10000, 0),
                    min(tss + 10000, printer.genome.chrom_sizes[transcript.chrom] - 1),
                    gene_name,
                )
            )
        regions = pd.DataFrame(tss_list, columns=["chrom", "start", "end", "gene_id"])
        adata = make_peak_matrix(
            printer, regions, cell_grouping=cell_grouping, group_names=group_names, sparse=sparse
        )
        adata.var["gene_id"] = list(regions["gene_id"])
        gene_groups = adata.var.groupby("gene_id").indices
        X = adata.X
        merged_X = []

        new_gene_ids = []

        for gene_id, idx_list in tqdm(
            gene_groups.items(), desc="Merging transcripts with the same gene name"
        ):
            cols = X[:, idx_list]
            if issparse(cols):
                mean_col = cols.sum(axis=1)
            else:
                mean_col = np.sum(cols, axis=1)
            merged_X.append(csc_matrix(mean_col))
            new_gene_ids.append(gene_id)

        # Stack and create new AnnData
        if issparse(X):
            merged_matrix = hstack(merged_X).tocsr()
        else:
            merged_matrix = np.column_stack(merged_X)
        adata_merged = anndata.AnnData(X=merged_matrix, obs=adata.obs.copy())
        adata_merged.var["gene_id"] = new_gene_ids
        adata_merged.var_names = new_gene_ids

        return adata_merged
    else:
        print("Sorry, the current implementation only supports 'gene' strategy.")


def make_peak_matrix(
    printer: scPrinter,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    region_width: int | None = None,
    cell_grouping: list[list[str]] | list[str] | np.ndarray | None = None,
    group_names: list[str] | str | None = None,
    sparse=True,
):
    """
    Generate a peak matrix for the given regions.

    Parameters
    ----------
    printer : scPrinter
        The scPrinter object containing the insertion profile and other relevant information.
    regions : str | Path | pd.DataFrame | pyranges.PyRanges | list[str]
        The regions for which the peak matrix needs to be generated. It can be a file path, a DataFrame,
        a PyRanges object, or a list of region identifiers.
    region_width : int | None, optional
        The width of the regions. If None, the width will be determined based on the input regions.
    cell_grouping : list[list[str]] | list[str] | np.ndarray | None, optional
        The cell grouping for creating the peak matrix. If None, the peak matrix will be created at single
        cell resolution.
    group_names : list[str] | str | None, optional
        The names of the groups. If None, the group names will be determined based on the cell grouping.
    sparse : bool, optional
        Whether to use sparse matrices for storing the peak matrix. Default is True.

    Returns
    -------
    adata : AnnData
        The peak matrix in AnnData format.
    """

    # Parse the regions to a bed DataFrame
    regions = regionparser(regions, printer, region_width)
    region_identifiers = df2regionidentifier(regions)

    # Get the insertion profiles
    insertion_profile = printer.fetch_insertion_profile()
    res = []

    # depending on whether to use sparse matrices, we use different sparse matrix classes
    # COO first, because it makes vstack more efficient
    # Then CSC later, because it makes hstack more efficient
    wrapper1 = coo_matrix if sparse else np.array
    wrapper2 = csc_matrix if sparse else np.array

    if cell_grouping is not None:
        cell_grouping_idx = cell_grouping2cell_grouping_idx(printer, cell_grouping)
        if type(group_names) not in [np.ndarray, list, None]:
            group_names = [group_names]
            cell_grouping = [cell_grouping]

    for i, (chrom, start, end) in enumerate(
        zip(
            tqdm(regions.iloc[:, 0], desc="Making peak matrix"),
            regions.iloc[:, 1],
            regions.iloc[:, 2],
        )
    ):
        # First create single cell resolution -> CSC matrix for each region
        v = wrapper2(insertion_profile[chrom][:, start:end].sum(axis=-1), dtype="uint16")
        res.append(v)
    # We now have a cell x region peak matrix
    res = hstack(res).tocsr() if sparse else np.hstack(res)
    # Now merge them into pseudobulk x region AnnData object
    if cell_grouping is not None:
        a = [wrapper1(np.array(res[barcodes].sum(axis=0))) for barcodes in cell_grouping_idx]
        res = wrapper2(vstack(a))
        res = res.tocsr() if sparse else res

    adata = AnnData(X=res)
    adata.obs.index = printer.insertion_file.obs_names[:] if cell_grouping is None else group_names
    adata.var.index = region_identifiers
    return adata


"""
These are both historical names for the same function. My bad...
"""


def collapse_barcodes(*args, **kwargs):
    sync_footprints(*args, **kwargs)


def sync_footprints(*args, **kwargs):
    export_bigwigs(*args, **kwargs)


def export_bigwigs(
    printer: scPrinter,
    cell_grouping: list[list[str]] | list[str] | np.ndarray,
    group_names: list[str] | str,
    smooth_window=0,
    resolution=1,
):
    """
    Generate bigwig files for each group which can be used for synchronized footprint visualization

    Parameters
    ----------
    printer: scPrinter object
        The printer object you generated by `scprinter.pp.import_fragments` or loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    group_names: list[str] | str
        The name of the group you want to visualize.
        If you want to visualize multiple groups, you can provide a list of names, e.g. `['group1', 'group2']`
    smooth_window: int
        The smooth window for the bigwig file. Default is 0.
    resolution: int
        The resolution for the bigwig file. Default is 1.
    Returns
    -------

    """
    smooth_window = smooth_window // 2
    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
        cell_grouping = [cell_grouping]

    cell_grouping = cell_grouping2cell_grouping_idx(printer, cell_grouping)
    insertion_profile = printer.fetch_insertion_profile()
    chrom_list = list(insertion_profile.keys())
    chunksize = 1000000

    a = (
        printer.insertion_file.uns["group_bigwig"]
        if "group_bigwig" in printer.insertion_file.uns
        else {}
    )
    a["bias"] = printer.insertion_file.uns["bias_bw"]

    for name, grouping in zip(group_names, cell_grouping):
        print("Creating bigwig for %s" % name)

        path = os.path.join(printer.file_path, "%s.bw" % name)

        bw = pyBigWig.open(path, "w")
        header = []
        for chrom in chrom_list:
            sig = insertion_profile[chrom]
            length = sig.shape[-1]
            header.append((chrom, length))
        bw.addHeader(header, maxZooms=10)
        for chrom in tqdm(chrom_list):
            sig = insertion_profile[chrom]

            for i in range(0, sig.shape[-1], chunksize):
                temp_sig = sig[
                    :,
                    slice(
                        max(i - smooth_window, 0), min(i + chunksize + smooth_window, sig.shape[-1])
                    ),
                ]
                left_pad = 0 if i <= smooth_window else smooth_window
                if temp_sig.nnz == 0:
                    continue
                pseudo_bulk = np.array(temp_sig[grouping].sum(axis=0)).reshape((-1))
                pseudo_bulk = (
                    rz_conv(pseudo_bulk, smooth_window)[left_pad : left_pad + chunksize]
                    if smooth_window > 0
                    else pseudo_bulk
                )
                if resolution > 1:
                    pseudo_bulk = pseudo_bulk[::resolution]  # export every resolution-th base

                pseudo_bulk = coo_matrix(pseudo_bulk)
                if len(pseudo_bulk.data) == 0:
                    continue

                col, data = pseudo_bulk.col, pseudo_bulk.data
                indx = np.argsort(col)

                bw.addEntries(
                    str(chrom),
                    (col[indx] * resolution) + i,
                    values=data[indx].astype("float"),
                    span=resolution,
                )
        bw.close()
        a[str(name)] = str(path)

    printer.insertion_file.uns["group_bigwig"] = a


def create_frag_group(
    temp_path: str | Path,
    frag_file: str | list[str] | Path | list[Path],
    cell_grouping: list[str] | np.ndarray,
    group_name: str,
    sample_names,
):
    """
    Create a whitelist file for a specified group of cells and filter the fragments file based on the whitelist.

    Parameters
    ----------
    temp_path: str | Path
        The path to the temporary directory.
    frag_file: str | list[str] | Path | list[Path]
        The path to the fragments file or a list of paths to the fragments files.
    cell_grouping: list[str] | np.ndarray
        The list of cell barcodes for the group.
    group_name: str
        The name of the group.

    Returns
    -------

    """

    if type(frag_file) is not list:
        frag_file = [frag_file]
    if sample_names is None:
        sample_names = [None] * len(frag_file)
    if type(sample_names) is not list:
        sample_names = [sample_names]

    bcs = set(np.sort(np.unique(cell_grouping)))
    filtered_frag_file = os.path.join(temp_path, f"{group_name}_filtered_frag.tsv.gz")
    for frag, sample_name in zip(frag_file, sample_names):
        reader = pd.read_csv(frag, sep="\t", header=None, chunksize=100000, comment="#")

        for chunk in reader:
            chunk_bc = (
                chunk[3] if sample_name is None else [sample_name + "_" + xx for xx in chunk[3]]
            )
            boolean_mask = [xx in bcs for xx in chunk_bc]
            chunk = chunk[boolean_mask]
            if len(chunk) > 0:
                chunk.to_csv(filtered_frag_file, sep="\t", header=False, index=False, mode="a")

    # if type(frag_file) is list:
    #     frag_file = " ".join(frag_file)
    #
    # with open(os.path.join(temp_path, f"{group_name}_whitelist.txt"), "w") as f:
    #     for w in bcs:
    #         f.write(w + "\n")
    # whitelist_file = os.path.join(temp_path, f"{group_name}_whitelist.txt")
    #
    # command = f"zcat {frag_file} | awk -v OFS='\t' 'NR==FNR{{a[$1]; next}} ($4 in a)' {whitelist_file} - | gzip > {filtered_frag_file}"
    # print(command)
    # # Execute the command using subprocess.run
    # subprocess.run(command, shell=True, check=True)


def call_peak_one_group(
    file_path, frag_file, grouping, name, preset, clean_temp=True, sample_names=None
):
    if grouping is not None:
        create_frag_group(file_path, frag_file, grouping, name, sample_names)
    macs2(
        (
            os.path.join(file_path, f"{name}_filtered_frag.tsv.gz")
            if grouping is not None
            else frag_file
        ),
        name,
        os.path.join(file_path, "macs2"),
        p_cutoff=0.01 if preset == "seq2PRINT" else None,
    )
    if clean_temp:
        if grouping is not None:
            os.remove(os.path.join(file_path, f"{name}_filtered_frag.tsv.gz"))


def call_peaks(
    printer: scPrinter,
    frag_file: str | list[str] | Path | list[Path],
    cell_grouping: list[list[str]] | list[str],
    group_names: list[str] | str,
    iterative_peak_merging=True,
    merge_across_groups=False,
    peak_width=1000,
    clean_temp=True,
    preset: Literal["seq2PRINT", "chromvar", None] = None,
    n_jobs=20,
    overwrite=False,
    sample_names=None,
    **kwargs,
):
    """
    Calling peaks for the specified groups of cells and store them in the printer object.

    Parameters
    ----------
    printer: scPrinter
        The scPrinter object containing the insertion profile and other relevant information.
    frag_file: str | list[str] | Path | list[Path]
        The path to the fragments file or a list of paths to the fragments files.
    cell_grouping: list[list[str]] | list[str]
        The list of cell barcodes for group definitions.
    group_names: list[str] | str
        The names of the groups.
    iterative_peak_merging: bool
        Whether to perform iterative peak merging. Default is True.
    merge_across_groups: bool
        Whether to merge peaks across groups, or do iterative peak merging for each group. Default is False.
    peak_width: int
        The width of the peaks. Default is 1000.
    clean_temp: bool
        Whether to clean temporary files. Default is True.
    preset: Literal["seq2PRINT", "chromvar", None]
        The preset for peak calling. Default is None. preset includes 'seq2PRINT' and 'chromvar'. The former is better for training the seq2PRINT model, and the latter is better for general cellxpeak analysis.
    n_jobs: int
        Number of workers for parallel processing. Default is 20.
    overwrite: bool
        Overwrite the existing peak files. Default is True.
    kwargs: dict
        Additional keyword arguments to be passed to the clean_macs2 function.

    Returns
    -------

    """
    if type(group_names) not in [np.ndarray, list]:
        group_names = [group_names]
        cell_grouping = [cell_grouping]
        sample_names = [sample_names]
    n_jobs = min(n_jobs, len(group_names))
    if n_jobs > 1:
        pool = ProcessPoolExecutor(max_workers=n_jobs)

    if "peak_calling" in printer.uns:
        peak_calling = printer.uns["peak_calling"]
    else:
        peak_calling = {}

    for name, grouping in zip(group_names, cell_grouping):
        if (
            os.path.exists(os.path.join(printer.file_path, "macs2", f"{name}_peaks.narrowPeak"))
        ) and (not overwrite):
            print("continue")
            continue
        if n_jobs > 1:
            pool.submit(
                call_peak_one_group,
                printer.file_path,
                frag_file,
                grouping,
                name,
                preset,
                clean_temp,
                sample_names,
            )
        else:
            call_peak_one_group(
                printer.file_path, frag_file, grouping, name, preset, clean_temp, sample_names
            )
    if n_jobs > 1:
        pool.shutdown(wait=True)
    for name in group_names:
        peak = os.path.join(printer.file_path, "macs2", f"{name}_peaks.narrowPeak")
        peak = resize_bed_df(pd.read_csv(peak, sep="\t", header=None, comment="#"), peak_width)
        peak_calling[name] = peak

    if iterative_peak_merging:
        if merge_across_groups:
            cleaned_peaks = clean_macs2(
                group_names,
                os.path.join(printer.file_path, "macs2"),
                peak_width,
                printer.genome,
                preset=preset,
                **kwargs,
            )
            peak_calling["merged"] = cleaned_peaks
        else:
            for name in group_names:
                cleaned_peaks = clean_macs2(
                    [name],
                    os.path.join(printer.file_path, "macs2"),
                    peak_width,
                    printer.genome,
                    preset=preset,
                    **kwargs,
                )
                peak_calling[name + "_cleaned"] = cleaned_peaks
    printer.uns["peak_calling"] = peak_calling
