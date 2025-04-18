from __future__ import annotations

import gzip
import json
import os
import re
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import anndata
import h5py
import MOODS
import numpy as np
import pandas as pd
import pybedtools
import pyranges
import scipy
import snapatac2 as snap
import torch
from pyfaidx import Fasta
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

# def DNA_one_hot(sequence, alphabet="ACGT", dtype="float", device="cpu"):
#     """
#     Convert a DNA sequence string into a one-hot encoding tensor.
#
#     Parameters
#     ----------
#     sequence : str
#         DNA sequence string.
#     dtype : str
#         Data type of the returned tensor.
#     device : str
#         Device of the returned tensor.
#     verbose : bool
#         If True, print progress.
#
#     Returns
#     -------
#     torch.Tensor
#         One-hot encoding tensor.
#     """
#     lookup = {char: i for i, char in enumerate(alphabet)}
#     lookup["N"] = -1
#     embed = torch.zeros((len(alphabet) + 1, len(sequence)), dtype=torch.int8, device=device)
#     embed[[lookup[char] for char in sequence], torch.arange(len(sequence))] = 1
#
#     return embed[:-1, :]

DNAmapping = np.zeros((128, 4), dtype=int)
DNAmapping[65] = [1, 0, 0, 0]  # A
DNAmapping[67] = [0, 1, 0, 0]  # C
DNAmapping[71] = [0, 0, 1, 0]  # G
DNAmapping[84] = [0, 0, 0, 1]  # T
DNAmapping[78] = [0, 0, 0, 0]  # N


def DNA_one_hot(sequence, dtype="float", device="cpu"):
    """
    Convert a DNA sequence string into a one-hot encoding tensor.

    Parameters
    ----------
    sequence : str
        DNA sequence string.
    dtype : str
        Data type of the returned tensor.
    device : str
        Device of the returned tensor.
    verbose : bool
        If True, print progress.

    Returns
    -------
    torch.Tensor
        One-hot encoding tensor.
    """
    if type(sequence) is list or type(sequence) is np.ndarray:
        sequence = "".join(sequence)
    ascii_values = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    # Use the ASCII values to index into the mapping array
    one_hot_encoded = DNAmapping[ascii_values]
    return torch.from_numpy(one_hot_encoded).type(torch.int8).to(device).T


def zscore2pval(footprint):
    """
    Convert z-scores to p-values from z-test

    Parameters
    ----------
    footprint: np.ndarray | torch.Tensor | torch.cuda.FloatTensor
        Z-scores of the footprint scores.

    Returns
    -------
    pval: np.ndarray | torch.Tensor | torch.cuda.FloatTensor
        The -log 10 p-values
    """
    if type(footprint) in [torch.Tensor, torch.cuda.FloatTensor]:
        return zscore2pval_torch(footprint)
    else:
        pval = scipy.stats.norm.cdf(footprint, 0, 1)
        pval = -np.log10(pval)
        pval[np.isnan(pval)] = 0
        return pval


def zscore2pval_torch(footprint):
    """
    Convert z-scores to p-values from z-test (pytorch version)
    Parameters
    ----------
    footprint

    Returns
    -------
    torch.tensor of -log 10 p-values

    """
    # Calculate the CDF of the normal distribution for the given footprint
    pval = torch.distributions.Normal(0, 1).cdf(footprint)

    # Clamp pval to prevent log(0) which leads to -inf. Use a very small value as the lower bound.
    eps = torch.finfo(pval.dtype).eps
    pval_clamped = torch.clamp(pval, min=eps)

    # Compute the negative log10, using the clamped values to avoid -inf
    pval_log = -torch.log10(pval_clamped)

    # Optionally, to handle values very close to 1 (which would result in a negative pval_log),
    # you can clamp the output to be non-negative. This is a design choice depending on your requirements.
    pval_log = torch.clamp(pval_log, min=0, max=10)

    return pval_log


def get_protein_sequence(tf_name, species):
    from io import StringIO

    import requests
    from Bio import SeqIO

    # UniProt search API
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{tf_name}+AND+organism_name:{species}&format=fasta"

    # Send the request to UniProt
    response = requests.get(url)

    if response.status_code == 200:
        # Check if there are results
        if not response.text.strip():
            print(f"No sequences found for TF: {tf_name}")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

    fasta_data = response.text
    for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
        print("Using the first sequence found in the database")
        print(str(record.id))
        print("Sequence:", str(record.seq))
        return str(record.seq)


def get_stats_for_genome(fasta_file):
    """
    This is a function that reads a fasta file and return a dictionary of chromosome names and their lengths
    It also counts the frequency of ACGT in the genome.
    You can use this to calculate stats for your own genome.Genome object

    Parameters
    ----------
    fasta_file: str
        Path to the fasta file

    Returns
    -------
    chrom_sizes: dict
        Dictionary of chromosome names and their lengths
    bg: tuple
        Tuple of background frequencies of ACGT

    """
    genome_seq = Fasta(fasta_file)
    b = ""
    chrom_sizes = {}
    for chrom in genome_seq.keys():
        s = str(genome_seq[chrom])
        b += s
        chrom_sizes[chrom] = len(s)
    bg = MOODS.tools.bg_from_sequence_dna(str(b), 1)
    return chrom_sizes, bg


def GC_content(dna_string):
    """
    Calculate the GC content of a DNA string

    Parameters
    ----------
    dna_string: str
        DNA sequence string

    Returns
    -------
    tuple of (A, C, G, T) counts
    """
    a, c, g, t = 0, 0, 0, 0
    # Iterate over each character in the string and increment the corresponding count
    for nucleotide in dna_string:
        if nucleotide == "G":
            g += 1
        elif nucleotide == "C":
            c += 1
        elif nucleotide == "A":
            a += 1
        elif nucleotide == "T":
            t += 1

    return a, c, g, t


def get_peak_bias(
    data: anndata.AnnData | str | Path | pd.DataFrame | pyranges.PyRanges | list[str], genome
):
    """
    Calculate the GC content of the peaks (chromvar function)

    Parameters
    ----------
    data: anndata.AnnData | pd.DataFrame
        AnnData object containing cell x peak data. Or anything that the regionparser function can parse.
    genome: Genome
        Genome object

    Returns
    -------
    gc content of the peaks (from 0-1)
    """
    adata = data
    if type(adata) is anndata.AnnData:
        peaks = regionparser(list(adata.var_names))
    else:
        peaks = regionparser(adata)

    gc_contents = []
    overall_freq = [0, 0, 0, 0]
    for chrom, start, end in tqdm(peaks.iloc[:, 0:3].values, desc="Fetching GC content"):
        seq = genome.fetch_seq(chrom, start, end).upper()
        a, c, g, t = GC_content(seq)
        overall_freq[0] += a
        overall_freq[1] += c
        overall_freq[2] += g
        overall_freq[3] += t
        if (a + c + g + t) == 0:
            gc_contents.append(0.5)
        else:
            gc_contents.append((g + c) / (a + c + g + t))
    gc_contents = np.asarray(gc_contents)
    gc_contents[np.isnan(gc_contents)] = 0.5
    overall_freq = np.array(overall_freq)
    overall_freq = overall_freq / np.sum(overall_freq)
    if type(adata) is not anndata.AnnData:
        return gc_contents, overall_freq

    adata.var["gc_content"] = gc_contents
    adata.uns["bg_freq"] = overall_freq
    return None


def get_genome_bg(genome):
    """
    Get the background nucleotide frequency of the genome
    Parameters
    ----------
    genome

    Returns
    -------
    Background nucleotide frequency
    """
    genome_seq = Fasta(genome.fetch_fa())
    b = ""
    for chrom in genome_seq.keys():
        b += str(genome_seq[chrom])
    bg = MOODS.tools.bg_from_sequence_dna(str(b), 1)
    return bg


# a function to parse all kinds of regions specification
def regionparser(
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    printer=None,
    width: int | None = None,
):
    """
    This function parses the regions specification and returns a dataframe with the first three columns ['Chromosome', 'Start', 'End']
    This is the base function that makes scPrinter compatible with different regions specification.

    Parameters
    ----------
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str]
        - a pandas dataframe with the first three columns correspond to [chrom, start, end],
        - a pyranges object of the regions,
        - or a list of region identifiers, e.g. `['chr1:1000-2000', 'chr1:3000-4000']`
        - or a list of gene identifiers, e.g. `['Gene:CTCF', 'Gene:GATA1']`, the transcript start site will be used as the region center


    printer: scPrinter
        The printer object that contains the gff information. It is required if the regions are specified by gene names.
    width: int | None
        The width of the regions. When specified, the regions will be resized to the specified width.
        If None, the width will be the same as the input regions, and would be 1000bp when regions are specified by gene names.

    Returns
    -------
    regions: pd.DataFrame
        The regions dataframe with the first three columns ['Chromosome', 'Start', 'End']
    """
    if type(regions) is list:
        if ":" in regions[0] and "-" in regions[0]:
            regions = pd.DataFrame([re.split(":|-", xx) for xx in regions])
            regions.columns = ["Chromosome", "Start", "End"] + list(regions.columns)[3:]
            regions["Start"] = regions["Start"].astype("int")
            regions["End"] = regions["End"].astype("int")
        elif "Gene:" in regions[0]:
            regions = [printer.gff_db[xx[5:]] for xx in regions]
            chrom, start = [xx.chrom for xx in regions], [
                xx.start if xx.strand == "+" else xx.end for xx in regions
            ]
            regions = pd.DataFrame({"Chromosome": chrom, "Start": start})
            regions["End"] = regions["Start"] + int(printer.gene_region_width / 2)
            regions["Start"] -= int(printer.gene_region_width / 2)
        # regions_pr = dftopyranges(regions)
    elif type(regions) is pd.core.frame.DataFrame:
        regions = regions
        # regions_pr = dftopyranges(regions)
    elif type(regions) is pd.core.series.Series:
        regions = pd.DataFrame(regions.values[None])
    elif type(regions) is str:
        if ":" in regions and "-" in regions:
            # regions = pd.DataFrame([re.split(':|-', regions)], columns=['Chromosome', 'Start', 'End'])
            regions = pd.DataFrame([re.split(":|-", regions)])
            regions.columns = ["Chromosome", "Start", "End"] + list(regions.columns)[3:]

            regions["Start"] = regions["Start"].astype("int")
            regions["End"] = regions["End"].astype("int")
            # regions_pr = dftopyranges(regions)
        elif "Gene:" in regions:
            regions = printer.gff_db[regions[5:]]
            chrom, start = [regions.chrom], [
                regions.start if regions.strand == "+" else regions.end
            ]
            regions = pd.DataFrame({"Chromosome": chrom, "Start": start})
            regions["End"] = regions["Start"] + int(printer.gene_region_width / 2)
            regions["Start"] -= int(printer.gene_region_width / 2)
        else:
            # regions_pr = pyranges.readers.read_bed(regions)
            regions = pd.read_csv(regions, sep="\t", header=None)

    else:
        print("Expecting type of list, pd.dataframe, str. Got: ", type(regions), regions)

    regions.columns = ["Chromosome", "Start", "End"] + list(regions.columns)[3:]
    if width is not None:
        regions = resize_bed_df(regions, width, True)
    regions["Start"] = regions["Start"].astype("int")
    regions["End"] = regions["End"].astype("int")
    return regions


def frags_to_insertions(
    data,
    split: bool = False,
    extra_plus_shift: int = 0,
    extra_minus_shift: int = 0,
    to_csc=True,
):
    """
    Underlying function that transform snapatac fragment format to insertions

    Parameters
    ----------
    data: the anndata object that contains "fragment_paired" or "fragment_single" in the obsm
    split: split the insertions into "chr1_insertions", "chr2_insertions", ...,
    extra_plus_shift: the extra shift for the + strand (left of the paired end )
    extra_minus_shift: the extra shift for the - strand (right of the paired end)

    Returns
    -------
    insertions

    """
    if "fragment_paired" in data.obsm:
        x = data.obsm["fragment_paired"]
        insertion = csr_matrix(
            (
                np.ones(len(x.indices) * 2, dtype="uint16"),
                np.stack(
                    [x.indices + extra_plus_shift, x.indices + x.data + extra_minus_shift], axis=-1
                ).reshape((-1)),
                x.indptr * 2,
            ),
            shape=x.shape,
        )
        insertion.sort_indices()
        insertion.sum_duplicates()
    elif "fragment_single" in data.obsm:
        # The snapatac2 format fragment_single stores the start end of the reads as the column indices, data = read length
        # positive = + strand, negative read length = - strand
        # I assumed that the shift hasn't been done at all.
        x = data.obsm["fragment_single"]
        indices = np.copy(x.indices)

        mask = x.data > 0
        indices[mask] += extra_plus_shift
        indices[~mask] += 1 + extra_minus_shift
        insertion = csr_matrix(
            (
                np.ones(len(x.indices), dtype="uint16"),
                indices,
                x.indptr,
            ),
            shape=x.shape,
        )
        insertion.sort_indices()
        insertion.sum_duplicates()
    else:
        raise ValueError("No fragment data found in the obsm")

    if split:
        indx = list(
            np.cumsum(data.uns["reference_sequences"]["reference_seq_length"]).astype("int")
        )
        start = [0] + indx
        end = indx
        for chrom, start, end in zip(
            data.uns["reference_sequences"]["reference_seq_name"], start, end
        ):
            data.obsm["insertion_%s" % chrom] = (
                insertion[:, start:end].tocsc() if to_csc else insertion[:, start:end]
            )

    else:
        data.obsm["insertion"] = insertion
    # data.obsm['insertion'] = insertion
    return data


def check_snap_insertion(shift_left=0, shift_right=0):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_fragments = gzip.open(f"{tempdir}/temp_fragments.tsv.gz", "wt")
        # This checks the end to see if snapatac2 now would do insertion at end, or end-1
        for i in range(100):
            temp_fragments.write("chr1\t%d\t%d\tbarcode1\t1\n" % (4, 100))
        temp_fragments.close()
        data = snap.pp.import_data(
            f"{tempdir}/temp_fragments.tsv.gz",
            chrom_sizes=snap.genome.hg38.chrom_sizes,
            min_num_fragments=0,
            shift_left=shift_left,
            shift_right=shift_right,
            # file='testabcdefg.h5ad'
        )
        data = frags_to_insertions(data)
        v = np.array(data.obsm["insertion"][0, :200].toarray()).reshape((-1))
        # If true: The fixed version I compiled or they fixed it, else not fixed
    return v[100] == 100


def cell_grouping2cell_grouping_idx(printer, cell_grouping: list[list[str]] | np.ndarray):
    """
    Convert a cell grouping from string based to index based.
    E.g., you pass a list of lists of barcodes `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`, and you get a list of lists of indices [[0], [1,2,3]] which corresponds to the index in the single cell insertion profile.

    Parameters
    ----------
    printer: scPrinter
        The scPrinter object

    cell_grouping: list[list[str]] | np.ndarray
        The cell grouping in string based format

    Returns
    -------
    cell_grouping_idx: list[list[int]]
        The cell grouping in index based format
    """

    # concatenate all the embeddings
    barcodes_all = np.concatenate(cell_grouping)
    # get the unique ones
    uniq = np.unique(barcodes_all)
    # turn barcodes into index in the insertion sparse matrix
    uniq_ix = np.array(printer.insertion_file.obs_ix(np.array(uniq)))
    # construct a mapping from uniq barcode name to its index
    dict1 = {b: i for i, b in zip(uniq_ix, uniq)}
    # and use that to turn string based matching to index based matching
    cell_grouping_idx = [[dict1[b] for b in barcodes] for barcodes in cell_grouping]
    return cell_grouping_idx


def df2cell_grouping(printer, barcodeGroups):
    """
    Convert a dataframe of barcodes and their groupings (The input format for R's implementation)
    to a list of lists of barcodes (which is scPrinter's input format)

    Parameters
    ----------
    printer: scPrinter
        The scPrinter object
    barcodeGroups: pd.DataFrame
        The dataframe of barcodes and their groupings. It needs to at least have two columns,
        where the first column is the barcode and the second column is the group name.

    Returns
    -------
    grouping: list[list[str]]
        The list of lists of barcodes
    uniq_groups: np.ndarray
        The unique group names

    """
    set1_ = set(printer.insertion_file.obs_names)
    mask = [xx in set1_ for xx in barcodeGroups.iloc[:, 0]]
    barcodeGroups = barcodeGroups.iloc[mask,]

    grouping = []
    bar = np.array(barcodeGroups.iloc[:, 0])
    groups = np.array(barcodeGroups.iloc[:, 1])
    uniq_groups = np.unique(groups)
    for group in uniq_groups:
        grouping.append(list(bar[groups == group]))
    return grouping, uniq_groups


def resize_bed_df(
    bed: pd.DataFrame, width: int = 1000, copy: bool = True, center: str | None = None
):
    """
    Resize a bed dataframe to a given width,

    Parameters
    ----------
    bed: pd.DataFrame
        The bed dataframe with the first three columns being chromosome, start, end
    width: int
        The width of the resized bed dataframe
    copy: bool
        Whether to make a copy of the input bed dataframe or change them inplace

    Returns
    -------
    bed: pd.DataFrame
        The resized bed dataframe

    """
    bed = deepcopy(bed) if copy else bed
    if center is None:
        center = np.floor((bed.iloc[:, 1] + bed.iloc[:, 2]) / 2).astype("int")

    bed.iloc[:, 1] = np.floor(center - width / 2).astype("int")
    bed.iloc[:, 2] = np.floor(center + width / 2).astype("int")
    return bed


def merge_overlapping_bed(bed: pd.DataFrame, min_requierd_bp: int = 250, copy: bool = True):
    """
    Merge overlapping entries in a bed dataframe

    Parameters
    ----------
    bed: pd.DataFrame
        The bed dataframe with the first three columns being chromosome, start, end
    min_requierd_bp: int
        The minimum number of bp required to merge two entries
    copy: bool
        Whether to make a copy of the input bed dataframe or change them inplace

    Returns
    -------
    bed: pd.DataFrame
        The merged bed dataframe
    """
    bed = deepcopy(bed) if copy else bed
    bed = pybedtools.BedTool.from_dataframe(bed)
    bed = bed.merge(d=-1 * min_requierd_bp).to_dataframe()
    bed.columns = ["Chromosome", "Start", "End"] + list(bed.columns[3:])
    return bed


def load_entire_hdf5(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_entire_hdf5(v)
    return ret


def dftopyranges(df):
    return pyranges.PyRanges(df)


def df2regionidentifier(df):
    """
    Convert a dataframe of regions to a list of region identifiers such as "chr1:0-1000".

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame with columns ['Chromosome', 'Start', 'End'] or ['chrom', 'start', 'end'].
        If these columns are not present, assume the first three columns are the chromosome, start, and end.

    Returns
    -------
    list of str
        A list of region identifiers in the format "chr:start-end".
    """
    if "Chromosome" in df.columns and "Start" in df.columns and "End" in df.columns:
        return np.array(
            [f"{c}:{s}-{e}" for c, s, e in zip(df["Chromosome"], df["Start"], df["End"])]
        )
    elif "chrom" in df.columns and "start" in df.columns and "end" in df.columns:
        return np.array([f"{c}:{s}-{e}" for c, s, e in zip(df["chrom"], df["start"], df["end"])])
    else:
        # assume first 3 columns are chrom, start, end
        return np.array(
            [f"{c}:{s}-{e}" for c, s, e in zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])]
        )


# This function will fetch the existing entries in f
# Make comparisons to the provided list
# Then return the uniq stuff in the provided list, existing stuff, and the new stuff
def Unify_meta_info(f, addition_feats=[], entries=["group", "region"], dtypes=["str", "str"]):
    results = []
    for feat, entry, dtype in zip(addition_feats, entries, dtypes):
        if entry in f:
            if dtype == "str":
                existing = np.array(f[entry].asstr()[:])
            else:
                existing = np.array(f[entry][:])
        else:
            existing = np.array([])

        uniq = np.array(feat)
        new = uniq[~np.isin(uniq, existing)]
        results += [uniq, existing, new]

    return tuple(results)


def sort_region_identifier(regions_all):
    regions_all = pd.DataFrame(
        [re.split(":|-", xx) for xx in regions_all],
        columns=["Chromosome", "Start", "End"],
    )
    regions_all = dftopyranges(regions_all).df
    regions_all = [
        "%s:%d-%d" % (c, s, e)
        for c, s, e in zip(regions_all["Chromosome"], regions_all["Start"], regions_all["End"])
    ]
    return regions_all


# Load TF binding / habitation prediction model
def loadBindingScoreModel(h5Path):
    with h5py.File(h5Path, "r") as h5file:
        footprintMean = np.array(h5file["footprint_mean"])
        footprintSd = np.array(h5file["footprint_sd"])
        weights = [np.array(h5file["weight_%d" % i]) for i in range(6)]
        weights = [torch.from_numpy(w).float() for w in weights]
    return_dict = {
        "footprintMean": torch.from_numpy(footprintMean).float(),
        "footprintSd": torch.from_numpy(footprintSd).float(),
        "scales": [10, 20, 30, 50, 80, 100],
        "weights": weights,
    }

    return return_dict


# Load TF binding / habitation prediction model
def loadBindingScoreModel_pt(h5Path):
    model = torch.jit.load(h5Path)
    model.eval()
    return_dict = {
        "model": model,
        "scales": [int(xx) for xx in model.scales],
        "version": str(model.version),
    }

    return return_dict


def loadDispModel(h5Path):
    with h5py.File(h5Path, "r") as a:
        dispmodels = load_entire_hdf5(a)
        for model in dispmodels:
            if model in ["version"]:
                # Skip the version key
                # This is just a version string
                continue

            dispmodels[model]["modelWeights"] = [
                torch.from_numpy(dispmodels[model]["modelWeights"][key]).float()
                for key in ["ELT1", "ELT2", "ELT3", "ELT4"]
            ]
    return dispmodels


def downloadDispModel():
    return None


def downloadTFBSModel():
    return None


# Doing the same thing as conv in R, but more generalizable and in torch
def rz_conv_torch(a, n=2):
    if n == 0:
        return a
    shapes = a.shape
    if len(shapes) == 1:
        a = a[None, None, :]
    else:
        a = a.reshape((-1, 1, shapes[-1]))

    a = torch.nn.functional.conv1d(
        a,
        torch.ones((1, 1, n * 2), device=a.device, dtype=a.dtype),
        torch.zeros((1), device=a.device, dtype=a.dtype),
        padding=n,
    )[..., 0, 1:]

    if len(shapes) == 1:
        a = a[0]
    else:
        a = a.reshape(shapes)
    return a


# Doing the same thing as conv in R, but more generalizable
def rz_conv(a, n=2):
    if n == 0:
        return a
    # a can be shape of (batch, sample,...,  x) and x will be the dim to be conv on
    # pad first:
    shapes = np.array(a.shape)
    shapes[-1] = n
    a = np.concatenate([np.zeros(shapes), a, np.zeros(shapes)], axis=-1)
    ret = np.nancumsum(a, axis=-1)
    # ret[..., n * 2:] = ret[..., n * 2:] - ret[..., :-n * 2]
    # ret = ret[..., n * 2:]
    ret = ret[..., n * 2 :] - ret[..., : -n * 2]
    return ret


def strided_axis0(a, L):
    """
    This function is used to create a view of a 1D array with a sliding window of length L
    along the first axis. This is useful for creating a view of a 1D array with a sliding window
    Parameters
    ----------
    a: np.ndarray
        input array of shape (N, M, ...)
    L: int
        sliding window length

    Returns
    -------
    np.ndarray
        output array of shape (N-L+1, L, M, ...)

    """
    # Store the shape and strides info
    shp = a.shape
    s = a.strides

    # Compute length of output array along the first axis
    nd0 = shp[0] - L + 1

    # Setup shape and strides for use with np.lib.stride_tricks.as_strided
    # and get (n+1) dim output array
    shp_in = (nd0, L) + shp[1:]
    strd_in = (s[0],) + s
    return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)


# This is a fast trick to turn sth shape of (x,y,z) to (x,y,z-L+1, L)
# by tiling windows on the last axis.
def strided_lastaxis(a, L):
    s0, s1, s2 = a.strides
    l, m, n = a.shape
    return np.lib.stride_tricks.as_strided(a, shape=(l, m, n - L + 1, L), strides=(s0, s1, s2, s2))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


data = {"number": np.int64(42)}
