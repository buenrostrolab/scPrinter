import gzip
import math
import multiprocessing as mpl
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Literal

import bioframe
import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def prep_peak(file, chrs, fdr_threshold, use_quantile_scores, pad, blacklist, chrom_sizes):
    """
    Prepares a peak file by importing blacklisted regions, filtering peaks by FDR cut-off,
    padding peak summits, removing blacklisted peaks and peaks out of bounds, and adding a column
    of quantile-normalized MACS scores if specified.

    Parameters:

    file (str): The path to the peak file.
    chrs (list): A list of valid chromosome IDs.
    fdr_threshold (float): The FDR cut-off value.
    use_quantile_scores (bool): A flag indicating whether to use quantile-normalized MACS scores.
    pad (int): The padding value for peak summits.
    blacklist (str): The path to the blacklist file.
    chrom_sizes (dict): A dictionary containing chromosome sizes.

    Returns:
    peaksR (DataFrame): A DataFrame containing the prepared peak information.
    """

    file, name = file
    # Import blacklisted regions to remove if overlapping peaks
    bdf = pd.read_table(blacklist, header=None)[[0, 1, 2]]
    bdf.columns = ["chrom", "start", "end"]

    chrom = list(chrom_sizes.keys())
    end = list(chrom_sizes.values())
    sizedf = pd.DataFrame({"chrom": chrom, "end": end})
    sizedf["start"] = 0
    chrranges = sizedf[["chrom", "start", "end"]]

    dt = pd.read_table(file, header=None)
    if len(dt.columns) == 10:
        # narrowpeak format:
        # chrom, start, end, name, score, strand, signalValue, pValue, qValue, peak
        dt[1] = dt[1] + dt[9]
        dt[2] = dt[1] + 1
        dt = dt[[0, 1, 2, 3, 8]].rename(columns={8: 4})
    else:
        # summit format:
        # chrom, start, end, name, score
        dt = dt[[0, 1, 2, 3, 4]]

    # Filter in valid chromosome IDs (removing ambiguous, mito and Y chromosomes)
    dt = dt[dt[0].isin(chrs)]

    # Filter peaks by specified FDR cut-off
    dt = dt[dt[4].astype(float) > -1 * np.log10(fdr_threshold)]

    peaksR = dt
    peaksR.columns = ["chrom", "start", "end", "name", "score"]
    names = peaksR["name"]
    peaksR["name"] = [f"{name}_{x}" if name not in x else x for x in names]
    if len(peaksR) > len(np.unique(peaksR["name"])):
        # print('WARNING: Duplicate peak names detected. Appending a,b,c,d to peak names ..')
        # Step 1: Identify duplicated entries
        duplicates = peaksR.duplicated(
            subset="name", keep="first"
        )  # Keep the first occurrence of each duplicated value
        names = list(np.array(peaksR["name"]))
        # Step 2: Create a mapping of duplicated values to additional strings
        mapping = Counter()
        strings = "abcdefghijklmnopqrstuvwxyz"
        # before = len(peaksR) - len(np.unique(peaksR['name']))
        for index, is_duplicate in enumerate(duplicates):
            if is_duplicate:
                mapping[names[index]] = mapping[names[index]] + 1
                try:
                    names[index] = (
                        names[index]
                        + strings[mapping[names[index]] - 1 : mapping[names[index]] + 1]
                    )  # Mapping to a, b, c, ...
                except:
                    print(names[index], mapping[names[index]])
                    raise EOFError
        peaksR["name"] = names
        # print('before', before, len(peaksR) - len(np.unique(peaksR['name'])), "\n")

    # Padding peak summits
    peaksR = bioframe.expand(peaksR, pad=pad)

    # Remove blacklist and peaks out of bounds
    peaksR = bioframe.setdiff(peaksR, bdf)
    peaksR = bioframe.trim(peaksR, chrranges)

    # Add column of quantile-normalized MACS scores, if using this instead, to filter overlapping peaks
    if use_quantile_scores:
        peaksR["final_score"] = np.round(
            quantile_transform(
                np.array(peaksR["score"])[:, None],
                n_quantiles=min(1000, len(peaksR)),
                output_distribution="uniform",
                copy=True,
            ).squeeze(),
            3,
        )
    else:
        peaksR["final_score"] = peaksR["score"]

    peaksR["cell"] = name
    return peaksR


def cluster_remove_peaks_bioframe(df=None, by="score", decreasing=True):
    """
    This function clusters and removes duplicate peaks from a DataFrame based on a specified column.

    Parameters:
    df (DataFrame): The input DataFrame containing peak information. It should contain a "cluster" column.
    by (str): The column name to sort the peaks by. Default is "score".
    decreasing (bool): Whether to sort the peaks in descending order. (keep the ones with higher scores) Default is True.

    Returns:
    DataFrame: The modified DataFrame with duplicate peaks removed.
    """

    # Sort the ranges by sequence levels (chromosomes)
    df = bioframe.cluster(df)
    df.sort_values(by=[by], ascending=not decreasing, inplace=True)
    # Remove duplicated ranges based on cluster IDs
    df.drop_duplicates(subset=["cluster"], inplace=True)

    # Remove the cluster column from the DataFrame
    df.drop(columns=["cluster", "cluster_start", "cluster_end"], inplace=True)

    return df


def make_peaks_df_bioframe(
    summit_files,
    names,
    peak_width,
    genome,
    filter_chr=False,
    blacklist_peak_width=None,
    n=999999999,
    fdr_threshold=0.01,
    use_quantile_scores=False,
    final_peak_width=None,
    max_iter=1000,
):
    """
    Function to process MACS2 summit files, filter overlapping peaks,
    and generate a summary of the final peaks.

    Parameters:
    - summit_files (List[str]): A list of paths to the MACS2 summit files.
    - names (List[str]): A list of names corresponding to the summit files.
    - peak_width (int): The width of the peaks.
    - genome (GenomeObject): An object representing the genome.
    - filter_chr (bool): A boolean indicating whether to filter chromosomes.
    - blacklist_peak_width (Optional[int]): The width of the blacklist peaks.
    - n (int): The number of peaks to return.
    - fdr_threshold (float): The false discovery rate threshold.
    - use_quantile_scores (bool): A boolean indicating whether to use quantile scores for filtering.
    - final_peak_width (Optional[int]): The width of the final peaks.
    - max_iter (int): The maximum number of iterations for peak clearing.

    Returns:
    - cleaned_peaks (pd.DataFrame): A DataFrame containing the cleaned peaks. (Bed format)
    - summary (pd.DataFrame): A DataFrame containing a summary of the cleaned peaks. (Contains for summary statistics)

    """

    chrom_sizes = genome.chrom_sizes
    blacklist = genome.fetch_blacklist()
    if blacklist_peak_width is None:
        blacklist_peak_width = peak_width
    # Length to add/pad on each end
    blacklist_pad = int(blacklist_peak_width / 2)
    pad = int(peak_width / 2)

    print("Reading in peak summit file(s):")
    print("NOTE: Assuming all start coordinates are 0-based ..\n")

    print(f"Padding peak summits by: {blacklist_pad} bp on either side for")
    print(
        "Removing peaks overlapping with blacklisted regions and out of bound peaks based on chromosome sizes ..\n"
    )

    # Import chromosome sizes and remove summits not mapping to valid seqnames (chr 1-22 and X)
    chrom = list(chrom_sizes.keys())
    end = list(chrom_sizes.values())
    sizedf = pd.DataFrame({"seqnames": chrom, "end": end})
    sizedf["start"] = 0

    chrs = sizedf["seqnames"].loc[~sizedf["seqnames"].isin(["chrY", "chrM", "MT"])].tolist()
    if filter_chr:
        chrs = [x for x in chrs if "_" not in x]

    # Combine peak sets, if multiple
    func = partial(
        prep_peak,
        chrs=chrs,
        fdr_threshold=fdr_threshold,
        use_quantile_scores=use_quantile_scores,
        pad=blacklist_pad,
        blacklist=blacklist,
        chrom_sizes=chrom_sizes,
    )
    cpu_num = min(mpl.cpu_count(), len(summit_files))
    if cpu_num == 1:
        peaks = func((summit_files[0], names[0]))
    else:
        peaks = list(
            process_map(
                func,
                zip(summit_files, names),
                max_workers=cpu_num,
                chunksize=math.ceil(len(summit_files) / cpu_num),
                total=len(summit_files),
            )
        )

        peaks = pd.concat(peaks, axis=0)

    peaks = peaks.iloc[np.random.permutation(len(peaks))]
    if blacklist_pad != pad:
        peaks_center = (peaks["start"] + peaks["end"]) // 2
        peaks["start"] = peaks_center - pad
        peaks["end"] = peaks_center + pad

    # Filter peaks based on summit score/quantile normalized summit score
    print(
        "Filtering overlapping peaks based on normalized quantiles of peak summit scores .."
        if use_quantile_scores
        else "Filtering overlapping peaks based on peak summit score .."
    )

    i = 0
    peakscoverage = peaks
    raw_peaks = peaks.copy()
    peaks_all = []
    # return peaks
    while len(peakscoverage) > 0:
        i += 1
        peak_select = cluster_remove_peaks_bioframe(
            peakscoverage, by="final_score", decreasing=True
        )
        print(
            "round:",
            i,
            len(peakscoverage),
            "peaks unresolved",
            len(peak_select),
            "peaks selected",
        )
        peakscoverage = bioframe.setdiff(peakscoverage, peak_select).copy()
        peaks_all.append(peak_select)
        if i > max_iter:
            print("WARNING: Maximum number of iterations reached. Exiting ..")
            print("Usually, it's because you use the wrong genome name")
            break
    peaks = pd.concat(peaks_all)
    print("finish clearing")
    # Export the final result as a DataFrame; getting the top (or as many) n peaks based on the score and then resort based on genomic position.
    fP = peaks
    fP["rank"] = range(1, len(fP) + 1)

    nout = n if n < len(fP) else len(fP)

    # print(f"Returning only the top {n} peaks post-filtering ..")
    if final_peak_width is not None:
        final_pad = int(final_peak_width / 2)
        summit = ((fP["start"] + fP["end"]) / 2).astype("int")
        fP["start"] = (summit - final_pad).astype("int")
        fP["end"] = (summit + final_pad).astype("int")

    cleaned_peaks = (
        fP.sort_values(by="final_score", ascending=False)
        .head(nout)
        .sort_values(by=["chrom", "start", "end"])
    )
    print("finish sorting")
    summary = (
        bioframe.overlap(cleaned_peaks, raw_peaks)
        .groupby("name")
        .agg(
            {
                "chrom": "first",
                "start": "first",
                "end": "first",
                "name_": list,
                "score_": list,
                "final_score_": list,
                "cell_": list,
                "rank": "first",
            }
        )
        .reset_index()
    )
    summary.drop(columns=["name"], inplace=True)
    summary = summary.rename(
        columns={
            "name_": "name",
            "score_": "score",
            "final_score_": "final_score",
            "cell_": "cell",
        }
    )
    print("finished summary")
    summary = summary[["chrom", "start", "end", "name", "final_score", "score", "cell", "rank"]]
    summary["n_cell_line"] = [len(xx) for xx in summary["score"]]
    summary = summary.sort_values(by=["chrom", "start", "end"])

    return cleaned_peaks, summary


def clean_macs2(
    name,
    outdir,
    peak_width,
    genome,
    filter_chr=False,
    blacklist_peak_width=None,
    n=999999999,
    fdr_threshold=0.01,
    use_quantile_scores=False,
    final_peak_width=None,
    max_iter=1000,
    preset: Literal["seq2PRINT", "chromvar", None] = None,
):
    """
    This function reads in MACS2 summit files, filters overlapping peaks,
    and generates a summary of the final peaks.

    Parameters
    ----------
    name : str or List[str]
        The name(s) corresponding to the summit files.
    outdir : str
        The output directory where the summit files are located.
    peak_width : int
        The width of the peaks.
    genome : GenomeObject
        An object representing the genome.
    filter_chr : bool, optional
        A boolean indicating whether to filter chromosomes. Default is False.
    blacklist_peak_width : int, optional
        The width of the blacklist peaks. Default is None.
    n : int, optional
        The number of peaks to return. Default is 999999999.
    fdr_threshold : float, optional
        The false discovery rate threshold. Default is 0.01.
    use_quantile_scores : bool, optional
        A boolean indicating whether to use quantile scores for filtering. Default is False.
    final_peak_width : int, optional
        The width of the final peaks. Default is None.
    max_iter : int, optional
        The maximum number of iterations for peak clearing. Default is 1000.
    preset : Literal['seq2PRINT', 'chromvar', None], optional
        This would overwrite most of the parameters above. Default is None.

    Returns
    -------
    cleaned_peaks : pd.DataFrame
        A DataFrame containing the cleaned peaks in Bed format.
    """
    if preset == "seq2PRINT":
        peak_width = 128
        blacklist_peak_width = 1000
        filter_chr = False
        n = 300000
        fdr_threshold = 0.01
        final_peak_width = 1000
    elif preset == "chromvar":
        peak_width = 800
        blacklist_peak_width = 800
        final_peak_width = 300

    if type(name) not in [list, np.ndarray]:
        name = [name]
    cleaned_peaks, summary = make_peaks_df_bioframe(
        [os.path.join(outdir, n + "_summits.bed") for n in name],
        name,
        peak_width,
        genome,
        filter_chr=filter_chr,
        blacklist_peak_width=blacklist_peak_width,
        n=n,
        fdr_threshold=fdr_threshold,
        use_quantile_scores=use_quantile_scores,
        final_peak_width=final_peak_width,
        max_iter=max_iter,
    )
    return cleaned_peaks, summary


def macs2(frag_file, name, outdir, format="BEDPE", p_cutoff=None, genome_size=None):
    """
    This function runs the MACS2 peak calling algorithm.

    Parameters
    ----------
    frag_file : str or List[str]
        The path(s) to the fragment file(s).
    name : str
        The name for the output files.
    outdir : str
        The output directory.
    format : str, optional
        The format of the fragment file(s). Default is "BEDPE".
    p_cutoff : float, optional
        The p-value cutoff for peak calling. Default is None. When set as None, it will use qvalue instead.
        In general, use p_cutoff=0.01 for training seq2PRINT model. Use qvalue instead for short-listed peaks
        to calculate seqattrs.

    Returns
    -------
    None
    """

    if type(frag_file) is not list:
        frag_file = [frag_file]
    for i in range(len(frag_file)):
        frag_file[i] = str(frag_file[i]).replace("//", "/")  # macs2 does not like double slashes
    # assert genome is not None, "genome must be provided"
    commands = (
        ["macs2", "callpeak", "--nomodel", "-t"]
        + frag_file
        + [
            "--outdir",
            outdir,
            "-n",
            name,
            "-f",
            format,
            "--nolambda",
            "--keep-dup",
            "all",
            "--call-summits",
            "--nomodel",
            "-B",
            "--SPMR",
            "--shift",
            "75",
            "--extsize",
            "150",
        ]
    )
    # commands = [
    #     "macs2",
    #     "callpeak",
    #     "-t"] + frag_file + [
    #     "--outdir",
    #     outdir,
    #     "-n",
    #     name,
    #     "-f",
    #     format,
    #     "--nolambda",
    #     "--keep-dup",
    #     "all",
    #     "--call-summits"
    # ]
    if p_cutoff is not None:
        commands.extend(["-p", str(p_cutoff)])
    else:
        commands.extend(["-q", "0.01"])
    if genome_size is not None:
        commands.extend(["-g", str(genome_size)])
    print("running macs2 with", " ".join(commands))
    subprocess.run(commands)


def macs3(frag_file, name, outdir, format="BEDPE"):
    commands = [
        "macs3",
        "hmmratac",
        "-i",
        frag_file,
        "-f",
        "BEDPE",
        "-n",
        name,
        "--outdir",
        outdir,
    ]
    subprocess.run(commands)


def make_tile_window(genome, window_size=5000):
    """
    Create a tiled window across the genome.

    Parameters
    ----------
    genome
    window_size

    Returns
    -------

    """

    chrom_sizes = genome.chrom_sizes
    chroms = list(chrom_sizes.keys())
    chroms.sort()
    windows = []
    for chrom in chroms:
        chrom_length = chrom_sizes[chrom]
        for start in range(0, chrom_length, window_size):
            end = min(start + window_size, chrom_length)
            windows.append([chrom, start, end])
    windows_df = pd.DataFrame(windows, columns=["chrom", "start", "end"])
    return windows_df
