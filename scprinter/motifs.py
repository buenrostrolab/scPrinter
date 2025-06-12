"""
Scripts adapted from 10x motif-matching
Tools for analyzing motifs in the genome.
"""

from __future__ import annotations

import itertools
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from itertools import product
from pathlib import Path

import h5py
import MOODS
import MOODS.parsers
import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
from Bio import motifs
from pyfaidx import Fasta
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, diags, hstack, vstack
from statsmodels.stats.multitest import fdrcorrection
from tangermeme.io import read_meme

try:
    from tangermeme.tools.tomtom import tomtom
except:
    from memelite import tomtom

from tqdm.auto import tqdm, trange
from typing_extensions import Literal

from . import genome
from .datasets import CisBP_Human, FigR_motifs_human, FigR_motifs_mouse, JASPAR2022_core
from .utils import regionparser

"""
Most of the code are from modiscolite, and adapted to include the delta effects.
Chose to copy code because modiscolite somehow changes the backend of matplotlib and inline version
"""


def read_pfms(motif, trim_threshold=0.3, prefix=""):
    if isinstance(motif, Motifs):
        dict1 = {}
        for mtx in motif.all_motifs:
            name = mtx.name
            c = mtx.counts
            mtx = np.array([c["A"], c["C"], c["G"], c["T"]]).T
            mtx = mtx / np.sum(mtx, axis=1)[:, None]
            dict1[prefix + name] = mtx.T
    elif isinstance(motif, dict):
        dict1 = {prefix + name: mtx for name, mtx in motif.items()}
    else:
        if ".h5" in motif:
            # modisco format
            modisco_motifs = {}
            with h5py.File(motif, "r") as modisco_results:
                for contribution_dir_name in modisco_results.keys():
                    metacluster = modisco_results[contribution_dir_name]
                    key = lambda x: int(x[0].split("_")[-1])

                    for idx, (key_, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                        ppm = np.array(pattern["sequence"][:])
                        cwm = np.array(pattern["contrib_scores"][:])
                        pattern_name = f"{contribution_dir_name}.pattern_{idx}"
                        score = np.sum(np.abs(cwm), axis=1)
                        trim_thresh = (
                            np.max(score) * trim_threshold
                        )  # Cut off anything less than 30% of max score
                        pass_inds = np.where(score >= trim_thresh)[0]
                        trimmed = ppm[np.min(pass_inds) : np.max(pass_inds) + 1]
                        modisco_motifs[prefix + pattern_name] = trimmed.T
            dict1 = modisco_motifs
        else:
            dict1 = read_meme(motif)
            dict1 = {prefix + k: v.detach().numpy() for k, v in dict1.items()}
    return dict1


def tomtom_motif_motif_matrix(
    motifs_1, prefix_1="", motifs_2=None, prefix_2="", trim_threshold=0.3
):
    """
    create a motif motif matrix using tomtom
    Parameters
    ----------
    motif: list[str, Path, motifs.Motifs] | str, Path, motifs.Motifs

    Returns
    -------
    p values: np.ndarray
    """

    if not isinstance(motifs_1, list):
        motifs_1 = [motifs_1]
    if not isinstance(prefix_1, list):
        prefix_1 = [prefix_1]
    if motifs_2 is None:
        motifs_2 = motifs_1
        prefix_2 = prefix_1
    if not isinstance(motifs_2, list):
        motifs_2 = [motifs_2]
    if not isinstance(prefix_2, list):
        prefix_2 = [prefix_2]

    all_names_1 = []
    all_pfms_1 = []

    for m, p in zip(motifs_1, prefix_1):
        m = read_pfms(m, trim_threshold=trim_threshold, prefix=p)
        all_names_1 += list(m.keys())
        all_pfms_1 += list(m.values())

    all_names_2 = []
    all_pfms_2 = []

    for m, p in zip(motifs_2, prefix_2):
        m = read_pfms(m, trim_threshold=trim_threshold, prefix=p)
        all_names_2 += list(m.keys())
        all_pfms_2 += list(m.values())

    p, scores, offsets, overlaps, strands = tomtom(all_pfms_1, all_pfms_2, n_cache=1000)
    df = pd.DataFrame(
        {
            "Query_ID": np.repeat(list(all_names_1), len(all_names_2)),
            "Target_ID": list(all_names_2) * len(all_names_1),
            "Optimal_offset": offsets.reshape(-1),
            "p-value": p.reshape(-1),
        }
    )
    df["E-value"] = df["p-value"] * len(all_names_2)
    df["q-value"] = fdrcorrection(df["p-value"])[1]
    df["Overlap"] = overlaps.flatten()
    df["Orientation"] = ["+" if x == 0 else "-" for x in strands.reshape((-1))]
    return df


def consecutive(data, stepsize=1):
    """
    Find consecutive numbers in a list, used to merge overlapping hits.

    Parameters:
    ----------
    data : list
        A list of numbers.
    stepsize : int, optional
        The difference between consecutive numbers. Default is 1.

    Returns:
    -------
    list
        A list of consecutive numbers. Each element in the list is a numpy array of consecutive numbers.
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


# This function is necessary for the MOODS package to work under multiprocessing
def thresholds_from_p(m, bg, pvalue):
    return MOODS.tools.threshold_from_p(m, bg, pvalue)


class PFM:
    """
    A simple wrapper for PFM matrices to make it compatible with Bio.motifs
    """

    def __init__(self, name, counts):
        self.name = name
        self.counts = counts
        self.length = len(counts["A"])


def parse_jaspar(file_path):
    """
    Parse Jaspar-like format motifs.

    Parameters
    ----------
    file_path : str
        The path to the file containing the motifs in Jaspar format.

    Returns
    -------
    list of PFM objects
        A list of PFM (Position Frequency Matrix) objects representing the parsed motifs.

    Notes
    -----
    The Jaspar format is a simple text format used to store position frequency matrices (PFMs)
    representing transcription factor binding sites. Each motif is represented by a name and
    a set of weights for each nucleotide (A, C, G, T) at each position.

    The format of the input file is as follows:
    - Each motif is represented by a line starting with ">".
    - The name of the motif follows the ">" symbol.
    - Each line representing the weights for a nucleotide starts with the nucleotide symbol.
    - The weights for each position are separated by spaces.

    Example
    -------
    The following is an example of a Jaspar format file:

    >MA0004.1_Drosophila_melanogaster
    A [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
    C [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
    G [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
    T [0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]

    """
    # Initialize variables
    records = []
    record = None

    # Open the file for reading
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line.startswith(">"):  # Record name line
                # Save the previous record if it exists
                if record:
                    records.append(PFM(record["name"], record["weights"]))
                # Start a new record
                record = {
                    "name": line[1:].split(" ")[-1],
                    "weights": {"A": [], "C": [], "G": [], "T": []},
                }
            elif len(line) > 0:
                # Parse the weight line
                nucleotide, values_str = line.split(" ", 1)
                values = list(map(float, values_str.strip(" ").strip("[]").split()))
                if record:
                    record["weights"][nucleotide] = values

        # Don't forget to add the last record after exiting the loop
        if record:
            records.append(PFM(record["name"], record["weights"]))

    return records


def _jaspar_to_moods_matrix(jaspar_motif, bg, pseudocount, mode="motifmatchr"):
    """
    Convert a JASPAR motif to a MOODS matrix.

    Parameters
    ----------
    jaspar_motif : list of PFM matrices
        The JASPAR motif to be converted.

    bg : tuple of float
        The background nucleotide frequency.

    pseudocount : float
        The pseudocount for the motif.

    mode : str, optional
        The mode of the motif matching. By default, it is set to "motifmatchr".

    Returns
    -------
    m : list of tuples
        The converted MOODS matrix. Each tuple represents the log odds ratio for each nucleotide.
    """
    with tempfile.NamedTemporaryFile() as fn:
        f = open(fn.name, "w")
        for base in "ACGT":
            line = " ".join(str(x) for x in jaspar_motif.counts[base])
            f.write(line + "\n")

        f.close()
        if mode != "motifmatchr":
            m = MOODS.parsers.pfm_to_log_odds(fn.name, bg, pseudocount, 2)
        else:
            # This is to be consistent with motifmatchr
            even = [0.25, 0.25, 0.25, 0.25]
            m = MOODS.parsers.pfm_to_log_odds(fn.name, even, pseudocount, 2)
            m = [tuple(m[i] - (np.log2(0.25) - np.log2(bg[i]))) for i in range(len(bg))]
        return m


def scan_func(
    seq,
    motifs_length,
    motifs_name,
    chr,
    start,
    end,
    peak_idx,
    tfs,
    clean,
    split_tfs,
    strand,
):
    """
    Scan a sequence for motifs. This function is used as a global function for multiprocessing.
    This is basically the _parse_scan_results_all function, but with the scanner scan function within each child process.

    Parameters
    ----------
    seq : str
        String of "ACGT" representing the sequence to scan for motifs.
    motifs_length : list | np.ndarray
        Length of the motifs. This is necessary to infer the end positons of the motif matches.
    motifs_name : list | np.ndarray
        Name of the motifs.
    chr : str
        Chromosome name.
    start : int
        Start position of the sequence.
    end : int
        End position of the sequence.
    peak_idx : int
        The index of the peak (to associate results back to the peak).
    tfs : set
        Set of TFs to keep. (For cistarget motif collection, if one motif keeps 20 TFs, you can use this to keep only the TF you care about).
    clean : bool
        Whether to clean the results (merge overlapping hits).
    split_tfs : bool
        Whether to split the TFs by comma (for cistarget motif collection).
    strand : bool
        Whether to keep the strand information.

    Returns
    -------
    list
        List of parsed scan results. Each result is a list containing the motif start position, motif end position, score, and other relevant information.
    """
    # fetch the MOODS scanner and scan the sequence
    global scanner
    scan_res = scanner.scan(str(seq))
    # parse the results, keep only the position and score information
    results = [[[xxx.pos, xxx.score] for xxx in xx] for xx in scan_res]

    # parse the scan results.
    return _parse_scan_results_all(
        results,
        motifs_length,
        motifs_name,
        {"chr": chr, "start": start, "end": end, "index": peak_idx},
        tfs,
        clean,
        split_tfs,
        strand,
    )


def _parse_scan_results_all(
    moods_scan_res: list,
    motifs_length: list | np.ndarray,
    motifs_name: list | np.ndarray,
    bed_coord: dict,
    tfs: set,
    clean=False,
    split_tf=True,
    strand_spec=True,
):
    """
    Parse results of MOODS.scan.scan_dna and return

    Parameters
    ----------
    moods_scan_res: list
        list of scan results from MOODS length of motifs x 2 (first forward then reverse strand)
        each item in the list is another list of scanned results [position, scores]
    motifs_length: list | np.ndarray
        length of the motifs
    motifs_name: list | np.ndarray
        name of the motifs
    bed_coord: dict
        coordinates of the peak
    tfs: set
        set of TFs to keep. (for cistarget motif collection, if one motif keeps 20 TFs, you can use this to keep only the TF you care)
    clean: bool
        clean the results (merge overlapping hits)
    split_tf: bool
        split the TFs by comma (for cistarget motif collection)
    strand_spec:
        whether to keep the strand information

    Returns
    -------

    """
    all_hits = []  # final return, lists of hits in this region.

    # First group by motif.name
    tf2results = {}
    num_motifs = len(motifs_name)

    for motif_idx, hits in enumerate(moods_scan_res):
        # The scan results from MOODS is organized in a way that the first half is the forward strand,
        # the second half is the reverse strand, hence idx % len(motifs_names), gives the correct motif name
        # get information of motif_name, length and stand information (which is * if strand_spec=False)
        motif_name = motifs_name[motif_idx % num_motifs]
        motif_name = motif_name.split(",") if split_tf else [motif_name]
        motif_length = motifs_length[motif_idx % num_motifs]
        strand = -1 if motif_idx >= len(motifs_name) else 1
        if not strand_spec:
            strand = "*"

        for name in motif_name:
            if name not in tfs:
                continue

            # Maintain a dictionary of tf name, strand to results.
            if (name, strand) not in tf2results:
                tf2results[(name, strand)] = []

            v = []
            for h in hits:
                # get the motif start position
                motif_start = int(h[0])
                motif_end = int(h[0]) + motif_length
                score = round(h[1], 4)
                # If direct output, then organize the record directly without merge overlapping hits
                if not clean:
                    record = [
                        bed_coord["chr"],
                        bed_coord["start"],
                        bed_coord["end"],
                        bed_coord["index"],
                        name,
                        score,
                        strand,
                        motif_start,
                        motif_end,
                    ]
                    all_hits.append(record)
                else:
                    # If wants to clean, then stack them directly then clean later.
                    motif_start = int(h[0])
                    motif_end = int(h[0]) + motif_length
                    v.append([motif_start, motif_end])
            if clean:
                tf2results[(name, strand)] += v
    if not clean:
        return all_hits

    for key in tf2results:
        name, strand = key
        v = tf2results[(name, strand)]
        if len(v) == 0:
            continue
        v = [np.arange(s, e) for s, e in v]
        v = np.unique(np.concatenate(v).reshape((-1)))
        v = np.sort(v)
        v = consecutive(v)
        for h in v:
            motif_start = np.min(h)
            motif_end = np.max(h)
            record = [
                bed_coord["chr"],
                bed_coord["start"],
                bed_coord["end"],
                bed_coord["index"],
                name,
                0,
                strand,
                motif_start,
                motif_end,
            ]

            all_hits.append(record)
    return all_hits


def scan_func_kmer(seq, id):
    """
    Scan a sequence for kmers. This function is used as a global function for multiprocessing.

    Parameters
    ----------
    seq : str
        The sequence to scan for kmers.
    id : int
        The identifier for the sequence.

    Returns
    -------
    tuple
        A tuple containing the kmer counts and the identifier. The kmer counts are stored in a numpy array,
        and the identifier is an integer.
    """
    global kmer2id
    for k in kmer2id:
        length = len(k)
        break
    res = np.zeros((len(kmer2id)))
    for i in range(len(seq) - length):
        kmer = seq[i : i + length]
        if kmer in kmer2id:
            res[kmer2id[kmer]] += 1
    return res, id


class Kmers:
    """
    A class for kmer matching to do chromvar
    ref_path_fa: str | Path
        Path to the reference genome fasta file. An eazy way would be passing, the genome.fetch_fa() function
    k: int
        kmer length
    reverse: bool
        whether to scan the reverse strand
    n_jobs: int
        number of cores to use

    """

    def __init__(
        self,
        ref_path_fa: str | Path,
        k: int = 6,
        reverse: bool = False,
        n_jobs: int = 32,
    ):
        self.k = k
        # self.gap = gap
        self.n_jobs = n_jobs
        self.genome_seq = Fasta(ref_path_fa)
        self.kmers = self._generate_kmers()
        self.reverse = reverse

    def _generate_kmers(self):
        # Generate all possible k-mers of length k using ACTG
        nucleotides = ["A", "C", "G", "T"]
        return ["".join(kmer) for kmer in product(nucleotides, repeat=self.k)]

    def chromvar_scan(self, adata, verbose: bool = True):
        """
        This function performs motif matching on the given peaks using k-mers.
        It scans the reference genome sequence for k-mers and counts their occurrences.
        The results are stored in the `adata.varm["motif_match"]` field.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing the peak information.
        verbose : bool, optional
            A boolean indicating whether to display progress bars.

        Returns
        -------
        None
            The results are stored in the `adata.varm["motif_match"]` field.
        """

        peaks = regionparser(list(adata.var_names))
        peaks = np.array(peaks)
        global kmer2id
        kmer2id = {}
        for i, kmer in enumerate(self.kmers):
            kmer2id[kmer] = i
        pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        p_list = []
        bar = trange(len(peaks) if not self.reverse else int(len(peaks) * 2), disable=not verbose)
        for peak_idx, peak in enumerate(peaks):
            # peak = peaks_iter[peak_idx]
            chr = peak[0]
            start = int(peak[1])
            end = int(peak[2])
            seq = self.genome_seq[chr][start:end].seq.upper()
            p_list.append(
                pool.submit(
                    scan_func_kmer,
                    seq,
                    peak_idx,
                )
            )
            if self.reverse:
                seq = self.genome_seq[chr][start:end].reverse.complement.seq.upper()
                p_list.append(
                    pool.submit(
                        scan_func_kmer,
                        seq,
                        peak_idx,
                    )
                )
        res = np.zeros((len(peaks), len(self.kmers)))
        for p in as_completed(p_list):
            bar.update(1)
            v, id = p.result()
            res[id] += v
        pool.shutdown(wait=True)

        res = pd.DataFrame(res, index=adata.var_names, columns=self.kmers)
        adata.varm["motif_match"] = res.to_numpy()
        adata.uns["motif_name"] = list(res.columns)


class Motifs:
    """
    A class for motif matching based on MOODS

    Parameters
    ----------
    ref_path_motif : str | Path
        Path to the motif file, in JASPAR format
    ref_path_fa : str | Path
        Path to the reference genome fasta file. An eazy way would be passing, the genome.fetch_fa() function
    bg : Literal['even', 'genome'] | tuple, optional
        Background nucleotide frequency, by default 'even' stands for equal frequency of A, C, G, T
        'genome' stands for the frequency of A, C, G, T in the reference genome
        When passing a tuple of size 4, they would be used as background frequency
    pseudocount : float, optional
        Pseudocount for the motif, by default 0.8 (same as motifmatchR)
    pvalue : float, optional
        P-value threshold for motif matching, by default 5e-5
    n_jobs : int, optional
        Number of cores to use, by default 32
    split_tf: bool, optional
        Whether to split the TFs by comma (for cistarget motif collection), by default True
    mode: Literal['motifmatchr', 'moods'], optional
        The mode of the motif matching, by default 'motifmatchr'
    """

    def __init__(
        self,
        ref_path_motif: str | Path,
        ref_path_fa: str | Path,
        bg: Literal["even", "genome"] | tuple = "even",
        pseudocount: float = 0.8,
        pvalue: float = 5e-5,
        n_jobs: int = 32,
        split_tf: bool = True,
        mode: Literal["motifmatchr", "moods"] = "motifmatchr",
        motif_name_func=None,
    ):
        self.n_jobs = n_jobs
        self.mode = mode
        self.all_motifs = parse_jaspar(ref_path_motif)  # now it's a list of PFM objects
        self.names = [
            set(motif.name.split(",")) if split_tf else {motif.name} for motif in self.all_motifs
        ]  # split the TFs by comma (for cistarget motif collection)

        # If there is a motif_name_func, use it to convert motif names to the desired format (remove \t, _ etc.)
        self.names = (
            [set([motif_name_func(m) for m in motif]) for motif in self.names]
            if motif_name_func is not None
            else self.names
        )
        if motif_name_func is not None:
            for m in self.all_motifs:
                m.name = motif_name_func(m.name)
        # a non-duplicated set of motif names by TF

        self.tfs = set().union(*self.names)
        self.genome_seq = Fasta(ref_path_fa)
        if bg == "even":
            self.bg = [0.25, 0.25, 0.25, 0.25]
        elif bg == "genome":
            b = ""
            for chrom in self.genome_seq.keys():
                b += str(self.genome_seq[chrom])

            self.bg = MOODS.tools.bg_from_sequence_dna(str(b), 1)
        else:
            self.bg = bg
        (
            self.pre_matrices_p,
            self.pre_thresholds_p,
            self.pre_matrices_m,
            self.pre_thresholds_m,
        ) = self._prepare_moods_settings(self.all_motifs, self.bg, pseudocount, pvalue)
        self.pseudocount = pseudocount

        self.pvalue = pvalue

    def prep_scanner(
        self,
        tf_genes: list[str] | None = None,
        pseudocount: float = 0.8,
        pvalue: float = 5e-5,
        window: int = 7,
    ):
        """
        Prepare the MOODS scanner for motif matching

        Parameters
        ----------
        tf_genes: list[str] | None
            List of TFs to be used for motif matching, by default None, which means all TFs. Note that if you pass motif_name_func and/or split_tf,
            you should specify TFs by their names after being processed by those (so likely the true TF names).
        pseudocount: float
            Pseudocount for the motif, by default 0.8 (same as motifmatchR)
        pvalue: float
            P-value threshold for motif matching, by default 5e-5
        window: int
            Window size for motif matching, by default 7 (same as motifmatchR), passed to MOODS

        Returns
        -------
        scanner: MOODS.scan.Scanner

        """
        global scanner
        if tf_genes is None:
            tf_genes = self.tfs
        tf_genes_set = set(tf_genes)
        select = (
            slice(None)
            if tf_genes is None
            else [len(name & tf_genes_set) > 0 for name in self.names]
        )
        self.select = select

        if pseudocount != self.pseudocount or pvalue != self.pvalue:
            # motif = self.all_motifs[select]
            motif = [motif for motif, keep in zip(self.all_motifs, select) if keep]
            # Each TF gets a matrix for the + and for the - strand, and a corresponding threshold
            matrices_p, threshold_p, matrices_m, threshold_m = self._prepare_moods_settings(
                motif, self.bg, pseudocount, pvalue
            )
        else:

            matrices_p, threshold_p, matrices_m, threshold_m = (
                self.pre_matrices_p[select],
                self.pre_thresholds_p[select],
                self.pre_matrices_m[select],
                self.pre_thresholds_m[select],
            )

        matrices = np.concatenate([matrices_p, matrices_m])
        thresholds = np.concatenate([threshold_p, threshold_m])

        scanner_ = MOODS.scan.Scanner(window)  # parameter is the window size
        scanner_.set_motifs(matrices, self.bg, thresholds)
        self.scanner = scanner_
        scanner = scanner_
        self.tfs = set(tf_genes)
        # print ("finish preparing scanner")
        return scanner_

    def chromvar_scan(self, adata, verbose: bool = True):
        """
        Perform motif scanning on the given peaks and store the results in the AnnData object.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the peak information. The peak information should be stored in the `.var_names` attribute.
        verbose : bool, optional
            Whether to display a progress bar during the motif scanning process. Default is True.

        Returns
        -------
        None
            The function modifies the AnnData object in-place by adding the motif match information as `.obsm['motif_match']` and storing the motif names as `.uns["motif_name"]`.
        """
        peaks = regionparser(list(adata.var_names))
        # Function implementation goes here
        res = self.scan_motif(peaks, verbose=verbose, count=True)
        res = np.array(res)
        res = pd.DataFrame(
            res,
            index=adata.var_names,
            columns=[m.name for m, select in zip(self.all_motifs, self.select) if select],
        )
        adata.varm["motif_match"] = res.to_numpy()
        adata.uns["motif_name"] = list(res.columns)

    def collect_child_process(
        self, p_list, verbose, bar, count, motifs_length, name2id, maps, break_on_min_jobs
    ):
        """
        Collect completed processes from the process list and process their motif mapping results.

        Parameters
        ----------
        p_list : list
            A list of Process objects representing the child processes.
        verbose : bool
            A flag indicating whether to display a progress bar.
        bar : tqdm.tqdm
            A progress bar object for displaying the progress.
        count : bool
            A flag indicating whether to count the occurrences of each motif.
        motifs_length : list
            A list containing the lengths of the motifs.
        name2id : dict
            A dictionary mapping motif names to their indices.
        maps : list
            A list to store the results of the motif scanning process.
        break_on_min_jobs : bool
            A flag indicating whether to break the loop when the number of remaining processes is less than or equal to the number of workers.

        Returns
        -------
        None
            The function modifies the `maps` list in-place by adding the results of the completed processes.
        """

        for p in as_completed(p_list):
            if verbose:
                bar.update(1)
            all_hits = p.result()
            if len(all_hits) > 0:
                idx = all_hits[0][3]
                if not count:
                    maps[idx] = all_hits
                else:
                    freq = np.zeros((1, len(motifs_length)))
                    for hit in all_hits:
                        freq[0, name2id[hit[4]]] = 1
                    maps[idx] = freq
            p_list.remove(p)
            del p

            if break_on_min_jobs and len(p_list) <= self.n_jobs:
                break

    def scan_motif(
        self,
        peaks_iter,
        clean: bool = False,
        concat: bool = True,
        verbose: bool = False,
        split_tfs: bool = True,
        strand: bool = True,
        count: bool = False,
    ):
        """
        Scan motifs in the peaks iterator.

        Parameters
        ----------
        peaks_iter : iterable
            An iterable yielding a list [chrom, start, end], e.g., a pandas dataframe with three columns.
            This function currently only supports a specific format of regions.

        clean : bool, optional
            Whether to clean the output. If True, overlapping motif hits of the same TF will be merged into one hit.
            Default is False.

        concat : bool, optional
            Whether to concatenate the output. If False, the output will be a list of the same length as the number of peaks,
            with each element being a list of motif hits for each TF. Default is True.

        verbose : bool, optional
            Whether to display a progress bar. Default is False.

        split_tfs : bool, optional
            Whether to split the output by transcription factors. Default is True.

        strand : bool, optional
            Whether to consider the strand information. Default is True.

        count : bool, optional
            Whether to count the occurrences of each motif. If True, the output will be a frequency matrix. Default is False.

        Returns
        -------
        list or numpy.ndarray
            The output of the motif scanning process. The format depends on the values of the `clean`, `concat`, and `count` parameters.
            But roughly the output by columns are: [peak_chrom,peak_start,peak_end,peak_index,tf_name,
                        scan_score,
                        matching_strand,
                        matching_relative start (within peak),
                        matching_relative_end (within peak),]
        """
        global scanner
        scanner = self.scanner

        maps = [[] for i in range(len(peaks_iter))]
        motif = [motif for motif, keep in zip(self.all_motifs, self.select) if keep]
        motifs_length = [m.length for m in motif]
        motifs_name = [m.name for m in motif]
        name2id = {name: i for i, name in enumerate(motifs_name)}

        peaks_iter = np.array(peaks_iter)
        if verbose:
            bar = trange(len(peaks_iter) * 2)
        else:
            bar = None

        pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        p_list = []
        for peak_idx, peak in enumerate(peaks_iter):
            chr = peak[0]
            start = int(peak[1])
            end = int(peak[2])
            seq = self.genome_seq[chr][start:end].seq.upper()

            if len(peaks_iter) == 1:
                # When we only need to scan once
                return scan_func(
                    seq,
                    motifs_length,
                    motifs_name,
                    chr,
                    start,
                    end,
                    peak_idx,
                    self.tfs,
                    clean,
                    split_tfs,
                    strand,
                )

            p_list.append(
                pool.submit(
                    scan_func,
                    seq,
                    motifs_length,
                    motifs_name,
                    chr,
                    start,
                    end,
                    peak_idx,
                    self.tfs,
                    clean,
                    split_tfs,
                    strand,
                )
            )

            if len(p_list) >= (self.n_jobs * 10):
                self.collect_child_process(
                    p_list,
                    verbose,
                    bar,
                    count,
                    motifs_length,
                    name2id,
                    maps,
                    break_on_min_jobs=True,
                )

            if verbose:
                bar.update(1)

        self.collect_child_process(
            p_list, verbose, bar, count, motifs_length, name2id, maps, break_on_min_jobs=False
        )
        if count:
            for i in range(len(maps)):
                if len(maps[i]) == 0:
                    maps[i] = np.zeros((1, len(motifs_length)))
        pool.shutdown(wait=True)

        if verbose:
            bar.close()

        if concat:
            return list(itertools.chain.from_iterable(maps))
        return maps

    def _prepare_moods_settings(self, jaspar_motifs, bg, pseduocount, pvalue=1e-5):
        """Find hits of list of jaspar_motifs in pyfasta object fasta, using the background distribution bg and
        pseudocount, significant to the give pvalue
        """
        start = time.time()
        pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        matrices_p = list(
            pool.map(
                partial(
                    _jaspar_to_moods_matrix,
                    bg=bg,
                    pseudocount=pseduocount,
                    mode=self.mode,
                ),
                jaspar_motifs,
                chunksize=100,
            )
        )
        matrices_m = list(pool.map(MOODS.tools.reverse_complement, matrices_p, chunksize=100))
        # print('get_mtx', time.time() - start)
        start = time.time()

        thresholds_p = list(
            pool.map(
                partial(thresholds_from_p, bg=bg, pvalue=pvalue),
                matrices_p,
                chunksize=100,
            )
        )
        thresholds_m = (
            thresholds_p
            if self.mode == "motifmatchr"
            else list(
                pool.map(
                    partial(thresholds_from_p, bg=bg, pvalue=pvalue),
                    matrices_m,
                    chunksize=100,
                )
            )
        )
        # thresholds_p = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices_p]
        # thresholds_m = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices_m]
        # print('get_thresholds', time.time() - start)
        start = time.time()
        # print (thresholds_p, thresholds_m)
        pool.shutdown(wait=True)
        # self.pool = ProcessPoolExecutor(max_workers=self.n_jobs)
        return (
            np.array(matrices_p, dtype=object),
            np.array(thresholds_p, dtype=object),
            np.array(matrices_m, dtype=object),
            np.array(thresholds_m, dtype=object),
        )


def figr_motif_name(name):
    return name.split("_")[2]


def jaspar_motif_name(name):
    return name.split("\t")[-1]


def cisbp_motif_name(name):
    return name.split(" ")[-1]


def FigR_Human_Motifs(genome: genome.Genome, bg, **kwargs):
    """
    Initialize and return a Motifs object using the FigR human motifs.

    Parameters
    ----------
    genome : scp.genome.Genome
        The Genome object containing the reference genome sequence.
    bg : str or tuple
        The background nucleotide frequency. If set to "genome", the function will calculate the background frequency
        from the entire genome sequence. Otherwise, it should be a tuple of 4 values representing the background frequency
        for A, C, G, and T nucleotides, respectively.

    Returns
    -------
    Motifs
        An instance of the Motifs class initialized with the FigR human motifs, the reference genome sequence, the
        background frequency, and the motif name function.
    """
    if bg == "genome":
        bg = genome.bg
    return Motifs(
        FigR_motifs_human, genome.fetch_fa(), bg, motif_name_func=figr_motif_name, **kwargs
    )


def FigR_Mouse_Motifs(genome: genome.Genome, bg, **kwargs):
    """
    Initialize and return a Motifs object using the FigR mouse motifs.

    Parameters
    ----------
    genome : scp.genome.Genome
        The Genome object containing the reference genome sequence.
    bg : str or tuple
        The background nucleotide frequency. If set to "genome", the function will calculate the background frequency
        from the entire genome sequence. Otherwise, it should be a tuple of 4 values representing the background frequency
        for A, C, G, and T nucleotides, respectively.

    Returns
    -------
    Motifs
        An instance of the Motifs class initialized with the FigR mouse motifs, the reference genome sequence, the
        background frequency, and the motif name function.
    """
    if bg == "genome":
        bg = genome.bg
    return Motifs(
        FigR_motifs_mouse, genome.fetch_fa(), bg, motif_name_func=figr_motif_name, **kwargs
    )


def JASPAR2022_core_Motifs(genome: genome.Genome, bg, **kwargs):
    """
    Initialize and return a Motifs object using the JASPAR2022 core motifs.

    Parameters
    ----------
    genome : genome.Genome
        The Genome object containing the reference genome sequence.
    bg : str or tuple
        The background nucleotide frequency. If set to "genome", the function will calculate the background frequency
        from the entire genome sequence. Otherwise, it should be a tuple of 4 values representing the background frequency
        for A, C, G, and T nucleotides, respectively.
    kwargs : dict
        Additional keyword arguments to be passed to the Motifs class constructor.

    Returns
    -------
    Motifs
        An instance of the Motifs class initialized with the JASPAR2022 core motifs, the reference genome sequence, the
        background frequency, and the motif name function.
    """
    if bg == "genome":
        bg = genome.bg
    return Motifs(
        JASPAR2022_core(), genome.fetch_fa(), bg, motif_name_func=jaspar_motif_name, **kwargs
    )


def CisBP_Human_Motifs(genome: genome.Genome, bg, **kwargs):
    """
    CisBP Human motifs function to initialize and return a Motifs object using the CisBP Human motifs.

    Parameters
    ----------
    genome : genome.Genome
        The Genome object containing the reference genome sequence.
    bg : str or tuple
        The background nucleotide frequency. If set to "genome", the function will calculate the background frequency
        from the entire genome sequence. Otherwise, it should be a tuple of 4 values representing the background frequency
        for A, C, G, and T nucleotides, respectively.
    kwargs : dict
        Additional keyword arguments to be passed to the Motifs class constructor.

    Returns
    -------
    Motifs
        An instance of the Motifs class initialized with the CisBP Human motifs, the reference genome sequence, the
        background frequency, and the motif name function.
    """
    if bg == "genome":
        bg = genome.bg
    return Motifs(CisBP_Human(), genome.fetch_fa(), bg, motif_name_func=cisbp_motif_name, **kwargs)
