import multiprocessing as mpl
import os
import sys
from functools import partial

import numpy as np
import pandas
import pyBigWig
import pyfaidx
import torch
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

try:
    from scprinter.utils import DNA_one_hot
except ImportError:
    from ..utils import DNA_one_hot


class seq2PRINTDataset(torch.utils.data.Dataset):
    """A general Data generator for training and evaluation.
       It takes in a bigwig, a list of summits, and some parameters specifying the length of the sequences
       the length of outputs etc. and generate training data on the fly.

    Parameters
    ----------
    signal: str
        Path to the bigwig file.
    ref_seq: str
        Path to the reference genome fasta file.
    summits: pandas.DataFrame
        A dataframe containing the summits of the peaks.
    DNA_window: int
        Length of the input DNA sequence. The sequence will be centered at the summit.
    signal_window: int
        Length of the output signal track. The signal sequence will be centered at the summit.
    max_jitter: int
        Maximum jitter to apply to the sequences.
    min_counts: int | None
        Minimum coverage to consider a peak to be used as training data
    max_counts: int | None
        Maximum coverage to consider a peak to be used as training data
    cached: bool
        Whether to cache the data first. If True, the data will be cached first before training. Otherwise, the data will be generated on the fly.
    lazy_cache: bool
        Whether to cache the data lazily. If True, the data will be cached when it's being indexed. So slow for the first time, but faster for the next time.
    reverse_compliment: bool
        Whether to reverse compliment the sequences.
    device: str
        The device to use for training.
    initialize: bool
        Whether to initialize the dataset. If False, the dataset will be initialized later, usually used as a placeholder
    """

    def __init__(
        self,
        signals,
        ref_seq,
        summits,
        DNA_window,
        signal_window,
        max_jitter,
        min_counts,
        max_counts,
        cached=False,
        lazy_cache=False,
        reverse_compliment=False,
        device="cpu",
        initialize=True,
        verbose=True,
    ):
        self.signal_paths = signals
        self.signals = [pyBigWig.open(signal) for signal in signals]
        self.ref_seq_path = ref_seq
        self.ref_seq = pyfaidx.Fasta(ref_seq)
        self.chrom_size = {k: len(v[:].seq) for k, v in self.ref_seq.items()}
        self.max_jitter = max_jitter
        self.min_counts = min_counts
        self.max_counts = max_counts
        self.reverse_compliment = reverse_compliment
        self.DNA_window = DNA_window
        self.signal_window = signal_window
        self.DNA_flank = DNA_window // 2
        self.signal_flank = signal_window // 2
        self.max_flank = max(self.DNA_flank, self.signal_flank)
        self.cached = cached
        self.device = device
        self.summits = summits
        self.lazy_cache = lazy_cache
        if self.min_counts is None:
            self.min_counts = -1
        if self.max_counts is None:
            self.max_counts = 1e16
        self.verbose = verbose
        if initialize:
            if verbose:
                print("input summits", len(summits))
            summits_valid = np.array(
                [
                    self.validate_loci(chrom, summit)
                    for chrom, summit in zip(
                        tqdm(summits.iloc[:, 0], desc="validating loci", disable=not verbose),
                        summits.iloc[:, 1],
                    )
                ]
            )
            if verbose:
                print("valid summits after trimming edges", np.sum(summits_valid))
            self.summits = summits.loc[summits_valid].copy().reset_index(drop=True)

            coverage = np.array(
                [
                    self.fetch_cov(chrom, summit)
                    for chrom, summit in zip(
                        tqdm(
                            self.summits.iloc[:, 0], desc="fetching coverage", disable=not verbose
                        ),
                        self.summits.iloc[:, 1],
                    )
                ]
            )
            if verbose:
                print(coverage.shape)
                print(
                    "coverage min max",
                    coverage.min(axis=-1).min(),
                    coverage.max(axis=-1).max(),
                )
            summits_valid = (coverage.min(axis=-1) > self.min_counts) & (
                coverage.max(axis=-1) < self.max_counts
            )
            if verbose:
                print("valid summits after min/max count filter", np.sum(summits_valid))
            self.summits = self.summits.loc[summits_valid]
            self.coverage = coverage[summits_valid]
            self.summits = np.array(self.summits)
            # When doing lazy caching, we set everything as None
            if lazy_cache:
                self.cache_seqs = None
                self.cache_signals = None
                self.uncache_index = set(np.arange(len(self.summits)))

            if self.cached:
                self.cache_seqs = []
                self.cache_signals = []
                for chrom, summit in tqdm(
                    self.summits[:, :2], desc="Caching sequences", disable=not verbose
                ):
                    DNA, signal = self.fetch_loci(chrom, summit)
                    self.cache_seqs.append(DNA)
                    self.cache_signals.append(signal)
                self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
                self.cache_signals = torch.stack(self.cache_signals, dim=0)

    def change_jitter(self, max_jitter):
        """
        Change the jitter of the dataset.

        Parameters
        ----------
        max_jitter: int
            The new jitter to apply to the sequences.

        Returns
        -------

        """

        # The change of the jitter would only affect some of the loci being valid or not

        self.max_jitter = max_jitter
        summits_valid = np.array(
            [
                self.validate_loci(chrom, summit)
                for chrom, summit in zip(
                    tqdm(self.summits[:, 0], desc="validating loci", disable=not self.verbose),
                    self.summits[:, 1],
                )
            ]
        )
        print("valid summits after trimming edges", np.sum(summits_valid))
        self.summits = self.summits[summits_valid]

        self.coverage = np.array(
            [
                self.fetch_cov(chrom, summit)
                for chrom, summit in zip(
                    tqdm(self.summits[:, 0], desc="fetching coverage", disable=not self.verbose),
                    self.summits[:, 1],
                )
            ]
        )

        if self.cached:
            self.cache(force=True)

    def cache(self, force=False):
        """
        Cache the data.

        Parameters
        ----------
        force: bool
            Whether to force the re-caching, when sth like max_jitter is changed, this is necessary.

        Returns
        -------

        """
        if self.cached and not force:
            print("Already cached")
            return

        self.cached = True
        self.cache_seqs = []
        self.cache_signals = []
        for chrom, summit in tqdm(
            self.summits[:, :2], desc="Caching sequences", disable=not self.verbose
        ):
            DNA, signal = self.fetch_loci(chrom, summit)
            self.cache_seqs.append(DNA)
            self.cache_signals.append(signal)
        self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
        self.cache_signals = torch.stack(self.cache_signals, dim=0)

    def append(self, dataset):
        self.summits = np.concatenate([self.summits, dataset.summits])
        self.coverage = np.concatenate([self.coverage, dataset.coverage])
        if self.cached or dataset.cached:
            self.cache()
            dataset.cache()
            self.cache_seqs = torch.cat([self.cache_seqs, dataset.cache_seqs])
            self.cache_signals = torch.cat([self.cache_signals, dataset.cache_signals])

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        return len(self.summits)

    def apply_jitter(self, dna, signal):
        jitter = (
            0 if self.max_jitter == 0 else np.random.default_rng().integers(self.max_jitter * 2)
        )
        return (
            dna[:, jitter : jitter + self.DNA_window],
            signal[:, jitter : jitter + self.signal_window],
        )

    def __getitem__(self, idx):
        reshape = False
        if self.cached:
            dnas, signals = (
                self.cache_seqs[idx].to(self.device),
                self.cache_signals[idx].to(self.device),
            )
            if len(dnas.shape) == 2:
                reshape = True
                dnas = dnas[None]
                signals = signals[None]

        else:
            # print (self.summits)
            summits = self.summits[idx][..., :2]
            if len(summits.shape) == 1:
                summits = summits[None]
                reshape = True
            dnas, signals = [], []
            for summit in summits:
                chrom, summit = summit
                dna, signal = self.fetch_loci(chrom, summit, device=self.device)

                dnas.append(dna)
                signals.append(signal)
            dnas = torch.stack(dnas, dim=0)
            signals = torch.stack(signals, dim=0)
            if self.lazy_cache:
                if self.cache_seqs is None:
                    shapes = list(dnas.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_seqs = torch.zeros(shapes, dtype=torch.float32, device=self.device)
                    shapes = list(signals.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_signals = torch.zeros(
                        shapes, dtype=torch.float32, device=self.device
                    )

                self.cache_seqs[idx] = dnas
                self.cache_signals[idx] = signals
                try:
                    self.uncache_index.remove(idx)
                except:
                    for i in idx:
                        self.uncache_index.remove(i)

                if len(self.uncache_index) == 0:
                    print("finish caching")
                    self.cached = True

        dnas_final, signals_final = [], []
        for dna, signal in zip(dnas, signals):
            dna, signal = self.apply_jitter(dna, signal)
            if self.reverse_compliment:
                if np.random.default_rng().random() > 0.5:
                    dna = dna.flip(dims=(0, 1))
                    signal = signal.flip(dims=(1,))

            dnas_final.append(dna)
            signals_final.append(signal)

        dnas, signals = torch.stack(dnas_final, dim=0), torch.stack(signals_final, dim=0)
        if reshape:
            dnas = dnas[0]
            signals = signals[0]
        dnas = dnas.float()

        return (dnas.float(), signals)

    def downsample(self, num):
        """
        Downsample the dataset to a smaller size.

        Parameters
        ----------
        num: int
            The number of loci to downsample to.

        Returns
        -------

        """
        idx = torch.as_tensor(np.random.permutation(len(self.summits))[:num]).to(self.device)
        summits = self.summits[idx]
        coverage = self.coverage[idx]
        if self.cached:
            cache_seqs = self.cache_seqs[idx]
            cache_signals = self.cache_signals[idx]
        downsampled_dataset = seq2PRINTDataset(
            signals=self.signal_paths,
            ref_seq=self.ref_seq_path,
            summits=summits,
            DNA_window=self.DNA_window,
            signal_window=self.signal_window,
            max_jitter=self.max_jitter,
            min_counts=self.min_counts,
            max_counts=self.max_counts,
            cached=self.cached,
            reverse_compliment=self.reverse_compliment,
            initialize=False,
            device=self.device,
        )
        downsampled_dataset.coverage = coverage
        downsampled_dataset.summits = summits
        if self.cached:
            downsampled_dataset.cache_seqs = cache_seqs
            downsampled_dataset.cache_signals = cache_signals
        return downsampled_dataset

    def filter_by_coverage(self, min_coverage, max_coverage):
        valid = (self.coverage.sum(axis=-1) > min_coverage) & (
            self.coverage.sum(axis=-1) < max_coverage
        )
        self.summits = self.summits[valid]
        self.coverage = self.coverage[valid]
        self.uncache_index = set(np.arange(len(self.summits)))
        if self.cached:
            self.cache_seqs = self.cache_seqs[valid]
            self.cache_signals = self.cache_signals[valid]

    def fetch_cov(self, chrom, summit):
        cov = [
            np.nansum(
                signal.values(
                    chrom,
                    summit - self.signal_flank,
                    summit + self.signal_flank,
                    numpy=True,
                )
            )
            for signal in self.signals[:-1]  # last one is usually the bias track
        ]
        return np.array(cov)

    def validate_loci(self, chrom, summit):
        if summit - self.max_flank - self.max_jitter < 0:
            return False
        if summit + self.max_flank + self.max_jitter >= self.chrom_size[chrom]:
            return False

        return True

    def fetch_loci(self, chrom, summit, device="cpu"):
        """Fetch the DNA sequence and the signal track for a given locus.
        Parameters
        ----------
        chrom: str
            Chromosome name.
        start: int
            Start position of the locus.
        end: int
            End position of the locus.

        Returns
        -------
        tuple
            A tuple of two torch.Tensor objects: DNA sequence and signal track.
        """
        DNA = DNA_one_hot(
            self.ref_seq[chrom][
                summit
                - self.DNA_flank
                - self.max_jitter : summit
                + self.DNA_flank
                + self.max_jitter
            ].seq.upper(),
            device=device,
        )
        signal = np.zeros((len(self.signals), (self.signal_flank + self.max_jitter) * 2))
        for i, signal_file in enumerate(self.signals):
            try:
                signal[i, :] = np.nan_to_num(
                    signal_file.values(
                        chrom,
                        summit - self.signal_flank - self.max_jitter,
                        summit + self.signal_flank + self.max_jitter,
                    )
                )
            except Exception as e:
                print(
                    "signal error",
                    e,
                    chrom,
                    summit,
                    summit - self.signal_flank - self.max_jitter,
                    summit + self.signal_flank + self.max_jitter,
                )
                raise EOFError
        signal = torch.tensor(signal, dtype=torch.float32, device=device)
        return DNA, signal


def fetch_cov_insertion(summit, group2idx):
    """
    Fetch the coverages for a given summit.
    Parameters
    ----------
    summit: tuple
        A tuple containing the chromosome, start, and end of the summit.
    group2idx: lists
        lists of indices for the groups.

    Returns
    -------

    """
    global global_insertion_dict
    chrom, start, end = summit
    signal = global_insertion_dict[chrom][:, start:end].sum(axis=-1)
    coverages = np.zeros((len(group2idx),))
    for j in range(len(group2idx)):
        coverages[j] = signal[group2idx[j].astype("int")].sum()

    return coverages


class scseq2PRINTDataset(torch.utils.data.Dataset):
    """
    A general Data generator for training and evaluation for the LoRA version of seq2PRINT.

    Parameters
    ----------
    insertion_dict: dict
        A dictionary containing the insertion profiles. keys are chr names, values are sparse matrix of single cell x genome positions.
    bias: str
        Path to the bias track bigwig file.
    group2idx: list[list]
        A list of lists containing the indices of the cells in each group.
    ref_seq: str
        Path to the reference genome fasta file.
    summits: pandas.DataFrame
        A dataframe containing the summits of the peaks.
    DNA_window: int
        Length of the input DNA sequence. The sequence will be centered at the summit.
    signal_window: int
        Length of the output signal track. The signal sequence will be centered at the summit.
    max_jitter: int
        Maximum jitter to apply to the sequences.
    cached: bool
        Whether to cache the data first. If True, the data will be cached first before training. Otherwise, the data will be generated on the fly.
    lazy_cache: bool
        Whether to cache the data lazily. If True, the data will be cached when it's being indexed. So slow for the first time, but faster for the next time.
    reverse_compliment: bool
        Whether to reverse compliment the sequences.
    device: str
        The device to use for training.
    initialize: bool
        Whether to initialize the dataset. If False, the dataset will be initialized later, usually used as a placeholder
    coverages: np.array
        A numpy array containing the coverages for the summits.
    data_augmentation: bool
        Whether to apply data augmentation to the dataset. (including some peaks with really low insertions)
    mode: Literal["uniform", "peak"]
        The mode to sample group x peak for finetuning, "uniform" means uniformly sample the combination, "peak" means subset some peak first, then sample some groups, so there are contrasts of groups within same set of peaks
    cell_sample: int
        The number of cells to sample for each peak in the "peak" mode.

    """

    def __init__(
        self,
        insertion_dict,
        bias,
        group2idx,
        ref_seq,
        summits,
        DNA_window,
        signal_window,
        max_jitter,
        cached=False,
        lazy_cache=False,
        reverse_compliment=False,
        device="cpu",
        initialize=True,
        coverages=None,
        data_augmentation=False,
        mode="uniform",
        cell_sample=10,
        verbose=True,
    ):
        # set global var for easy access
        global global_insertion_dict
        global_insertion_dict = insertion_dict
        self.bias_path = bias
        self.bias = pyBigWig.open(bias)
        self.group2idx = group2idx
        self.ref_seq_path = ref_seq
        self.ref_seq = pyfaidx.Fasta(ref_seq)
        self.chrom_size = {k: len(v[:].seq) for k, v in self.ref_seq.items()}

        self.max_jitter = max_jitter
        self.reverse_compliment = reverse_compliment
        self.DNA_window = DNA_window
        self.signal_window = signal_window
        self.DNA_flank = DNA_window // 2
        self.signal_flank = signal_window // 2
        self.max_flank = max(self.DNA_flank, self.signal_flank)
        self.cached = cached
        self.device = device
        self.summits = summits
        self.lazy_cache = lazy_cache
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.cell_sample = cell_sample
        self.verbose = verbose
        if initialize:
            print("input summits", len(summits))
            summits_valid = np.array(
                [
                    self.validate_loci(chrom, summit)
                    for chrom, summit in zip(
                        tqdm(summits.iloc[:, 0], desc="validating loci", disable=not self.verbose),
                        summits.iloc[:, 1],
                    )
                ]
            )
            print("valid summits after trimming edges", np.sum(summits_valid))
            self.summits = summits.loc[summits_valid]
            self.summits = np.array(self.summits)

            self.idx = np.arange(len(self.summits) * len(self.group2idx))
            self.summit_idx = self.idx // len(self.group2idx)
            self.group_idx = self.idx % len(self.group2idx)

            regions = [
                self.summits[:, 0],
                self.summits[:, 1] - self.signal_flank - self.max_jitter,
                self.summits[:, 1] + self.signal_flank + self.max_jitter,
            ]
            regions = np.array(regions).T
            print(regions, regions.shape)
            regions[:, 1] = regions[:, 1].astype(int)
            regions[:, 2] = regions[:, 2].astype(int)

            if coverages is None:
                coverages = process_map(
                    # Fetch insertions
                    partial(fetch_cov_insertion, group2idx=self.group2idx),
                    regions,
                    max_workers=min(mpl.cpu_count(), 24),
                    chunksize=100,
                    desc="fetching coverages",
                )
                self.coverages = np.array(coverages).reshape((-1))
            else:
                self.coverages = coverages
                assert len(self.coverages) == len(self.summits) * len(
                    self.group2idx
                ), "coverages length mismatch"

            print(
                "coverages",
                self.coverages.shape,
                self.coverages.min(),
                self.coverages.max(),
            )
            self.normalized_coverages = self.coverages.reshape((-1, len(self.group2idx)))
            print(
                self.normalized_coverages.shape,
                self.normalized_coverages.min(),
                self.normalized_coverages.max(),
            )
            self.normalized_coverages = np.log1p(
                self.normalized_coverages
                / (self.normalized_coverages.mean(axis=1)[:, None] + 1e-16)
            )  # normalized coverage is normalized by the mean in each pseudobulk
            print(
                "normalized coverages",
                self.normalized_coverages.shape,
                self.normalized_coverages.min(),
                self.normalized_coverages.max(),
            )
            self.normalized_coverages = torch.from_numpy(
                self.normalized_coverages.reshape((-1))
            ).float()
            self.mask = (
                self.coverages > 10
            )  # at least 10 insertions to be considered. This is a hard threshold
            pos = self.idx[self.mask]
            neg = self.idx[~self.mask]
            if data_augmentation:
                ratio = 0.1
                sampled_neg = np.random.permutation(neg)[: int(len(pos) * ratio)]
                self.idx = np.concatenate([pos, sampled_neg])
                self.idx = self.idx
            else:
                self.idx = pos

            self.summit_idx = self.summit_idx[self.idx]
            self.group_idx = self.group_idx[self.idx]
            print("filtering from ", len(self.mask), "to ", len(self.idx))
            if lazy_cache:
                self.cache_seqs = None
                self.cache_bias = None
                self.uncache_index = set(np.arange(len(self.summits)))

            if self.cached:
                self.cache(force=True)

    def cache(self, force=False):
        if self.cached and not force:
            print("Already cached")
            return

        self.cached = True
        self.cache_seqs = []
        self.cache_bias = []
        for chrom, summit in tqdm(
            self.summits[:, :2], desc="Caching sequences", disable=not self.verbose
        ):
            DNA, bias = self.fetch_loci_dna_bias(chrom, summit)
            self.cache_seqs.append(DNA)
            self.cache_bias.append(bias)

        self.cache_seqs = torch.stack(self.cache_seqs, dim=0)
        self.cache_bias = torch.stack(self.cache_bias, dim=0)

    def to(self, device):
        self.device = device
        return self

    def __len__(self):
        if self.mode == "uniform":
            return len(self.idx)  # this is the number of training tuples
        elif self.mode == "peak":
            return len(self.summits)  # this is the number of peaks
        else:
            raise ValueError("mode not recognized")

    def apply_jitter(self, dna, signal):
        """
        Apply jitter to the sequences and signals

        Parameters
        ----------
        dna
        signal

        Returns
        -------

        """
        if self.max_jitter == 0:
            return dna, signal
        jitter = (
            0 if self.max_jitter == 0 else np.random.default_rng().integers(self.max_jitter * 2)
        )  # this random.default_rng() is necessary, don't change it due to np.random seed under multiprocessing / threading

        return (
            dna[:, jitter : jitter + self.DNA_window],
            signal[:, jitter : jitter + self.signal_window],
        )

    def getitem_by_summit_group(self, summit_idx, group_idx):
        """
        Get the training tuples by the summit index and group index.

        Parameters
        ----------
        summit_idx
        group_idx

        Returns
        -------

        """
        # DNA and bias bigwig first
        reshape = False
        if self.cached:
            dnas = self.cache_seqs[summit_idx].to(self.device)
            biases = self.cache_bias[summit_idx].to(self.device)
            if len(dnas.shape) == 2:
                reshape = True
                dnas = dnas[None]
                biases = biases[None]
        else:
            summits = self.summits[summit_idx][..., :2]
            if len(summits.shape) == 1:
                summits = summits[None]
                reshape = True
            dnas = []
            biases = []
            for summit in summits:
                chrom, summit = summit
                dna, bias = self.fetch_loci_dna_bias(chrom, summit, device=self.device)
                dnas.append(dna)
                biases.append(bias)

            dnas = torch.stack(dnas, dim=0)
            biases = torch.stack(biases, dim=0)
            if self.lazy_cache:
                if self.cache_seqs is None:
                    shapes = list(dnas.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_seqs = torch.zeros(shapes, dtype=torch.float32, device=self.device)
                    shapes = list(biases.shape)
                    if len(shapes) == 3:
                        shapes[0] = len(self.summits)
                    else:
                        shapes = [len(self.summits)] + shapes
                    self.cache_bias = torch.zeros(shapes, dtype=torch.float32, device=self.device)

                self.cache_seqs[summit_idx] = dnas.float()
                self.cache_bias[summit_idx] = biases
                try:
                    if summit_idx in self.uncache_index:
                        self.uncache_index.remove(summit_idx)
                except:
                    for i in summit_idx:
                        if i in self.uncache_index:
                            self.uncache_index.remove(i)

                if len(self.uncache_index) == 0:
                    print("finish caching")
                    self.cached = True

        # Now for signals, signals won't be cached because it would be too many
        signals = []
        summits = self.summits[summit_idx][..., :2]
        if len(summits.shape) == 1:
            summits = summits[None]
            group_idx = [group_idx]
            summit_idx = [summit_idx]
            reshape = True
        for summit, group in zip(summits, group_idx):
            chrom, summit = summit
            # still sparse, and is at single cell resolution
            signal = global_insertion_dict[chrom][
                :,
                summit
                - self.signal_flank
                - self.max_jitter : summit
                + self.signal_flank
                + self.max_jitter,
            ]
            # Then group at pseudobulk level
            signal = np.array(signal[self.group2idx[group]].sum(axis=0)).astype("float32")
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
            signals.append(signal)
        signals = torch.stack(signals, dim=0)
        # Now concatenate the bias to the signal
        signals = torch.concatenate([signals, biases], dim=1)

        dnas_final, signals_final = [], []
        for dna, signal in zip(dnas, signals):
            dna, signal = self.apply_jitter(dna, signal)
            if self.reverse_compliment:
                if np.random.default_rng().random() > 0.5:
                    dna = dna.flip(dims=(0, 1))
                    signal = signal.flip(dims=(1,))

            dnas_final.append(dna)
            signals_final.append(signal)

        dnas, signals = torch.stack(dnas_final, dim=0), torch.stack(signals_final, dim=0)
        if reshape:
            dnas = dnas[0]
            signals = signals[0]

        dnas = dnas.float()
        pos_mask = torch.as_tensor(
            self.mask[summit_idx * len(self.group2idx) + group_idx], dtype=torch.bool
        )
        return (
            dnas.float(),
            signals,
            torch.as_tensor(group_idx).long(),
            torch.as_tensor(summit_idx).long(),
            pos_mask,
        )

    def __getitem__(self, idx):

        if self.mode == "uniform":
            # At uniform mode, idx should be used directly to slice summit_idx and group_idx
            summit_idx = self.summit_idx[idx]
            group_idx = self.group_idx[idx]
            X, y, cell, peak, _ = self.getitem_by_summit_group(summit_idx, group_idx)
            coverage = self.normalized_coverages[self.idx[idx]]
            return X, y, cell, peak, coverage
        elif self.mode == "peak":
            # At peak mode, idx should be used to slice summits, then randomly sample some cells and do meshgrid
            summit_idx = idx
            rng = np.random.default_rng()
            group_idx = rng.integers(low=0, high=len(self.group2idx), size=self.cell_sample)
            grid1, grid2 = np.meshgrid(summit_idx, group_idx)
            combinations = np.column_stack((grid1.flatten(), grid2.flatten()))
            summit_idx, group_idx = combinations[:, 0], combinations[:, 1]

            X, y, cell, peak, pos_mask = self.getitem_by_summit_group(summit_idx, group_idx)
            coverage = self.normalized_coverages[summit_idx * len(self.group2idx) + group_idx]
            return X, y, cell, peak, pos_mask, coverage

    def validate_loci(self, chrom, summit):
        if summit - self.max_flank - self.max_jitter < 0:
            return False
        if summit + self.max_flank + self.max_jitter >= self.chrom_size[chrom]:
            return False

        return True

    def fetch_loci_dna_bias(self, chrom, summit, device="cpu"):
        """Fetch the DNA sequence and the signal track for a given locus.
        Parameters
        ----------
        chrom: str
            Chromosome name.
        start: int
            Start position of the locus.
        end: int
            End position of the locus.

        Returns
        -------
        tuple
            A tuple of two torch.Tensor objects: DNA sequence and signal track.
        """
        DNA = DNA_one_hot(
            self.ref_seq[chrom][
                summit
                - self.DNA_flank
                - self.max_jitter : summit
                + self.DNA_flank
                + self.max_jitter
            ].seq.upper(),
            device=device,
        )
        signal = np.nan_to_num(
            self.bias.values(
                chrom,
                summit - self.signal_flank - self.max_jitter,
                summit + self.signal_flank + self.max_jitter,
                numpy=True,
            ),
            nan=0.0,
        )[None]
        signal = torch.tensor(signal, dtype=torch.float32, device=device)

        return DNA, signal


class seq2PRINTDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset=None,
        batch_size=64,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=None,
        **kwargs,
    ):
        self.data_collection = None
        self.batch_size = batch_size
        if dataset is None:
            self.dataset = seq2PRINTDataset(**kwargs)
        elif type(dataset) is list:
            self.data_collection = dataset
            dataset_list = []
            for data, num in dataset:
                dataset_list.append(data.downsample(num))
            self.dataset = ConcatDataset(dataset_list)

        else:
            self.dataset = dataset
        self.shuffle = shuffle
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=lambda id: np.random.seed(id),
            collate_fn=collate_fn,
        )

    def resample(self):
        if self.data_collection is not None:
            dataset_list = []
            for data, num in self.data_collection:
                dataset_list.append(data.downsample(num))
            return seq2PRINTDataLoader(
                ConcatDataset(dataset_list),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle,
            )
        else:
            return self


def collate_fn_singlecell(batch):
    """A collate function for the single cell data.
    Parameters
    ----------
    batch: list
        A list of tuples of containing, DNA sequence, signal track, cell index, peak index, whether it's a positive samples and coverage.

    Returns
    -------
    tuple
        A tuple of torch.Tensor objects: DNA sequence, signal track, cell index, peak index, coverage.
    """

    X, y, cell, peak, pos_mask, coverage = list(zip(*batch))
    X = torch.concat(X, dim=0)
    y = torch.concat(y, dim=0)
    cell = torch.concat(cell, dim=0)[:, None]
    peak = torch.concat(peak, dim=0)[:, None]
    pos_mask = torch.concat(pos_mask, dim=0)
    idx = np.arange(len(y))
    pos = idx[pos_mask]
    neg = idx[~pos_mask]
    ratio = 0.1
    sampled_neg = np.random.permutation(neg)[: int(len(pos) * ratio)]
    idx = torch.as_tensor(np.random.permutation(np.concatenate([pos, sampled_neg])))
    y = y[idx]
    X = X[idx]
    cell = cell[idx]
    peak = peak[idx]
    return X, y, cell, peak, coverage
