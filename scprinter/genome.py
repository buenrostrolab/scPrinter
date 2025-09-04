# motivated by snapatac2.genome
from __future__ import annotations

import os.path
import pickle
from pathlib import Path

import gffutils
import h5py
import numpy as np
import pyBigWig
import torch
from pyfaidx import Fasta
from tqdm.auto import tqdm, trange

from .datasets import datasets, giverightstothegroup
from .utils import DNA_one_hot, get_stats_for_genome


class Genome:
    """
    Genome class
    It keep records of all the genome specific information
    including chrom sizes, gff gene annotation file,
    genome reference fasta file,
    precomputed Tn5 bias file, and background freq of ACGT

    Parameters
    ----------
    chrom_sizes: dict
       A dictionary of chromosome names and their lengths
       When setting as None, it will be inferred from the fasta file
    gff_file: str
       The path to the gff file
    fa_file: str
       The path to the fasta file
    bias_file: str
       The path to the bias h5 file
    bg: tuple
       A tuple of background frequencies of ACGT
       When setting as None, it will be inferred from the fasta file

    Returns
    -------
    None
    """

    def __init__(
        self,
        name: str,
        chrom_sizes: dict | None = None,
        gff_file: str | Path = "",
        fa_file: str | Path = "",
        bias_file: str | Path = "",
        blacklist_file: str | Path | None = None,
        bg: tuple | None = None,
        splits: list[dict] | None = None,
    ):

        self.name = name
        if chrom_sizes is None or bg is None:
            chrom_sizes_inferred, bg_inferred = get_stats_for_genome(fa_file)
            if chrom_sizes is None:
                self.chrom_sizes = chrom_sizes_inferred
            else:
                self.chrom_sizes = chrom_sizes
            if bg is None:
                self.bg = bg_inferred
            else:
                self.bg = bg
        else:
            self.chrom_sizes = chrom_sizes
            self.bg = bg
        self.gff_file = gff_file
        self.fa_file = fa_file
        self.bias_file = bias_file
        self.blacklist_file = blacklist_file
        self.bias_bw = None
        self.gff_db = None
        # self.gff_file = self.fetch_gff()
        # self.fa_file = self.fetch_fa()
        # self.bias_file = self.fetch_bias()
        # self.blacklist_file = self.fetch_blacklist()
        # self.bias_bw = self.fetch_bias_bw()
        # self.gff_db = self.fetch_gff_db()
        self.splits = splits

    def fetch_blacklist(self):
        """
        Fetch the blacklist file from the server if it is not present locally
        Returns
        -------

        """
        if self.blacklist_file is None:
            raise ValueError("No blacklist file provided")
        if not os.path.exists(self.blacklist_file):
            return str(
                datasets().fetch(
                    self.blacklist_file,
                    processor=giverightstothegroup,
                    progressbar=True,
                )
            )
        else:
            return self.blacklist_file

    def fetch_gff(self):
        """
        Fetch the gff file from the server if it is not present locally

        Returns
        -------
        str of the path to the local gff file
        """
        if not os.path.exists(self.gff_file):
            return str(
                datasets().fetch(self.gff_file, progressbar=True, processor=giverightstothegroup)
            )
        else:
            return self.gff_file

    def fetch_gff_db(self):
        """
        Fetch the gff file locally, if it is not present, create it

        Returns
        -------

        """
        gff = self.fetch_gff()
        if not os.path.exists(str(gff) + ".db"):
            print("Creating GFF database (Runs for new genome)")
            # Specifying the id_spec was necessary for gff files from NCBI.
            gffutils.create_db(
                gff,
                str(gff) + ".db",
                id_spec={"gene": "gene_name", "transcript": "transcript_id"},
                merge_strategy="create_unique",
            )
        return str(gff) + ".db"

    def fetch_fa(self):
        """
        Fetch the fasta file from the server if it is not present locally

        Returns
        -------
        str of the path to the local fasta file
        """
        if not os.path.exists(self.fa_file):
            return str(
                datasets().fetch(
                    self.fa_file,
                    processor=giverightstothegroup,
                    progressbar=True,
                )
            )
        else:
            return self.fa_file

    def fetch_seq(self, chrom, start, end):
        """
        Fetch the sequence from the fasta file

        Parameters
        ----------
        chrom: str
            The name of the chromosome
        start: int
            The start position of the sequence
        end: int
            The end position of the sequence

        Returns
        -------
        str of the sequence
        """
        if not hasattr(self, "fasta"):
            self.fasta = Fasta(self.fetch_fa())

        return self.fasta[chrom][start:end].seq

    def fetch_onehot_seq(self, chrom, start, end):
        """
        Fetch the onehot encoded sequence from the fasta file

        Parameters
        ----------
        chrom: str
            The name of the chromosome
        start: int
            The start position of the sequence
        end: int
            The end position of the sequence

        Returns
        -------
        np.array of the onehot encoded sequence
        """
        if not hasattr(self, "fasta"):
            self.fasta = Fasta(self.fetch_fa())

        return DNA_one_hot(self.fasta[chrom][start:end].seq.upper())

    def fetch_bias(self):
        """
        Fetch the bias file from the server if it is not present locally

        Returns
        -------
        str of the path to the local bias file
        """
        if not os.path.exists(self.bias_file):
            file = datasets().fetch(
                self.bias_file,
                processor=giverightstothegroup,
                progressbar=True,
            )
            if not isinstance(file, list):
                file = [file]
            for f in file:
                if ".h5" in f:
                    return str(f)
        return self.bias_file

    def fetch_bias_bw(self, verify=False):
        """
        Fetch the bias bigwig file locally, if it is not present, create it
        Returns
        -------

        """
        bias_file = self.fetch_bias()
        bias_bw = bias_file.replace(".h5", ".bw")
        if not os.path.exists(bias_bw):
            print("creating bias bigwig (runs for new bias h5 file)")
            with h5py.File(bias_file, "r") as dct:
                precomputed_bias = {chrom: np.array(dct[chrom]) for chrom in dct.keys()}
                bw = pyBigWig.open(bias_bw, "w")
                header = []
                for chrom in precomputed_bias:
                    sig = precomputed_bias[chrom]
                    length = sig.shape[-1]
                    header.append((chrom, length))
                bw.addHeader(header, maxZooms=0)
                for chrom in tqdm(precomputed_bias):
                    sig = precomputed_bias[chrom]
                    bw.addEntries(
                        str(chrom),
                        np.arange(len(sig)),
                        values=sig.astype("float"),
                        span=1,
                    )
                bw.close()
            print("bias bigwig created")
        else:
            if verify:
                pass_flag = True
                with h5py.File(bias_file, "r") as dct:
                    try:
                        with pyBigWig.open(bias_bw, "r") as bw:
                            for chrom in dct.keys():
                                bias = np.array(dct[chrom][:])
                                length = bias.shape[0]
                                sig = bw.values(chrom, 0, length)
                    except Exception as e:
                        print(f"Error: {e}")
                        print(chrom, length)
                        pass_flag = False

                if not pass_flag:
                    print("bias bigwig corrupted, removing it now")
                    os.remove(bias_bw)
                    print(
                        "recreating bias bigwig, but if you repeatedly see this message, stop and report an issue"
                    )
                    self.fetch_bias_bw(verify=True)

        return bias_bw

    def fetch_cpg_index(self):
        fa_path = str(self.fetch_fa())
        cpg_index = fa_path + "_cpg_index.pkl"
        if not os.path.exists(cpg_index):
            print("Building CpG index...")
            cpg_index_all, cpg_num, cpg_coord = build_CpG_index(self)
            with open(cpg_index, "wb") as f:
                pickle.dump((cpg_index_all, cpg_num, cpg_coord), f)
        else:
            with open(cpg_index, "rb") as f:
                cpg_index_all, cpg_num, cpg_coord = pickle.load(f)
        return cpg_index_all, cpg_num, cpg_coord


def build_CpG_index(genome):
    fa_path = Path(genome.fetch_fa())
    fasta = Fasta(fa_path, as_raw=True)
    cpg_index_all = {}
    cpg_num = {}
    chroms_all = set(genome.chrom_sizes.keys())
    cpg_coord = {}
    for chrom in chroms_all:
        cpg_index_all[chrom] = {}
        cpg_coord[chrom] = []
        seq = fasta[chrom][:].upper()
        count = 0
        for i in trange(len(seq) - 1):
            if seq[i : i + 2] == "CG":
                cpg_index_all[chrom][i + 1] = count
                cpg_coord[chrom].append(i + 1)
                count += 1
        cpg_num[chrom] = count
    fasta.close()
    return cpg_index_all, cpg_num, cpg_coord


def predict_genome_tn5_bias(
    fa_file,
    save_name,
    tn5_model=None,
    context_radius=50,
    batch_size=1000,
    device="cpu",
):
    """

    Parameters
    ----------
    fa_file: str
        path to the fasta file for the custome genome
    save_name: str
        path to save the calculated Tn5 bias
    tn5_model: str
        path to the pre-trained Tn5 model, if not provided, the default model will be used
    context_radius: int
        the context radius for the Tn5 model, default is 50
    batch_size: int
        batch size for the prediction, default is 1000
    device: str
        the device to run the model, default is "cpu", but if you have a GPU, you can set it to "cuda" to make it faster
    Returns
    -------

    """
    batch_size = max(batch_size, 101)
    if tn5_model is None:
        tn5_model = datasets().fetch("Tn5_NN_model.pt", processor=giverightstothegroup)
    tn5_model = torch.jit.load(tn5_model, map_location=torch.device("cpu")).to(device)
    ref_seq = Fasta(fa_file)
    chrom_size = {k: len(v[:].seq) for k, v in ref_seq.items()}
    with h5py.File(save_name, "w") as h5f:
        bar = trange(len(chrom_size), desc=f"Predicting Tn5 bias")
        for chrom in chrom_size:
            bar.set_description(f"Predicting Tn5 bias for {chrom}")
            seq = ref_seq[chrom][:].seq.upper()
            # padding for context radius
            if len(seq) < 100:
                continue
            seq = "N" * 50 + seq + "N" * 50
            seq_len = len(seq)
            # Initial number of batches based on desired batch_size
            num_batch = max(seq_len // batch_size, 1)
            stride = 2 * context_radius
            bs = valid_batch_size(seq_len, num_batch, stride)
            with torch.no_grad():
                bias_all = []
                for start in range(0, len(seq), bs - stride):
                    local_seq = DNA_one_hot(seq[start : start + bs], device=device).float()[None]
                    try:
                        bias = tn5_model(local_seq).detach().cpu().numpy().reshape((-1))
                    except:
                        print(local_seq.shape, bs)
                        raise EOFError
                    bias = np.power(10, (bias - 0.5) * 2) - 0.01
                    bias_all.append(bias)
                bias_all = np.concatenate(bias_all, axis=0).reshape((-1))
                h5f.create_dataset(chrom, data=bias_all)
            bar.update(1)
        bar.close()


def valid_batch_size(seq_len, num_batch, stride):
    init_bs = (seq_len + num_batch + stride) // num_batch
    lengths = []
    for start in range(0, seq_len, init_bs - stride):
        lengths.append(min(start + init_bs, seq_len) - start)
    if np.min(lengths) < stride + 1:
        return valid_batch_size(seq_len, num_batch - 1, stride)
    else:
        return init_bs


hg38_splits = [None] * 5
hg38_splits[0] = {
    "test": ["chr1", "chr3", "chr6"],
    "valid": ["chr8", "chr20"],
    "train": [
        "chr2",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr21",
        "chr22",
        "chrX",
        "chrY",
    ],
}
hg38_splits[1] = {
    "test": ["chr2", "chr8", "chr9", "chr16"],
    "valid": ["chr12", "chr17"],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr11",
        "chr13",
        "chr14",
        "chr15",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
        "chrY",
    ],
}
hg38_splits[2] = {
    "test": ["chr4", "chr11", "chr12", "chr15", "chrY"],
    "valid": ["chr22", "chr7"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr10",
        "chr13",
        "chr14",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chrX",
    ],
}
hg38_splits[3] = {
    "test": ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
    "valid": ["chr6", "chr21"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr7",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr13",
        "chr15",
        "chr16",
        "chr17",
        "chr19",
        "chrX",
        "chrY",
    ],
}
hg38_splits[4] = {
    "test": ["chr7", "chr13", "chr17", "chr19", "chr21", "chrX"],
    "valid": ["chr10", "chr18"],
    "train": [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr12",
        "chr14",
        "chr15",
        "chr16",
        "chr20",
        "chr22",
        "chrY",
    ],
}

mm10_splits = [None] * 5
mm10_splits[0] = {
    "test": ["chr1", "chr6", "chr12", "chr13", "chr16"],
    "valid": ["chr8", "chr11", "chr18", "chr19", "chrX"],
    "train": [
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr7",
        "chr9",
        "chr10",
        "chr14",
        "chr15",
        "chr17",
    ],
}
mm10_splits[1] = {
    "test": ["chr2", "chr7", "chr10", "chr14", "chr17"],
    "valid": ["chr5", "chr9", "chr13", "chr15", "chrY"],
    "train": [
        "chr1",
        "chr3",
        "chr4",
        "chr6",
        "chr8",
        "chr11",
        "chr12",
        "chr16",
        "chr18",
        "chr19",
        "chrX",
    ],
}
mm10_splits[2] = {
    "test": ["chr3", "chr8", "chr13", "chr15", "chr17"],
    "valid": ["chr2", "chr9", "chr11", "chr12", "chrY"],
    "train": [
        "chr1",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr10",
        "chr14",
        "chr16",
        "chr18",
        "chr19",
        "chrX",
    ],
}
mm10_splits[3] = {
    "test": ["chr4", "chr9", "chr11", "chr14", "chr19"],
    "valid": ["chr1", "chr7", "chr12", "chr13", "chrY"],
    "train": [
        "chr2",
        "chr3",
        "chr5",
        "chr6",
        "chr8",
        "chr10",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chrX",
    ],
}
mm10_splits[4] = {
    "test": ["chr5", "chr10", "chr12", "chr16", "chrY"],
    "valid": ["chr3", "chr7", "chr14", "chr15", "chr18"],
    "train": [
        "chr1",
        "chr2",
        "chr4",
        "chr6",
        "chr8",
        "chr9",
        "chr11",
        "chr13",
        "chr17",
        "chr19",
        "chrX",
    ],
}
GRCh38_v1 = Genome(
    "hg38",
    {
        "chr1": 248956422,
        "chr2": 242193529,
        "chr3": 198295559,
        "chr4": 190214555,
        "chr5": 181538259,
        "chr6": 170805979,
        "chr7": 159345973,
        "chr8": 145138636,
        "chr9": 138394717,
        "chr10": 133797422,
        "chr11": 135086622,
        "chr12": 133275309,
        "chr13": 114364328,
        "chr14": 107043718,
        "chr15": 101991189,
        "chr16": 90338345,
        "chr17": 83257441,
        "chr18": 80373285,
        "chr19": 58617616,
        "chr20": 64444167,
        "chr21": 46709983,
        "chr22": 50818468,
        "chrX": 156040895,
        "chrY": 57227415,
    },
    "gencode_v41_GRCh38.gff3.gz",
    "gencode_v41_GRCh38.fa.gz",
    "hg38Tn5Bias.tar.gz",
    "hg38-blacklist.v2.bed.gz",
    (0.29518279760588795, 0.20390602956403897, 0.20478356895235347, 0.2961276038777196),
    hg38_splits,
)
GRCh38 = Genome(
    "hg38",
    {
        "chr1": 248956422,
        "chr2": 242193529,
        "chr3": 198295559,
        "chr4": 190214555,
        "chr5": 181538259,
        "chr6": 170805979,
        "chr7": 159345973,
        "chr8": 145138636,
        "chr9": 138394717,
        "chr10": 133797422,
        "chr11": 135086622,
        "chr12": 133275309,
        "chr13": 114364328,
        "chr14": 107043718,
        "chr15": 101991189,
        "chr16": 90338345,
        "chr17": 83257441,
        "chr18": 80373285,
        "chr19": 58617616,
        "chr20": 64444167,
        "chr21": 46709983,
        "chr22": 50818468,
        "chrX": 156040895,
        "chrY": 57227415,
    },
    "gencode_v41_GRCh38.gff3.gz",
    "gencode_v41_GRCh38.fa.gz",
    "hg38_bias_v2.h5",
    "hg38-blacklist.v2.bed.gz",
    (0.29518279760588795, 0.20390602956403897, 0.20478356895235347, 0.2961276038777196),
    hg38_splits,
)
hg38_v1 = GRCh38_v1
hg38_v2 = GRCh38
hg38 = GRCh38


GRCm38_v1 = Genome(
    "mm10",
    {
        "chr1": 195471971,
        "chr2": 182113224,
        "chr3": 160039680,
        "chr4": 156508116,
        "chr5": 151834684,
        "chr6": 149736546,
        "chr7": 145441459,
        "chr8": 129401213,
        "chr9": 124595110,
        "chr10": 130694993,
        "chr11": 122082543,
        "chr12": 120129022,
        "chr13": 120421639,
        "chr14": 124902244,
        "chr15": 104043685,
        "chr16": 98207768,
        "chr17": 94987271,
        "chr18": 90702639,
        "chr19": 61431566,
        "chrX": 171031299,
        "chrY": 91744698,
    },
    "gencode_vM25_GRCm38.gff3.gz",
    "gencode_vM25_GRCm38.fa.gz",
    "mm10Tn5Bias.tar.gz",
    "mm10-blacklist.v2.bed.gz",
    (0.29149763779592625, 0.2083275235867118, 0.20834346947899296, 0.291831369138369),
    mm10_splits,
)

GRCm38 = Genome(
    "mm10",
    {
        "chr1": 195471971,
        "chr2": 182113224,
        "chr3": 160039680,
        "chr4": 156508116,
        "chr5": 151834684,
        "chr6": 149736546,
        "chr7": 145441459,
        "chr8": 129401213,
        "chr9": 124595110,
        "chr10": 130694993,
        "chr11": 122082543,
        "chr12": 120129022,
        "chr13": 120421639,
        "chr14": 124902244,
        "chr15": 104043685,
        "chr16": 98207768,
        "chr17": 94987271,
        "chr18": 90702639,
        "chr19": 61431566,
        "chrX": 171031299,
        "chrY": 91744698,
    },
    "gencode_vM25_GRCm38.gff3.gz",
    "gencode_vM25_GRCm38.fa.gz",
    "mm10_bias_v2.h5",
    "mm10-blacklist.v2.bed.gz",
    (0.29149763779592625, 0.2083275235867118, 0.20834346947899296, 0.291831369138369),
    mm10_splits,
)


mm10_v1 = GRCm38_v1

mm10_v2 = GRCm38
mm10 = GRCm38

mm39_v1 = Genome(
    name="mm39",
    chrom_sizes={
        "chr1": 195154279,
        "chr2": 181755017,
        "chrX": 169476592,
        "chr3": 159745316,
        "chr4": 156860686,
        "chr5": 151758149,
        "chr6": 149588044,
        "chr7": 144995196,
        "chr10": 130530862,
        "chr8": 130127694,
        "chr14": 125139656,
        "chr9": 124359700,
        "chr11": 121973369,
        "chr13": 120883175,
        "chr12": 120092757,
        "chr15": 104073951,
        "chr16": 98008968,
        "chr17": 95294699,
        "chrY": 91455967,
        "chr18": 90720763,
        "chr19": 61420004,
    },
    gff_file="gencode_vM30_GRCm39.gff3.gz",
    fa_file="gencode_vM30_GRCm39.fa.gz",
    bias_file="mm39Tn5Bias.h5",
    blacklist_file="mm39.excluderanges.bed",
    bg=(0.29149562991965305, 0.20831919210041502, 0.20833700706759098, 0.29184817091234094),
    splits=mm10_splits,
)


mm39 = Genome(
    name="mm39",
    chrom_sizes={
        "chr1": 195154279,
        "chr2": 181755017,
        "chrX": 169476592,
        "chr3": 159745316,
        "chr4": 156860686,
        "chr5": 151758149,
        "chr6": 149588044,
        "chr7": 144995196,
        "chr10": 130530862,
        "chr8": 130127694,
        "chr14": 125139656,
        "chr9": 124359700,
        "chr11": 121973369,
        "chr13": 120883175,
        "chr12": 120092757,
        "chr15": 104073951,
        "chr16": 98008968,
        "chr17": 95294699,
        "chrY": 91455967,
        "chr18": 90720763,
        "chr19": 61420004,
    },
    gff_file="gencode_vM30_GRCm39.gff3.gz",
    fa_file="gencode_vM30_GRCm39.fa.gz",
    bias_file="mm39_bias_v2.h5",
    blacklist_file="mm39.excluderanges.bed",
    bg=(0.29149562991965305, 0.20831919210041502, 0.20833700706759098, 0.29184817091234094),
    splits=mm10_splits,
)
