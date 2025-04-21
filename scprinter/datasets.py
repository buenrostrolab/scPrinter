import os
import stat

import pandas as pd
import pooch
from pooch import Decompress, Untar, Unzip

_datasets = None


def giverightstothegroup(fname, action, pooch):
    """
    Processes the downloaded file and returns a new file name.

    The function **must** take as arguments (in order):

    fname : str
        The full path of the file in the local data storage
    action : str
        Either: "download" (file doesn't exist and will be downloaded),
        "update" (file is outdated and will be downloaded), or "fetch"
        (file exists and is updated so no download is necessary).
    pooch : pooch.Pooch
        The instance of the Pooch class that is calling this function.

    The return value can be anything but is usually a full path to a file
    (or list of files). This is what will be returned by Pooch.fetch and
    pooch.retrieve in place of the original file path.
    """
    if action == "download":
        os.chmod(fname, stat.S_IRWXG | stat.S_IRWXU)
    if "tar" in fname:
        return Untar()(fname, action, pooch)
    elif "gz" in fname:
        return Decompress(method="gzip")(fname, action, pooch)
    elif "zip" in fname:
        return Unzip()(fname, action, pooch)
    return fname


def datasets():
    global _datasets
    if _datasets is None:
        # dir1 = os.path.dirname(pooch.__file__)
        # dir1 = "/".join(dir1.split("/")[:-1])
        # dir = os.path.join(dir1, 'scprinter_cache')
        _datasets = pooch.create(
            path=pooch.os_cache("scprinter"),
            base_url="",
            env="SCPRINTER_DATA",
            registry={
                # scp files
                "dispersion_model_py.h5": "md5:cbd6cefed73f36aaf121aa73f2d2b658",
                "nucleosome_model_py.pt": "md5:fc58e8698b1f869b67b2c1b7b4398b3b",
                "TFBS_0_conv_v2.pt": "md5:2284373c8c24ac94ff874755a9e18108",
                "TFBS_1_conv_v2.pt": "md5:c7ab928fb61641a5b6b61087db9b055f",
                "TFBS_model_py.pt": "md5:5bd79a9c4f3374241a6f4341eb39fe2c",
                "TFBS_model_model1_py.pt": "md5:7893684aa234df3b58995b212d9a8363",
                "Tn5_NN_model_py_v2.pt": "md5:5ea6c26fcd54ad2aff3d9705ed54dd3c",
                "dispersion_model_py_v2.h5": "md5:341a011610c7db9a1418917e359d3169",
                # motif database
                "JASPAR2022_core_nonredundant.jaspar": "md5:af268b3e9589f52440007b43cba358f8",
                "CisBP_Human.jaspar": "md5:23b85a4cd8299416dd5d85516c0cdcbf",
                "CisBPJASPA.jaspar": "md5:7f965084f748d9e91f950a7981ffd7d5",
                "CisBP_Mouse_FigR": "md5:f00120636a859a3de49aad1b5e6a8c1c",
                "CisBP_Human_FigR": "md5:333c0c141cc08f5e0bdf23f9eb335db7",
                "CisBP_Mouse_FigR_Bagging": "md5:e9ff3beff96d066239c0200f5f8940d7",
                "CisBP_Human_FigR_Bagging": "md5:3c075c36a05e4dc2e9666d4ad9cb3c85",
                "CisBP_Mouse_FigR_meme": "md5:45bdeee2222db8d36182b44acb63dc80",
                "CisBP_Human_FigR_meme": "md5:97e96fb17fc015e1d15c76609167606f",
                # TSS files
                "TSSRanges_hg19_FigR": "md5:856efa6efbeea92333887e846ce84711",
                "TSSRanges_hg38_FigR": "md5:1544c730295e883da1576633beb9b87a",
                "TSSRanges_mm10_FigR": "md5:a8af354b72ef45c5f2d0fb48124ead1b",
                # bias file
                # "hg38Tn5Bias.h5": "md5:5ff8b43c50eb23639e3d93b5b1e8a50a",
                "ce11Tn5Bias.tar.gz": "md5:10d8d17f94f695c06c0f66968f67b55b",
                "danRer11Tn5Bias.tar.gz": "md5:8d4fe94ccbde141f6edefc1f0ce36c10",
                "dm6Tn5Bias.tar.gz": "md5:7f256a41b7232bd5c3389b0e190d9788",
                "hg38Tn5Bias.tar.gz": "md5:89f205e6be682b15f87a2c2cc00e8cbd",
                "mm10Tn5Bias.tar.gz": "md5:901b928946b65e7bfba3a93e085f19f0",
                "panTro6Tn5Bias.tar.gz": "md5:ba208a4cdc2e1fc09d66cac44e85e001",
                "sacCer3Tn5Bias.tar.gz": "md5:ed811aabe1ffa4bdb1520d4b25ee9289",
                "mm39Tn5Bias.h5": "md5:1782f01f170982ea595228f7820f1d17",
                "ce11_bias_v2.h5": "md5:95a545d9acfc6121bf40f6330e74fd03",
                "danRer11_bias_v2.h5": "md5:4da32a3f73dbca46bf4528138e3b5ad0",
                "dm6_bias_v2.h5": "md5:c69a893ad99157223e880714af46c6fb",
                "hg19_bias_v2.h5": "md5:013ca480e6a7d3f541ef2404ee56a3c7",
                "hg38_bias_v2.h5": "md5:c56f03c32bc793bf2a5cf23d4f66f20e",
                "mm10_bias_v2.h5": "md5:29473ec7d3108683d90b92a783704cf0",
                "mm39_bias_v2.h5": "md5:36520483892ea158c0c4a1d672a4827c",
                "panTro6_bias_v2.h5": "md5:146d1d8213cb3b42159e6bdad2959945",
                "sacCer3_bias_v2.h5": "md5:a5e8c0162be0152c75e0af20125b9e42",
                # Genome files
                "gencode_v41_GRCh37.gff3.gz": "sha256:df96d3f0845127127cc87c729747ae39bc1f4c98de6180b112e71dda13592673",
                "gencode_v41_GRCh37.fa.gz": "sha256:94330d402e53cf39a1fef6c132e2500121909c2dfdce95cc31d541404c0ed39e",
                "gencode_v41_GRCh38.gff3.gz": "sha256:b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
                "gencode_v41_GRCh38.fa.gz": "sha256:4fac949d7021cbe11117ddab8ec1960004df423d672446cadfbc8cca8007e228",
                "gencode_vM25_GRCm38.gff3.gz": "sha256:e8ed48bef6a44fdf0db7c10a551d4398aa341318d00fbd9efd69530593106846",
                "gencode_vM25_GRCm38.fa.gz": "sha256:617b10dc7ef90354c3b6af986e45d6d9621242b64ed3a94c9abeac3e45f18c17",
                "gencode_vM30_GRCm39.gff3.gz": "sha256:6f433e2676e26569a678ce78b37e94a64ddd50a09479e433ad6f75e37dc82e48",
                "gencode_vM30_GRCm39.fa.gz": "sha256:3b923c06a0d291fe646af6bf7beaed7492bf0f6dd5309d4f5904623cab41b0aa",
                # Tutorial files
                "scprinter_BMMCTutorial.zip": "md5:d9027cf73b558d03276483384ddad88c",
                # Blacklist file
                "hg38-blacklist.v2.bed.gz": "md5:83fe6bf8187a64dee8079b80f75ba289",
                "mm10-blacklist.v2.bed.gz": "md5:4ae47e40309533c2a71de55494cda9bc",
                "mm39.excluderanges.bed": "md5:9445a55bcebb3940ad98178370980318",
            },
            urls={
                "dispersion_model_py.h5": "https://zenodo.org/records/14194242/files/dispersion_model_py.h5",
                "dispersion_model_py_v2.h5": "https://zenodo.org/records/15170402/files/dispersion_model_py_v2.h5",
                "nucleosome_model_py.pt": "https://zenodo.org/records/14194242/files/nucleosome_model_py.pt",
                "TFBS_model_py.pt": "https://zenodo.org/records/14194242/files/TFBS_model_py.pt",
                "TFBS_model_model1_py.pt": "https://zenodo.org/records/14194242/files/TFBS_model_cluster_I_py.pt",
                # Sequence TFBS models:
                "TFBS_0_conv_v2.pt": "https://zenodo.org/records/15085406/files/TFBS_0_conv_v2.pt",
                "TFBS_1_conv_v2.pt": "https://zenodo.org/records/15085406/files/TFBS_1_conv_v2.pt",
                "Tn5_NN_model_py_v2.pt": "https://zenodo.org/records/15103252/files/Tn5_NN_model_py_v2.pt",
                # motif database
                "JASPAR2022_core_nonredundant.jaspar": "https://drive.google.com/uc?export=download&id=1YmRZ3sABLJvv9uj40BY97Rdqyodd852P",
                "CisBP_Human.jaspar": "https://drive.google.com/uc?export=download&id=1IVcg27kxzG5TtnjqFrheGxXa-0kfAOW7",
                "CisBPJASPA.jaspar": "https://drive.google.com/uc?export=download&id=1I62z-JZaQOnue7iimU0Q8Uf7ZjpEHLGn",
                "CisBP_Mouse_FigR": "https://github.com/ruochiz/FigRmotifs/raw/main/mouse_pfms_v4.txt",
                "CisBP_Human_FigR": "https://github.com/ruochiz/FigRmotifs/raw/main/human_pfms_v4.txt",
                "CisBP_Mouse_FigR_Bagging": "https://github.com/ruochiz/FigRmotifs/raw/main/FigR_motifs_bagging_mouse.txt",
                "CisBP_Human_FigR_Bagging": "https://github.com/ruochiz/FigRmotifs/raw/main/FigR_motifs_bagging_human.txt",
                "CisBP_Mouse_FigR_meme": "https://github.com/ruochiz/FigRmotifs/raw/main/mouse_pfms_v4_meme.txt",
                "CisBP_Human_FigR_meme": "https://github.com/ruochiz/FigRmotifs/raw/main/human_pfms_v4_meme.txt",
                "TSSRanges_hg19_FigR": "https://github.com/ruochiz/FigRmotifs/raw/main/hg19TSSRanges.txt",
                "TSSRanges_hg38_FigR": "https://github.com/ruochiz/FigRmotifs/raw/main/hg38TSSRanges.txt",
                "TSSRanges_mm10_FigR": "https://github.com/ruochiz/FigRmotifs/raw/main/mm10TSSRanges.txt",
                # bias file
                "ce11Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/ce11Tn5Bias.tar.gz",
                "danRer11Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/danRer11Tn5Bias.tar.gz",
                "dm6Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/dm6Tn5Bias.tar.gz",
                "hg38Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/hg38Tn5Bias.tar.gz",
                "mm10Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/mm10Tn5Bias.tar.gz",
                "mm39Tn5Bias.h5": "https://zenodo.org/records/14164466/files/mm39Tn5Bias.h5",
                "panTro6Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/panTro6Tn5Bias.tar.gz",
                "sacCer3Tn5Bias.tar.gz": "https://zenodo.org/record/7121027/files/sacCer3Tn5Bias.tar.gz",
                "ce11_bias_v2.h5": "https://zenodo.org/record/15224770/files/ce11_bias_v2.h5",
                "danRer11_bias_v2.h5": "https://zenodo.org/record/15224770/files/danRer11_bias_v2.h5",
                "dm6_bias_v2.h5": "https://zenodo.org/record/15224770/files/dm6_bias_v2.h5",
                "hg19_bias_v2.h5": "https://zenodo.org/record/15224770/files/hg19_bias_v2.h5",
                "hg38_bias_v2.h5": "https://zenodo.org/record/15224770/files/hg38_bias_v2.h5",
                "mm10_bias_v2.h5": "https://zenodo.org/record/15224770/files/mm10_bias_v2.h5",
                "mm39_bias_v2.h5": "https://zenodo.org/record/15224770/files/mm39_bias_v2.h5",
                "panTro6_bias_v2.h5": "https://zenodo.org/record/15224770/files/panTro6_bias_v2.h5",
                "sacCer3_bias_v2.h5": "https://zenodo.org/record/15224770/files/sacCer3_bias_v2.h5",
                "gencode_v41_GRCh37.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/gencode.v41lift37.basic.annotation.gff3.gz",
                "gencode_v41_GRCh37.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/GRCh37.primary_assembly.genome.fa.gz",
                "gencode_v41_GRCh38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz",
                "gencode_v41_GRCh38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh38.primary_assembly.genome.fa.gz",
                "gencode_vM25_GRCm38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gff3.gz",
                "gencode_vM25_GRCm38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz",
                "gencode_vM30_GRCm39.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.basic.annotation.gff3.gz",
                "gencode_vM30_GRCm39.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/GRCm39.primary_assembly.genome.fa.gz",
                # Blacklist file
                "hg38-blacklist.v2.bed.gz": "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz",
                "mm10-blacklist.v2.bed.gz": "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz",
                "mm39.excluderanges.bed": "https://zenodo.org/records/14164466/files/mm39.excluderanges.bed",
                "scprinter_BMMCTutorial.zip": "https://zenodo.org/records/14164466/files/scprinter_BMMCTutorial.zip",
            },
        )
    return _datasets


def JASPAR2022_core():
    return str(
        datasets().fetch("JASPAR2022_core_nonredundant.jaspar", processor=giverightstothegroup)
    )


def CisBP_Human():
    return str(datasets().fetch("CisBP_Human.jaspar", processor=giverightstothegroup))


def CisBPJASPA():
    return str(datasets().fetch("CisBPJASPA.jaspar", processor=giverightstothegroup))


def TFBS_model():
    """
    A wrapper function to get Pretrained TFBS model
    You can also get it by `scprinter.datasets.pretrained_TFBS_model`

    Returns
    -------
    str: path to the TFBS model
    """
    return str(datasets().fetch("TFBS_model_py.pt", processor=giverightstothegroup))


def TFBS_model_classI():
    """
    A wrapper function to get Pretrained TFBS model (class I, meaning only TFs that left a strong footprints)
    You can also get it by `scprinter.datasets.pretrained_TFBS_model_classI`

    Returns
    -------
    str: path to the TFBS model (class I)
    """
    return str(datasets().fetch("TFBS_model_model1_py.pt", processor=giverightstothegroup))


def NucBS_model():
    """
    A wrapper function to get Pretrained NucBS model
    You can also get it by `scprinter.datasets.pretrained_NucBS_model`

    Returns
    -------
    str: path to the NucBS model
    """
    return str(datasets().fetch("nucleosome_model_py.pt", processor=giverightstothegroup))


def dispersion_model_v2():
    """
    A wrapper function to get Pretrained dispersion model
    You can also get it by `scprinter.datasets.pretrained_dispersion_model`

    Returns
    -------
    str: path to the dispersion model
    """
    return str(datasets().fetch("dispersion_model_py_v2.h5", processor=giverightstothegroup))


def dispersion_model_v1():
    """
    A wrapper function to get Pretrained dispersion model
    You can also get it by `scprinter.datasets.pretrained_dispersion_model`

    Returns
    -------
    str: path to the dispersion model
    """
    return str(datasets().fetch("dispersion_model_py.h5", processor=giverightstothegroup))


dispersion_model = dispersion_model_v2


def FigR_motifs(species="mouse"):
    """
    A wrapper function to get FigR motifs for mouse or human
    You can also get it by `scprinter.datasets.FigR_motifs_mouse` or `scprinter.datasets.FigR_motifs_human

    Returns
    -------
    str: path to the FigR motifs
    """
    if species == "mouse":
        return str(datasets().fetch("CisBP_Mouse_FigR", processor=giverightstothegroup))
    elif species == "human":
        return str(datasets().fetch("CisBP_Human_FigR", processor=giverightstothegroup))
    else:
        raise ValueError("species should be either 'mouse' or 'human'")


def FigR_motifs_bagging(species="mouse"):
    """
    A wrapper function to get FigR motifs bagging for mouse or human
    You can also get it by `scprinter.datasets.FigR_motifs_bagging_mouse` or `scprinter.datasets.FigR_motifs_bagging_human

    Returns
    -------
    str: path to the FigR motifs bagging
    """
    if species == "mouse":
        return str(datasets().fetch("CisBP_Mouse_FigR_Bagging", processor=giverightstothegroup))
    elif species == "human":
        return str(datasets().fetch("CisBP_Human_FigR_Bagging", processor=giverightstothegroup))
    else:
        raise ValueError("species should be either 'mouse' or 'human'")


def BMMCTutorial():
    """
    A wrapper function to get BMMC Tutorial data.

    Returns
    -------
    str: path to the BMMC Tutorial data
    """
    files = datasets().fetch("scprinter_BMMCTutorial.zip", processor=giverightstothegroup)
    dict1 = {}
    for f in files:
        if "bed" in f:
            dict1["region"] = f
        elif "Fragments" in f:
            dict1["fragments"] = f
        elif "groupInfo" in f:
            dict1["groupInfo"] = f
        elif "barcodeGrouping" in f:
            dict1["barcodeGrouping"] = f
    return dict1


pretrained_TFBS_model = datasets().fetch("TFBS_model_py.pt", processor=giverightstothegroup)
pretrained_TFBS_model_classI = datasets().fetch(
    "TFBS_model_model1_py.pt", processor=giverightstothegroup
)
pretrained_NucBS_model = datasets().fetch("nucleosome_model_py.pt", processor=giverightstothegroup)
pretrained_dispersion_model_v1 = datasets().fetch(
    "dispersion_model_py.h5", processor=giverightstothegroup
)
pretrained_dispersion_model_v2 = datasets().fetch(
    "dispersion_model_py_v2.h5", processor=giverightstothegroup
)
pretrained_dispersion_model = pretrained_dispersion_model_v2


pretrained_seq_TFBS_model0 = datasets().fetch("TFBS_0_conv_v2.pt", processor=giverightstothegroup)
pretrained_seq_TFBS_model1 = datasets().fetch("TFBS_1_conv_v2.pt", processor=giverightstothegroup)

pretrained_Tn5_bias_model = datasets().fetch(
    "Tn5_NN_model_py_v2.pt", processor=giverightstothegroup
)

FigR_motifs_mouse = datasets().fetch("CisBP_Mouse_FigR", processor=giverightstothegroup)
FigR_motifs_human = datasets().fetch("CisBP_Human_FigR", processor=giverightstothegroup)
FigR_motifs_bagging_mouse = datasets().fetch(
    "CisBP_Mouse_FigR_Bagging", processor=giverightstothegroup
)
FigR_motifs_bagging_human = datasets().fetch(
    "CisBP_Human_FigR_Bagging", processor=giverightstothegroup
)
FigR_motifs_mouse_meme = datasets().fetch("CisBP_Mouse_FigR_meme", processor=giverightstothegroup)
FigR_motifs_human_meme = datasets().fetch("CisBP_Human_FigR_meme", processor=giverightstothegroup)

FigR_hg19TSSRanges = pd.read_csv(
    datasets().fetch("TSSRanges_hg19_FigR", processor=giverightstothegroup), sep="\t"
)
FigR_hg19TSSRanges.index = FigR_hg19TSSRanges["gene_name"]
FigR_hg38TSSRanges = pd.read_csv(
    datasets().fetch("TSSRanges_hg38_FigR", processor=giverightstothegroup), sep="\t"
)
FigR_hg38TSSRanges.index = FigR_hg38TSSRanges["gene_name"]
FigR_mm10TSSRanges = pd.read_csv(
    datasets().fetch("TSSRanges_mm10_FigR", processor=giverightstothegroup), sep="\t"
)
FigR_mm10TSSRanges.index = FigR_mm10TSSRanges["gene_name"]
