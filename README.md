<h1 align="center">
  <br>
  <a href="https://ruochiz.com/scprinter_doc/"><img src="img/logo.png" alt="scPrinter" width="200"></a>
  <br>
  scPrinter
  <br>
</h1>

scPrinter is a computational framework for the multi-scale footprinting analysis of single-cell ATAC-seq data.
scPrinter is designed to identify and visualize the regulatory elements that drive cell-type-specific gene expression programs through footprinting.
scPrinter uses a deep learning model to predict the activity of transcription factors from single-cell ATAC-seq data.
scPrinter also provides a suite of visualization tools to explore the calculated multi-scale footprints.

## Getting Started

### Installation

Typical installation time: within 30mins

Please install pytorch (with gpu-compatibility if you will be using GPU) before installing scPrinter. You can install pytorch by following the instructions on the official website: https://pytorch.org/get-started/locally/.
A quick installation of scPrinter can be done by running the following command, and most of the dependencies would be resolved automatically.

```angular2html
# Clone this repository
$ git clone https://github.com/buenrostrolab/scPrinter

# Go into the repository
$ cd scPrinter

# pip install
pip install ./
```

A step by step guide that I use to setup a new environment is as follows:

```angular2html

mamba create -n scprinter -c rapidsai -c pytorch -c nvidia  -c conda-forge  -c bioconda pytorch torchvision torchaudio pytorch-cuda=12.1 jupyterlab ipywidgets ipykernel joblib  'llvm-openmp<16' htop nvtop wandb rapids=24.06 python=3.11.* 'cuda-version>=12.0,<=12.2' cupy rmm 'anndata>0.8.0' python-kaleido multiprocess natsort numpy pandas plotly polars pooch 'igraph>=0.10.4' pynndescent pyarrow pyfaidx rustworkx scipy scikit-learn tqdm typing_extensions umap-learn texttable setuptools-rust pkg-config openblas  pyranges biopython dna_features_viewer 'scanpy>=1.9' pyBigWig gffutils rust cmake  beartype jupyterlab ipywidgets jupyterlab_widgets ipykernel macs2 logomaker shap transformers gcc weasyprint libxml2 libxslt hdf5plugin leidenalg imagemagick

mamba activate scprinter
```

test the following in python:

```angular2html
import torch
torch.cuda.is_available()
import cupy
a = cupy.zeros((1000, 1000)) # check gpu usage
```

pip install the following because the conda version results in version incompatibility.

```angular2html
pip install bioframe pysam pybedtools
mamba install -c bioconda MACS3 cykhash hmmlearn python-igraph pandas polars pyfaidx
pip install MOODS-python snapatac2 ema_pytorch modisco-lite
git clone https://github.com/austintwang/finemo_gpu.git
cd finemo_gpu
pip install ./

cd ../
git clone https://github.com/jmschrei/tfmodisco-lite.git
cd tfmodisco-lite
pip install ./

```

Finally scprinter.

```angular2html
# Clone this repository
$ git clone https://github.com/buenrostrolab/scPrinter

# Go into the repository
$ cd scPrinter

# pip install
pip install ./
```

For personal environment, you don't need to do this, for shared environment make everyone use the same cache dir

```
mamba env config vars set SCPRINTER_DATA="/n/holylfs06/LABS/buenrostro_lab/Lab/.cache/scprinter/"
```

## Usage and Tutorial

Quick-start tutorials for seq2PRINT TF binding prediction can be found at [here](https://ruochiz.com/scprinter_doc/tutorials/PBMC_scATAC_tutorial.html) (for single cell ATAC) and [here](https://ruochiz.com/scprinter_doc/tutorials/PBMC_bulkATAC_tutorial.html) (for bulk ATAC). 

If you do not need the seq2PRINT-inferred TF binding, but only wish to calculate the raw multiscale footprints (e.g., for tracking nucleosome changes) across conditions and genomic loci, please use [this tutorial](https://github.com/buenrostrolab/PRINT/blob/main/analyses/degron/CTCF_degron_tutorial.ipynb) as a reference. 

For more detailed information please see our documentation and tutorials at https://ruochiz.com/scprinter_doc/. API reference can be found at https://ruochiz.com/scprinter_doc/reference/index.html.

## Citation

If you use scPrinter in your research, please cite the following paper:

```
@article{hu2025multiscale,
title={Multiscale footprints reveal the organization of cis-regulatory elements},
author={Hu, Yan and Horlbeck, Max A and Zhang, Ruochi and Ma, Sai and Shrestha, Rojesh and Kartha, Vinay K and Duarte, Fabiana M and Hock, Conrad and Savage, Rachel E and Labade, Ajay and others},
journal={Nature},
pages={1--8},
year={2025},
publisher={Nature Publishing Group UK London}
```

## License

MIT
