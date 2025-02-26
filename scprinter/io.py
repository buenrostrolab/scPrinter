from __future__ import annotations

import atexit
import gc
from pathlib import Path
from typing import Literal

import anndata
import gffutils
import h5py
import numpy as np
import pandas as pd
import pyBigWig
from tqdm.auto import tqdm, trange

from .genome import Genome
from .utils import *

anndata_keys = [
    "X",
    "obs",
    "var",
    "obsm",
    "obsp",
    "varm",
    "var",
    "uns",
    "shape",
    "varp",
    "obs_names",
    "var_names",
]


def get_global_disp_models():
    return globals()["dispModels"]


def get_global_bindingscore_model():
    return globals()["bindingScoreModels"]


#
# globals()['dispModels'] = printer.dispersionModel
#             globals()['bindingScoreModels'] = printer.bindingScoreModel
#             globals()['insertion_profile'] = printer.insertion_profile


class scPrinter:
    """
    Core Class of scprinter

    Parameters
    ----------
    adata: anndata.AnnData or snapatac.adataset
        snapatac adata or adataset, which stores insertion_profile
    adata_path: str
        path to adata, could be none
    insertion_profile: dict
        direct assignment of insertion_profile. By default, we used the insertion_profile stored in adata,
        but you can also directly assign the insertion_profile here.
    genome: scp.genome
        a scp.genome object
    attach: bool
        whether to attach the footprintsadata / bindingscoreadata to the printer object. Default is True

    """

    def __init__(
        self,
        adata: snap.AnnData | anndata.AnnData | None = None,
        adata_path: str | Path | None = None,
        insertion_profile: dict | None = None,  # direct assignment of insertion_profile
        genome: Genome | None = None,  # a scp.genome object
        attach: bool = True,
    ) -> None:
        """
        Initialize the scPrinter object
        Parameters
        ----------
        adata: anndata.AnnData or snapatac.adataset
            snapatac adata or adataset, which stores insertion_profile
        adata_path: str
            path to adata, could be none
        insertion_profile: dict
            direct assignment of insertion_profile. By default, we used the insertion_profile stored in adata,
            but you can also directly assign the insertion_profile here.
        genome: scp.genome
            a scp.genome object
        attach: bool
            whether to attach the footprintsadata / bindingscoreadata to the printer object. Default is True
        """

        self.bindingscoreadata = {}
        self.footprintsadata = {}
        self.insertionadata = {}

        self.gene_region_width = (
            1000  # by default when you use a gene name, we will fetch 1000bp around the TSS
        )
        self.genome = genome

        if adata is not None:
            # Lazy loading of insertion_profile, we open the anndata without actually loading the insertions

            # Create some empty slots to store binding and footprints adata
            if "binding score" not in adata.uns:
                adata.uns["binding score"] = {}
            if "footprints" not in adata.uns:
                adata.uns["footprints"] = {}
            if "insertion" not in adata.uns:
                adata.uns["insertion"] = {}
            # Associate some variables, the unique_string object is used to store the global variables and make sure
            # when there are multiple printer open, they won't overlap each other
            self.insertion_file = adata
            self.unique_string = self.insertion_file.uns["unique_string"]

            # Start to read in the binding score data and footprint data
            if attach:
                for result_key, save in zip(
                    ["binding score", "footprints", "insertion"],
                    [self.bindingscoreadata, self.footprintsadata, self.insertionadata],
                ):
                    remove_key = []
                    if result_key not in adata.uns:
                        continue
                    for save_key in adata.uns[result_key]:
                        p = adata.uns[result_key][save_key]
                        success = False
                        if p != "None" and len(p) > 0 and os.path.exists(p):
                            try:
                                # Only allow read when load.
                                print("loading", save_key, p)
                                save[save_key] = snap.read(p, backed="r")
                                success = True
                            except:
                                pass
                        if not success:
                            remove_key.append(save_key)

                    # Update uns
                    a = adata.uns[result_key]
                    for key in remove_key:
                        del a[key]
                    adata.uns[result_key] = a

        # Get genome related info
        if genome is not None:
            bias_path = str(genome.fetch_bias())
            bias_bw_path = str(genome.fetch_bias_bw())
            if adata is not None:
                adata.uns["bias_path"] = bias_path
                adata.uns["bias_bw"] = bias_bw_path

            gff = str(genome.fetch_gff_db())
            if adata is not None:
                adata.uns["gff_db"] = gff
            self.gff_db = gffutils.FeatureDB(gff)

        self.insertion_profile = insertion_profile

        if adata_path is not None:
            adata_name = os.path.basename(adata_path)
            adata_name = ".".join(adata_name.split(".")[:-1])
            self.file_path = os.path.join(os.path.dirname(adata_path), f"{adata_name}_supp")
            if not os.path.exists(self.file_path):
                os.makedirs(self.file_path)
        # load dispersion models
        self.load_disp_model()
        # initialize binding score model empty dict
        self.bindingScoreModel = {}
        return

    def remove_bindingscore(self, key: str):
        """
        Remove a binding score adata

        Parameters
        ----------
        key: str
            key of the binding score adata to be removed

        """
        a = self.insertion_file.uns["binding score"]
        if a[key] != "None":
            os.remove(a[key])
        del a[key]
        self.insertion_file.uns["binding score"] = a
        del self.bindingscoreadata[key]

    def remove_footprints(self, key: str):
        """
        Remove a footprints adata

        Parameters
        ----------
        key: str
            key of the footprints adata to be removed

        """
        a = self.insertion_file.uns["footprints"]
        if a[key] != "None":
            os.remove(a[key])
        del a[key]
        self.insertion_file.uns["footprints"] = a
        del self.bindingscoreadata[key]

    def remove_insertion(self, key: str):
        """
        Remove a insertion adata

        Parameters
        ----------
        key: str
            key of the insertion adata to be removed

        """
        a = self.insertion_file.uns["insertion"]
        if a[key] != "None":
            os.remove(a[key])
        del a[key]
        self.insertion_file.uns["insertion"] = a
        del self.insertionadata[key]

    def fetch_insertion_profile(self, set_global=False):
        """
        Fetch the insertion profile from the adata file
        If the insertion profile is not in csc format, it will be converted to csc format, split by chromosome, and saved
        for future use.

        Parameters
        ----------
        set_global: bool
            whether to set the insertion profile as a global variable (useful in some parallel computation). Default is False

        Returns
        -------
        insertion_profile: dict
            insertion profile in csc format for each chromosome, can be accessed by `insertion_profile[chrom]`

        """

        if self.insertion_profile is None:
            print("Loading insertion profiles")
            self.insertion_profile = {}
            for chrom in self.insertion_file.uns["reference_sequences"]["reference_seq_name"]:
                self.insertion_profile[chrom] = self.insertion_file.obsm["insertion_%s" % chrom]
                gc.collect()
        if set_global:
            unique_string = self.unique_string
            globals()[unique_string + "insertion_profile"] = self.insertion_profile
        return self.insertion_profile

    def close(self):
        self.insertion_file.close()
        for data in self.bindingscoreadata.values():
            try:
                data.close()
            except:
                pass
        for data in self.footprintsadata.values():
            try:
                data.close()
            except:
                pass
        for data in self.insertionadata.values():
            try:
                data.close()
            except:
                pass

    def __getattr__(self, name):
        if name in anndata_keys:
            return getattr(self.insertion_file, name)
        else:
            super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in anndata_keys:
            setattr(self.insertion_file, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        print("head project")
        print(self.insertion_file)
        if len(self.bindingscoreadata) > 0:
            print("detected %d bindingscoreadata" % len(self.bindingscoreadata))
            for key in self.bindingscoreadata:
                print("name", key)
                response = str(self.bindingscoreadata[key])
                response = response.split("\n")
                if len(response) > 1:
                    a = response[-1].strip().split(": ")[1].split(", ")
                    new_final = "    obsm: %d regions results in total: e.g. %s" % (
                        len(a),
                        a[0],
                    )
                    response[-1] = new_final
                    print("\n".join(response))
                else:
                    print(response)

        if len(self.footprintsadata) > 0:
            print("detected %d footprintsadata" % len(self.footprintsadata))
            for key in self.footprintsadata:
                print("name", key)
                response = str(self.footprintsadata[key])
                response = response.split("\n")
                if len(response) > 1:
                    a = response[-1].strip().split(": ")[1].split(", ")
                    new_final = "    obsm: %d regions results in total: e.g. %s" % (
                        len(a),
                        a[0],
                    )
                    response[-1] = new_final
                    print("\n".join(response))
                else:
                    print(response)
        if len(self.insertionadata) > 0:
            print("detected %d insertionadata" % len(self.insertionadata))
            for key in self.insertionadata:
                print("name", key)
                response = str(self.insertionadata[key])
                response = response.split("\n")
                if len(response) > 1:
                    a = response[-1].strip().split(": ")[1].split(", ")
                    new_final = "    obsm: %d regions results in total: e.g. %s" % (
                        len(a),
                        a[0],
                    )
                    response[-1] = new_final
                    print("\n".join(response))
                else:
                    print(response)
        return ""

    def load_disp_model(self, path: str | Path | None = None, set_global=False):
        """
        Load the dispersion model from the path
        When path is None, the default pretrained model will be loaded

        Parameters
        ----------
        path: str | Path | None
            path to the dispersion model

        """
        from .datasets import pretrained_dispersion_model

        if path is None:
            path = pretrained_dispersion_model
        self.dispersionModel = loadDispModel(path)
        if set_global:
            globals()[self.unique_string + "_dispModels"] = self.dispersionModel

    def set_global_bindingscore_model(self, key: str):
        globals()[self.unique_string + "_bindingscoremodel"] = self.bindingScoreModel[key]

    def set_global_var(self, shared=True):
        """
        Set global variables for the dispersionModel, bindingScoreModel, and insertion_profile

        Parameters
        ----------
        shared: bool
            When shared=True, global variables are named as 'dispModels', 'bindingScoreModels', and 'insertion_profile' respectively； When shared=False, global variables are named as f'{unique_string}_dispModels', f'{unique_string}_bindingScoreModels', and f'{unique_string}_insertion_profile'.

        Returns
        -------

        """
        if shared:
            globals()["dispModels"] = self.dispersionModel
            globals()["bindingScoreModels"] = self.bindingScoreModel
            globals()["insertion_profile"] = self.insertion_profile
        else:
            globals()[self.unique_string + "_dispModels"] = self.dispersionModel
            globals()[self.unique_string + "_bindingScoreModels"] = self.bindingScoreModel
            globals()[self.unique_string + "_insertion_profile"] = self.insertion_profile

    def load_bindingscore_model(self, key: str, path: str | Path | None = None, set_global=False):
        """
        Load the binding score model from the path and save it to the adata file with the key

        Parameters
        ----------
        key: str
            key to save the binding score model
        path: str | Path | None
            path to the binding score model

        """
        self.bindingScoreModel[key] = loadBindingScoreModel_pt(path)
        if set_global:
            globals()[self.unique_string + "_bindingscoremodel"] = self.bindingScoreModel[key]

    def __getitem__(self, indices):
        """
        Get the ATAC insertions for a bunch of cells and a given region
        Parameters
        ----------
        indices: We implemented a pretty advanced indexing system for the scPrinter object.
         The indices need to be a tuple of two elements, e.g. [cell_barcodes_part, regions_part],
         where cell_barcodes should be sth like [[barcode1, barcode2], [barcode3, barcode4]]. This would fetch the insertions for
         barcode1-2, sum them across cells, and then fetch the insertions for barcode3-4, sum them across cells, and then stack them together.
         The regions can be a list of region identifiers, e.g. ['chr1:1000-2000'] or a list of gene identifiers, e.g. ['Gene:CTCF'], the transcript start site will be used as the region center.
        Note that, in most numpy array, if you do sth like [[cell_grouping1], [cell_grouping2]], [region1, region2], it would
        fetch the insertions for cell_grouping1, region1, and then cell_grouping2, region2, and stack them together.
        But in scprinter, we decide to return all of them.
        Returns
        -------

        """
        assert (
            isinstance(indices, tuple) and len(indices) == 2
        ), "Only support 2D slicing, e.g. [cell_barcodes, regions]"
        cell_barcodes, regions = indices
        regions = regionparser(regions, self)
        atacs = []
        for region in regions.iterrows():
            atac = get_group_atac(self, cell_barcodes, region)
            atacs.append(atac)
        if len(atacs) == 1:
            atacs = atacs[0]
        else:
            atacs = np.stack(atacs, axis=1)
        return atacs


def load_printer(path: str | Path, genome: Genome, attach: bool = True):
    """
    Load a printer from adata file

    Parameters
    ----------
    path: str | Path
        path to the scprinter main h5ad file
    genome: Genome
        genome object. Must be the same as the one used to process the data
    attach: bool
        whether to attach the footprintsadata / bindingscoreadata to the printer object
    """
    data = snap.read(path)
    if "genome" in data.uns:
        assert data.uns["genome"] == genome.name, "Process data with %s, but now loading with %s" % (
            data.uns["genome"],
            genome.name,
        )
    else:
        print("No genome exists in data.uns. Adding the input genome.")
        data.uns["genome"] = genome.name
    printer = scPrinter(
        adata=data,
        adata_path=path,
        insertion_profile=None,
        genome=genome,
        attach=attach,
    )
    # register close automatically.
    atexit.register(printer.close)
    return printer


def _get_region_atac(printer: scPrinter, cell_barcodes: np.ndarray, region: pd.DataFrame):
    """
    The underlying function to get ATAC insertions for a bunch of cells and a given region

    Parameters
    ----------
    printer: scPrinter object
    cell_barcodes: np.ndarray, the index of cells
    region

    Returns
    -------
    insertion profile
    """
    insertion_profile = printer.fetch_insertion_profile()
    v = region.values.reshape((-1))
    chrom, start_pos, end_pos = v[0], v[1], v[2]
    cell_index = cell_barcodes

    v = insertion_profile[chrom][cell_index, start_pos:end_pos]
    return v


def get_region_atac(
    printer: scPrinter,
    cell_barcodes: list[str] | list[list[str]] | np.ndarray,
    region: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    **kwargs,
):
    """
    Get ATAC insertions for a bunch of cells and a given region (for more complicated situations)

    Parameters
    ----------
    printer: scPrinter
        The printer object you generated by `scprinter.pp.import_fragments` or loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    region: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end], a pyranges object of the regions,
        or a list of region identifiers, e.g. `['chr1:1000-2000']`
        or a list of gene identifiers, e.g. `['Gene:CTCF']`,
        the transcript start site will be used as the region center.
        Note that even when multiple regions are provided, only the first region would be fetched.
    kwargs
        keyword arguments for regionparser

    Returns
    -------
    insertion profile


    """
    region = regionparser(region, printer, **kwargs)
    cell_index = printer.insertion_file.obs_ix(np.array(cell_barcodes))

    return _get_region_atac(printer, cell_index, region)


def _get_group_atac(printer: scPrinter, cell_grouping_idx: np.ndarray, region: pd.DataFrame):
    """
    The underlying function to get ATAC insertions for groups and a given region
    Parameters
    ----------
    printer
    cell_grouping_idx
    region

    Returns
    -------
    insertion profile
    """

    # why slice(None)?
    # Because insertion files are csc_matrix, they are fast when doing col slice
    # But not so when row selection, so we would try to minimize such effect.
    atac_all = _get_region_atac(printer, slice(None), region)
    a = np.array(
        [
            np.asarray(atac_all[barcodes].sum(axis=0)).astype("float32")[0]
            for barcodes in cell_grouping_idx
        ]
    )

    return a


# A wrapper function for more complicated situations
def get_group_atac(printer, cell_grouping, region, **kwargs):
    """
    Get ATAC insertions for a bunch of cell groups and a given region (for more complicated situations)

    Parameters
    ----------
    printer: scPrinter
        The printer object you generated by `scprinter.pp.import_fragments` or loaded by `scprinter.load_printer`
    cell_grouping: list[list[str]] | list[str] | np.ndarray
        The cell grouping you want to visualize, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
    region: str | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want to calculate the binding score.
        You can provide a string of the path to the region bed file,
        a pandas dataframe with the first three columns correspond to [chrom, start, end], a pyranges object of the regions,
        or a list of region identifiers, e.g. `['chr1:1000-2000']`
        or a list of gene identifiers, e.g. `['Gene:CTCF']`,
        the transcript start site will be used as the region center.
        Note that even when multiple regions are provided, only the first region would be fetched.
    kwargs
        keyword arguments for regionparser

    Returns
    -------
    insertion profile


    """

    region = regionparser(region, printer, **kwargs)
    cell_grouping_idx = cell_grouping2cell_grouping_idx(printer, cell_grouping)

    return _get_group_atac(printer, cell_grouping_idx, region)


def _get_group_atac_bw(dict1: dict, cell_grouping: list[str] | np.ndarray, region: pd.DataFrame):
    """
    Get ATAC insertions for groups and a given region through an exported bigwig file from scprinter.pp.sync_visualization
    Parameters
    ----------
    dict1: the dictionary of the bigwig files
    cell_grouping: list[str]
    region

    Returns
    -------

    """
    v = region.values.reshape((-1))
    chrom, start_pos, end_pos = v[0], v[1], v[2]
    a = []
    for group in cell_grouping:
        with pyBigWig.open(dict1[group], "r") as bw:
            a.append(bw.values(chrom, start_pos, end_pos, numpy=True))
    a = np.array(a)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def get_bias_insertions(
    printer: scPrinter,
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str],
    bias_mode: Literal["h5", "bw"] = "h5",
    **kwargs,
):
    """
    Get the bias insertions for given regions

    Parameters
    ----------
    printer: scPrinter
    regions: str | Path | pd.DataFrame | pyranges.PyRanges | list[str]
        The genomic regions you want fetch bias from
    bias_mode:
        The mode to fetch bias, either from h5 file or bigwig file, h5 file works well for large amount of regions and faster
        but it can be memory intensive, bigwig file is slower but more memory efficient
    kwargs

    Returns
    -------

    """
    assert bias_mode in ["h5", "bw"], "bias_mode should be either h5 or bw"
    regions = regionparser(regions, printer, **kwargs)
    chrom_list, start_list, end_list = (
        np.array(regions["Chromosome"]),
        np.array(regions["Start"]).astype("int"),
        np.array(regions["End"]).astype("int"),
    )
    uniq_chrom = np.unique(chrom_list)
    width = end_list[0] - start_list[0]
    if bias_mode == "h5":
        with h5py.File(printer.genome.fetch_bias(), "r") as dct:
            precomputed_bias = {chrom: np.array(dct[chrom]) for chrom in dct.keys()}
        final_result = np.zeros((len(regions), width))
        for chrom in uniq_chrom:
            bias = precomputed_bias[chrom]
            mask = chrom_list == chrom
            bias_region_chrom = strided_axis0(bias, width)[start_list[mask]]
            final_result[mask] = bias_region_chrom
    else:
        bias_bw = printer.genome.fetch_bias_bw()
        bias_bw = pyBigWig.open(bias_bw, "r")
        final_result = np.array(
            [
                bias_bw.values(chrom, start, end, numpy=True)
                for chrom, start, end in zip(chrom_list, start_list, end_list)
            ]
        )
        final_result = np.nan_to_num(final_result, nan=0.0, posinf=0.0, neginf=0.0)
    return final_result
