import anndata
import bioframe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator
from pynndescent import NNDescent
from scipy.sparse import csr_matrix, hstack, issparse, vstack
from scipy.stats import norm, pearsonr, spearmanr
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

from .preprocessing import mean_norm_counts, mean_norm_counts_array
from .utils import get_peak_bias, regionparser


def ss(pair):
    x, y = pair
    return spearmanr(x.toarray().flatten(), y.toarray().flatten())[0]


def spearman_block(A, B, id1, id2, n_jobs):
    id1 = np.array(id1).astype("int")
    id2 = np.array(id2).astype("int")

    def generate_pairs():
        for i, j in zip(id1, id2):
            yield A[i], B[j]

    # Use process_map with a generator
    res = process_map(
        ss,
        generate_pairs(),
        max_workers=n_jobs,
        total=len(id1),
        desc="Calculating Spearman correlation",
    )
    return np.array(res)


def fast_gene_peak_corr(
    atac_adata,
    rna_adata,
    genome,
    tss_df,
    normalize_atac=True,
    gene_list=None,
    window_pad_size=50000,
    n_jobs=6,
    n_bg=100,
    pval_cut=None,
    pos_only=True,
    multimapping=False,
    recal_coverage=True,
):
    """
    Calculate the spearman correlation between ATAC-seq peaks and gene expression with chromvar-style significance testing

    Parameters
    ----------
    atac_adata: anndata.AnnData
        AnnData object containing ATAC-seq data (this function would automatically subset only the shared cells)
    rna_adata: anndata.AnnData
        AnnData object containing RNA-seq data
    genome: scp.genome.Genome
        Genome object containing genome information
    tss_df: pd.DataFrame
        DataFrame containing TSS regions (first 3 columns as chr,start,end, and index as gene name). You can get them from scp.datasets.FigR_hg19TSSRanges; scp.datasets.FigR_hg38TSSRanges; scp.datasets.FigR_mm10TSSRanges.
    normalize_atac: bool
        Whether to normalize ATAC-seq data before calculating correlation
    gene_list: list
        List of genes to calculate correlation. If None, all genes in rna_adata will be used
    window_pad_size: int
        The window size (on both sizes) around TSS regions to consider for correlation calculation
    n_jobs: int
        Number of jobs to use for parallel processing
    n_bg: int
        Number of background peaks to use for significance testing
    pval_cut: float
        P-value cutoff for significance testing
    pos_only: bool
        Whether to only consider positive correlation
    multimapping: bool
        Whether to keep all peak-gene pair or keep only the peak-gene pair with the highest correlation
    recal_coverage: bool
        Whether to re-calculate coverage for peaks and genes based on the provided anndata objects. If False, it would assume there are "coverage" cols in atac_adata.var and rna_adata.var and would use that to match background peaks and genes
    Returns
    -------
    pd.DataFrame
        DataFrame containing gene-peak correlation results

    """
    # make sure atac and rna anndata are in the same order
    shared_barcode = np.intersect1d(atac_adata.obs_names, rna_adata.obs_names)
    atac_adata = atac_adata[shared_barcode]
    rna_adata = rna_adata[shared_barcode]
    # valid_peaks = np.array((atac_adata.X != 0).sum(axis=0) != 0).reshape((-1))
    valid_peaks = np.unique(atac_adata.X.nonzero()[1])
    print("Found", len(valid_peaks), "Valid Peaks")
    atac_adata = atac_adata[:, valid_peaks].copy()
    get_peak_bias(atac_adata, genome)
    if normalize_atac:
        atac_mat = mean_norm_counts_array(atac_adata.X)
    else:
        atac_mat = atac_adata.X

    valid_gene = np.unique(rna_adata.X.nonzero()[1])
    print("Found", len(valid_gene), "Valid Genes")
    rna_adata = rna_adata[:, valid_gene].copy()
    if not gene_list:
        gene_list = rna_adata.var_names
    gene_list = list(set(gene_list) & set(tss_df.index) & set(rna_adata.var_names))
    rna_adata = rna_adata[:, gene_list].copy()
    atac_ranges = regionparser(list(atac_adata.var_names))
    atac_ranges.columns = ["chrom", "start", "end"] + list(atac_ranges.columns[3:])
    atac_ranges_center = (atac_ranges["start"] + atac_ranges["end"]) // 2
    atac_ranges["start"] = atac_ranges_center
    atac_ranges["end"] = atac_ranges_center + 1
    atac_ranges["atac_peak_name"] = atac_adata.var_names
    tss_df["gene_name"] = tss_df.index

    gene_name2id = {gene: i for i, gene in enumerate(gene_list)}
    tss_df = tss_df.loc[gene_list].reset_index(drop=True)
    atac_ranges["atac_peak_id"] = np.arange(atac_ranges.shape[0]).astype("int")
    tss_df.iloc[:, 1] = tss_df.iloc[:, 1] - window_pad_size
    tss_df.iloc[:, 2] = tss_df.iloc[:, 2] + window_pad_size
    tss_df.columns = ["chrom", "start", "end"] + list(tss_df.columns[3:])
    tss_df["gene_id"] = [gene_name2id[gene] for gene in tss_df["gene_name"]]

    # Find overlaps between peaks and TSS regions
    overlaps = bioframe.overlap(tss_df, atac_ranges)
    overlaps = overlaps[
        (~overlaps["gene_name"].isna()) & (~overlaps["atac_peak_name_"].isna())
    ].copy()

    print("Found", overlaps.shape[0], "Gene-Peak pairs")
    print(len(overlaps["gene_name"].unique()), "Unique Genes")
    print(len(overlaps["atac_peak_name_"].unique()), "Unique Peaks")

    # Initialize background peak selection
    num_bg_pairs = 100000
    bg_genes_id = np.random.choice(len(gene_list), num_bg_pairs, replace=True)
    bg_peaks_id = np.random.choice(len(atac_adata.var_names), num_bg_pairs, replace=True)
    if recal_coverage:
        atac_adata.var["coverage"] = np.array(atac_mat.mean(axis=0)).reshape((-1))
        rna_adata.var["coverage"] = np.array(rna_adata.X.mean(axis=0)).reshape((-1))

    # Background feature calculation
    bg_features = np.stack(
        [
            np.array(atac_adata.var.iloc[bg_peaks_id]["gc_content"]),
            np.array(atac_adata.var.iloc[bg_peaks_id]["coverage"]),
            np.array(rna_adata.var.iloc[bg_genes_id]["coverage"]),
        ],
        axis=1,
    )

    # Observed feature calculation

    ob_genes_id = overlaps["gene_id"]
    ob_peaks_id = overlaps["atac_peak_id_"]

    ob_features = np.stack(
        [
            np.array(atac_adata.var.iloc[ob_peaks_id]["gc_content"]),
            np.array(atac_adata.var.iloc[ob_peaks_id]["coverage"]),
            np.array(rna_adata.var.iloc[ob_genes_id]["coverage"]),
        ],
        axis=1,
    )

    # Rescale features for balanced weighting
    all_features = np.concatenate([bg_features, ob_features], axis=0)
    mean_, std_ = all_features.mean(axis=0), all_features.std(axis=0)
    bg_features = (bg_features - mean_) / std_
    ob_features = (ob_features - mean_) / std_

    # Find nearest neighbors in background
    index = NNDescent(bg_features, n_neighbors=n_bg, n_jobs=n_jobs)
    bg_indices, _ = index.query(ob_features, k=n_bg)

    correlations_obs = spearman_block(
        rna_adata.X.T.tocsr(),
        atac_mat.T.tocsr(),
        np.array(ob_genes_id),
        np.array(ob_peaks_id),
        n_jobs=n_jobs,
    )
    correlations_bg = spearman_block(
        rna_adata.X.T.tocsr(),
        atac_mat.T.tocsr(),
        np.array(bg_genes_id),
        np.array(bg_peaks_id),
        n_jobs=n_jobs,
    )

    chunk_size = 1000
    pvals = []
    for i in trange(0, len(ob_genes_id), chunk_size, desc="Calculating p-values"):
        obs_corr = correlations_obs[i : i + chunk_size]
        bg_corrs = correlations_bg[bg_indices[i : i + chunk_size]]
        mean_bg = np.mean(bg_corrs, axis=1)
        std_bg = np.std(bg_corrs, axis=1)
        zscore = (obs_corr - mean_bg) / std_bg
        pval = 1 - norm.cdf(zscore, 0, 1)
        pvals.append(pval)
    pvals = np.concatenate(pvals)

    # Prepare results
    ob_genes = overlaps["gene_name"]
    ob_peaks = overlaps["atac_peak_name_"]

    results = pd.DataFrame(
        {"Gene": ob_genes, "PeakRanges": ob_peaks, "rObs": correlations_obs, "pvalZ": pvals}
    )
    # for each gene - peak combination, keep with the highest correlation (due to multiple transcripts)
    results = results.loc[results.groupby(["Gene", "PeakRanges"])["rObs"].idxmax()]
    if not multimapping:
        results = results.loc[results.groupby("PeakRanges")["rObs"].idxmax()]
    if pval_cut:
        results = results[results["pvalZ"] <= pval_cut]
    if pos_only:
        results = results[results["rObs"] > 0]
    return results


def dorc_j_plot(dorc, cutoff=7, label_top=25, return_gene_list=False):
    """
    A function to make the J-plot for DORCs

    Parameters
    ----------
    dorc: pd.DataFrame
        DataFrame containing DORC results
    cutoff: int
        Cutoff for the number of correlated peaks
    label_top: int
        Number of top genes to label
    return_gene_list: bool
        Whether to return the list of DORC genes


    Returns
    -------
    List
        List of DORC genes if return_gene_list is

    """
    dorc_tab = dorc
    # Ensure required columns are present
    assert {"PeakRanges", "Gene", "pvalZ"}.issubset(
        dorc_tab.columns
    ), "dorcTab must have columns: 'PeakRanges', 'Gene', 'pvalZ'"

    # Count the number of significant peak associations for each gene
    num_dorcs = (
        dorc_tab.groupby("Gene")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    num_dorcs["Index"] = num_dorcs.index + 1  # Add order index
    num_dorcs["isDORC"] = num_dorcs["n"].apply(lambda x: "Yes" if x >= cutoff else "No")
    num_dorcs["Label"] = num_dorcs.apply(
        lambda row: row["Gene"] if row["isDORC"] == "Yes" and row["Index"] <= label_top else "",
        axis=1,
    )

    dorc_genes = num_dorcs.loc[num_dorcs["isDORC"] == "Yes", "Gene"].tolist()

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Index", y="n", data=num_dorcs, color="gray")
    sns.scatterplot(
        x="Index",
        y="n",
        hue="isDORC",
        data=num_dorcs,
        palette={"No": "gray", "Yes": "firebrick"},
        s=10,
    )

    # Add cutoff lines
    plt.axhline(y=cutoff, color="black", linestyle="dotted")
    if len(dorc_genes) > 0:
        plt.axvline(
            x=max(num_dorcs.loc[num_dorcs["isDORC"] == "Yes", "Index"]),
            color="black",
            linestyle="dotted",
        )

    # Add labels
    texts = []
    for _, row in num_dorcs.iterrows():
        if row["Label"]:
            texts.append(plt.text(row["Index"], row["n"], row["Label"], fontstyle="italic"))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        force_points=1.2,  # Increase repelling force on points
        force_text=1.5,  # Increase repelling force on text
        expand_points=(2, 2),  # Expand spacing around points
        expand_text=(2, 2),  # Expand spacing around text
    )

    # Customization
    plt.gca().invert_xaxis()  # Flip x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
    plt.title(f"# DORCs: (n >= {cutoff}) = {len(dorc_genes)}", fontsize=14)
    plt.xlabel("Ranked genes", fontsize=12)
    plt.ylabel("Number of correlated peaks", fontsize=12)
    plt.legend(title="isDORC", loc="upper left")
    plt.grid(visible=False)
    sns.despine()

    plt.show()
    if return_gene_list:
        return dorc_genes


def get_dorc_score(
    atac_adata,
    dorc,
    normalize_atac=True,
    gene_list=None,
):
    """
    Calculate the DORC score for each gene and cells

    Parameters
    ----------
    atac_adata: anndata.AnnData
        AnnData object containing ATAC-seq data
    dorc: pd.DataFrame
        DataFrame containing DORC results (post-filtering)
    normalize_atac: bool
        Whether to normalize ATAC-seq data before calculating correlation
    gene_list: list
        List of genes to calculate correlation. If None, all genes in dorc will be used

    Returns
    -------
    annadata.AnnData
        AnnData object containing DORC scores

    """
    assert {"PeakRanges", "Gene"}.issubset(
        dorc.columns
    ), "dorc must have columns: 'PeakRanges', 'Gene'"
    if gene_list is None:
        gene_list = list(dorc["Gene"].unique())
    all_peaks = set(atac_adata.var_names)
    assert all(
        [peak in all_peaks for peak in dorc["PeakRanges"].unique()]
    ), "All peaks in dorc must be present in atac_adata"
    all_gene = set(dorc["Gene"].unique())
    assert all(
        [gene in all_gene for gene in gene_list]
    ), "All gene in gene_list must be present in dorc"

    dorc_group = dorc.groupby("Gene")

    if normalize_atac:
        atac_mat = mean_norm_counts_array(atac_adata.X)
    else:
        atac_mat = atac_adata.X
    if issparse(atac_mat):
        atac_mat = atac_mat.tocsc()

    Xs = []
    atac_var = pd.DataFrame({"id_": np.arange(atac_adata.shape[1])}, index=atac_adata.var.index)
    bar = tqdm(gene_list)
    for gene in bar:
        bar.set_description(f"Processing {gene}")
        peaks = dorc_group.get_group(gene)["PeakRanges"]
        peaks_id = atac_var.loc[peaks, "id_"].values
        atac_gene = csr_matrix(atac_mat[:, peaks_id].sum(axis=1))
        Xs.append(atac_gene)
    Xs = hstack(Xs)
    dorc_adata = anndata.AnnData(Xs, obs=atac_adata.obs, var=pd.DataFrame(index=gene_list))

    return dorc_adata
