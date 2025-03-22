import os
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

pd.options.display.max_colwidth = 500
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import logomaker
from logomaker.src.Glyph import Glyph
from statsmodels.stats.multitest import fdrcorrection
from tangermeme.io import read_meme
from tangermeme.plot import plot_logo
from tangermeme.tools.tomtom import tomtom

try:
    from scprinter.motifs import read_pfms, tomtom_motif_motif_matrix
except:
    from ...motifs import read_pfms, tomtom_motif_motif_matrix

"""
Most of the code are from modiscolite, and adapted to include the delta effects.
Chose to copy code because modiscolite somehow changes the backend of matplotlib and inline version
"""


def generate_tomtom_dataframe(
    modisco_h5py: os.PathLike,
    meme_motif_db: Union[os.PathLike, None],
    pattern_groups: List[str],
    top_n_matches=3,
    trim_threshold=0.3,
    selected_pattern: set = None,
    prefix="",
):

    tomtom_results = {}

    for i in range(top_n_matches):
        tomtom_results[f"match{i}"] = []
        tomtom_results[f"qval{i}"] = []

    modisco_motifs = {}
    with h5py.File(modisco_h5py, "r") as modisco_results:
        for contribution_dir_name in pattern_groups:
            if contribution_dir_name not in modisco_results.keys():
                continue

            metacluster = modisco_results[contribution_dir_name]
            key = lambda x: int(x[0].split("_")[-1])

            for idx, (key_, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                # Rest of your code goes here
                if selected_pattern is not None:
                    # print (f"{contribution_dir_name}.{key_}", selected_pattern)
                    if f"{contribution_dir_name}.{key_}" not in selected_pattern:
                        continue
                ppm = np.array(pattern["sequence"][:])
                cwm = np.array(pattern["contrib_scores"][:])

                pattern_name = f"{contribution_dir_name}.pattern_{idx}"

                score = np.sum(np.abs(cwm), axis=1)
                trim_thresh = (
                    np.max(score) * trim_threshold
                )  # Cut off anything less than 30% of max score
                pass_inds = np.where(score >= trim_thresh)[0]
                trimmed = ppm[np.min(pass_inds) : np.max(pass_inds) + 1]
                modisco_motifs[pattern_name] = trimmed.T

        tomtom_res = tomtom_motif_motif_matrix(
            motifs_1=modisco_motifs, prefix_1=prefix, motifs_2=meme_motif_db
        )
        # print (tomtom_res)
        tomtom_res = tomtom_res.groupby("Query_ID")

        for pattern_name in modisco_motifs.keys():
            tp = tomtom_res.get_group(f"{prefix}{pattern_name}")
            tp = tp.sort_values("q-value").reset_index(drop=True)
            for i, row in tp.iloc[:top_n_matches].iterrows():
                tomtom_results[f"match{i}"].append(row["Target_ID"])
                tomtom_results[f"qval{i}"].append(row["q-value"])

    return pandas.DataFrame(tomtom_results)


def path_to_image_html(path):
    return '<img src="' + path + '" width="240" >'


class NoFussLogo(logomaker.Logo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_glyphs(self):
        """
        Specifies the placement and styling of all glyphs within the logo.
        """
        # Create a dataframe of glyphs
        glyph_df = pd.DataFrame(index=self.ps, columns=self.cs, dtype=object)

        # For each position
        for p in self.ps:

            # get values at this position
            vs = np.array(self.df.loc[p, :])

            # Sort values according to the order in which the user
            # wishes the characters to be stacked
            if self.stack_order == "big_on_top":
                ordered_indices = np.argsort(vs)

            elif self.stack_order == "small_on_top":
                tmp_vs = np.zeros(len(vs))
                indices = vs != 0
                tmp_vs[indices] = 1.0 / vs[indices]
                ordered_indices = np.argsort(tmp_vs)

            elif self.stack_order == "fixed":
                ordered_indices = np.array(range(len(vs)))[::-1]

            else:
                assert False, "This line of code should never be called."

            # Reorder values and characters
            vs = vs[ordered_indices]
            cs = [str(c) for c in self.cs[ordered_indices]]

            # Set floor
            floor = sum((vs - self.vsep) * (vs < 0)) + self.vsep / 2.0

            # For each character
            for v, c in zip(vs, cs):
                # Set ceiling
                ceiling = floor + abs(v)

                # Set color
                this_color = self.rgb_dict[c]

                # Set whether to flip character
                flip = v < 0 and self.flip_below

                # Create glyph if height is finite
                glyph = Glyph(
                    p,
                    c,
                    ax=self.ax,
                    floor=floor,
                    ceiling=ceiling,
                    color=this_color,
                    flip=flip,
                    zorder=self.zorder,
                    font_name=self.font_name,
                    alpha=self.alpha,
                    vpad=self.vpad,
                    **self.glyph_kwargs,
                )

                # Add glyph to glyph_df
                glyph_df.loc[p, c] = glyph

                # Raise floor to current ceiling
                floor = ceiling + self.vsep

        # Set glyph_df attribute
        self.glyph_df = glyph_df
        self.glyph_list = [g for g in self.glyph_df.values.ravel() if isinstance(g, Glyph)]


def _plot_weights(array, path, figsize=(10, 3)):
    """Plot weights as a sequence logo and save to file."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    df = pandas.DataFrame(array, columns=["A", "C", "G", "T"])
    df.index.name = "pos"

    crp_logo = NoFussLogo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())
    # plot_logo(array.T, ax=ax)
    plt.savefig(path)
    plt.close()


def make_logo(
    match,
    ppm,
    logo_dir,
):
    if match == "NA":
        return

    background = np.array([0.25, 0.25, 0.25, 0.25])
    ic = compute_per_position_ic(ppm, background, 0.001)

    _plot_weights(ppm * ic[:, None], path="{}/{}.png".format(logo_dir, match))


def _plot_weights(array, path, figsize=(10, 3)):
    """Plot weights as a sequence logo and save to file."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    df = pandas.DataFrame(array, columns=["A", "C", "G", "T"])
    df.index.name = "pos"

    crp_logo = NoFussLogo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())
    # plot_logo(array.T, ax=ax)
    plt.savefig(path)
    plt.close()


def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = (np.log((ppm + pseudocount) / (1 + pseudocount * alphabet_len)) / np.log(2)) * ppm - (
        np.log(background) * background / np.log(2)
    )[None, :]
    return np.sum(ic, axis=1)


def write_meme_file(ppm, bg, fname):
    f = open(fname, "w")
    f.write("MEME version 4\n\n")
    f.write("ALPHABET= ACGT\n\n")
    f.write("strands: + -\n\n")
    f.write("Background letter frequencies (from unknown source):\n")
    f.write("A %.3f C %.3f G %.3f T %.3f\n\n" % tuple(list(bg)))
    f.write("MOTIF 1 TEMP\n\n")
    f.write("letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n" % ppm.shape[0])
    for s in ppm:
        f.write("%.5f %.5f %.5f %.5f\n" % tuple(s))
    f.close()


def create_modisco_logos(
    modisco_h5py: os.PathLike,
    modisco_logo_dir,
    trim_threshold,
    pattern_groups: List[str],
    selected_patterns: set = None,
):
    """Open a modisco results file and create and write logos to file for each pattern."""
    modisco_results = h5py.File(modisco_h5py, "r")

    tags = []
    pool = ProcessPoolExecutor(max_workers=32)
    p_list = []
    for name in pattern_groups:
        if name not in modisco_results.keys():
            continue

        metacluster = modisco_results[name]
        key = lambda x: int(x[0].split("_")[-1])
        for pattern_name, pattern in sorted(metacluster.items(), key=key):
            tag = "{}.{}".format(name, pattern_name)
            if selected_patterns is not None:
                if tag not in selected_patterns:
                    continue
            tags.append(tag)

            cwm_fwd = np.array(pattern["contrib_scores"][:])
            cwm_rev = cwm_fwd[::-1, ::-1]

            score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
            score_rev = np.sum(np.abs(cwm_rev), axis=1)

            trim_thresh_fwd = np.max(score_fwd) * trim_threshold
            trim_thresh_rev = np.max(score_rev) * trim_threshold

            pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
            pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

            start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(
                np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1
            )
            start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(
                np.max(pass_inds_rev) + 4 + 1, len(score_rev) + 1
            )

            ppm_fwd = np.array(pattern["sequence"][:])
            ppm_rev = ppm_fwd[::-1, ::-1]

            trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
            trimmed_cwm_rev = cwm_rev[start_rev:end_rev]
            #
            # trimmed_ppm_fwd = ppm_fwd[start_fwd:end_fwd]
            # trimmed_ppm_rev = ppm_rev[start_rev:end_rev]
            #
            # background = np.array([0.25, 0.25, 0.25, 0.25])
            # ppm = trimmed_ppm_fwd
            # ic = compute_per_position_ic(ppm, background, 0.001)
            # _plot_weights(ppm*ic[:, None], path='{}/{}.cwm.fwd.pdf'.format(modisco_logo_dir, tag))
            # _plot_weights(ppm * ic[:, None], path='{}/{}.cwm.fwd.png'.format(modisco_logo_dir, tag))
            #
            # background = np.array([0.25, 0.25, 0.25, 0.25])
            # ppm = trimmed_ppm_rev
            # ic = compute_per_position_ic(ppm, background, 0.001)
            # _plot_weights(ppm*ic[:, None], path='{}/{}.cwm.rev.pdf'.format(modisco_logo_dir, tag))
            # _plot_weights(ppm * ic[:, None], path='{}/{}.cwm.rev.png'.format(modisco_logo_dir, tag))
            p_list.append(
                pool.submit(
                    _plot_weights,
                    trimmed_cwm_fwd,
                    "{}/{}.cwm.fwd.png".format(modisco_logo_dir, tag),
                )
            )
            p_list.append(
                pool.submit(
                    _plot_weights,
                    trimmed_cwm_rev,
                    "{}/{}.cwm.rev.png".format(modisco_logo_dir, tag),
                )
            )
            p_list.append(
                pool.submit(
                    _plot_weights,
                    trimmed_cwm_fwd,
                    "{}/{}.cwm.fwd.pdf".format(modisco_logo_dir, tag),
                )
            )
            p_list.append(
                pool.submit(
                    _plot_weights,
                    trimmed_cwm_rev,
                    "{}/{}.cwm.rev.pdf".format(modisco_logo_dir, tag),
                )
            )
            # _plot_weights(trimmed_cwm_fwd, path="{}/{}.cwm.fwd.pdf".format(modisco_logo_dir, tag))
            # _plot_weights(trimmed_cwm_rev, path="{}/{}.cwm.rev.pdf".format(modisco_logo_dir, tag))
            # _plot_weights(trimmed_cwm_fwd, path="{}/{}.cwm.fwd.png".format(modisco_logo_dir, tag))
            # _plot_weights(trimmed_cwm_rev, path="{}/{}.cwm.rev.png".format(modisco_logo_dir, tag))

    modisco_results.close()
    return tags


def report_motifs(
    modisco_h5pys: Path,
    output_dir: os.PathLike,
    summary_file_name: str,
    img_path_suffixs: os.PathLike,
    meme_motif_db: Union[os.PathLike, None],
    top_n_matches=3,
    delta_effect_paths: os.PathLike = None,
    motif_prefixs="",
    trim_threshold=0.3,
    selected_patterns: List[str] = None,
):
    """
    Adapted from modisco-lite, but make it able to accept multiple h5s

    Parameters
    ----------
    modisco_h5py
    output_dir
    summary_file_name
    img_path_suffix
    meme_motif_db
    is_writing_tomtom_matrix
    top_n_matches
    delta_effect_path
    motif_prefixs
    trim_threshold
    trim_min_length

    Returns
    -------

    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    pattern_groups = ["pos_patterns", "neg_patterns"]

    if meme_motif_db is not None:
        motifs = read_pfms(meme_motif_db)
        motifs = {key: motif.T for key, motif in motifs.items()}
        logo_dir = os.path.join(output_dir, "ref_motif_logos")
        if not os.path.isdir(logo_dir):
            os.mkdir(logo_dir)
        process_map(
            partial(make_logo, logo_dir=logo_dir),
            motifs.keys(),
            motifs.values(),
            desc="Creating logos for reference motifs",
        )

    df_all = []
    for (
        modisco_h5py,
        img_path_suffix,
        delta_effect_path,
        motif_prefix,
        selected_pattern,
    ) in zip(modisco_h5pys, img_path_suffixs, delta_effect_paths, motif_prefixs, selected_patterns):
        print("Processing", modisco_h5py)
        selected_pattern = set(selected_pattern) if selected_pattern is not None else None
        results = {
            "pattern": [],
            "num_seqlets": [],
            "modisco_cwm_fwd": [],
            "modisco_cwm_rev": [],
            "delta_effects": [],
        }
        modisco_logo_dir = os.path.join(img_path_suffix, "trimmed_logos")
        if not os.path.isdir(modisco_logo_dir):
            os.mkdir(modisco_logo_dir)
        create_modisco_logos(
            modisco_h5py,
            modisco_logo_dir,
            trim_threshold,
            pattern_groups,
            selected_patterns=selected_pattern,
        )

        with h5py.File(modisco_h5py, "r") as modisco_results:
            for name in pattern_groups:
                if name not in modisco_results.keys():
                    continue

                metacluster = modisco_results[name]
                key = lambda x: int(x[0].split("_")[-1])
                for pattern_name, pattern in sorted(metacluster.items(), key=key):
                    num_seqlets = pattern["seqlets"]["n_seqlets"][:][0]
                    pattern_tag = f"{motif_prefix}{name}.{pattern_name}"
                    if selected_pattern is not None:
                        if f"{name}.{pattern_name}" not in selected_pattern:
                            continue
                    results["pattern"].append(pattern_tag)
                    results["num_seqlets"].append(num_seqlets)
                    results["modisco_cwm_fwd"].append(
                        os.path.join(
                            img_path_suffix, "trimmed_logos", f"{name}.{pattern_name}.cwm.fwd.png"
                        )
                    )
                    results["modisco_cwm_rev"].append(
                        os.path.join(
                            img_path_suffix, "trimmed_logos", f"{name}.{pattern_name}.cwm.rev.png"
                        )
                    )
                    results["delta_effects"].append(
                        os.path.join(delta_effect_path, f"{pattern_tag}.png")
                    )

        patterns_df = pd.DataFrame(results)
        reordered_columns = [
            "pattern",
            "num_seqlets",
            "modisco_cwm_fwd",
            "modisco_cwm_rev",
            "delta_effects",
        ]

        # If the optional meme_motif_db is not provided, then we won't generate TOMTOM comparison.
        if meme_motif_db is not None:

            tomtom_df = generate_tomtom_dataframe(
                modisco_h5py,
                meme_motif_db,
                top_n_matches=top_n_matches,
                pattern_groups=pattern_groups,
                trim_threshold=trim_threshold,
                selected_pattern=selected_pattern,
                prefix=motif_prefix,
            )
            patterns_df = pandas.concat([patterns_df, tomtom_df], axis=1)

        for i in range(top_n_matches):
            name = f"match{i}"
            logos = []

            for _, row in patterns_df.iterrows():
                if name in patterns_df.columns:
                    if pandas.isnull(row[name]):
                        logos.append("NA")
                    else:

                        logos.append(os.path.join(logo_dir, f"{row[name]}.png"))
                else:
                    break

            patterns_df[f"{name}_logo"] = logos
            reordered_columns.extend([name, f"qval{i}", f"{name}_logo"])

        patterns_df = patterns_df[reordered_columns]
        df_all.append(patterns_df)
    patterns_df = pd.concat(df_all, axis=0)
    patterns_df.to_html(
        open(os.path.join(output_dir, f"{summary_file_name}.html"), "w"),
        escape=False,
        formatters=dict(
            modisco_cwm_fwd=path_to_image_html,
            modisco_cwm_rev=path_to_image_html,
            delta_effects=path_to_image_html,
            match0_logo=path_to_image_html,
            match1_logo=path_to_image_html,
            match2_logo=path_to_image_html,
        ),
        index=False,
    )

    from weasyprint import CSS, HTML

    additional_css = """
	@page {
			size: 2400mm 1300mm;
			margin: 0in 0in 0in 0in;
  }
	@media print {
    /* Set font size for print. You can remove this if not needed. */
    table {
        font-size: 36pt;
        /* Center the table itself within the page. */
        margin-left: auto;
        margin-right: auto;
    }
    /* Center-align text and images in each cell. */
    th, td {
        text-align: center;
        vertical-align: middle;
    }
    }
	"""
    css = CSS(string=additional_css)
    HTML(os.path.join(output_dir, f"{summary_file_name}.html")).write_pdf(
        os.path.join(output_dir, f"{summary_file_name}.pdf"), stylesheets=[css]
    )
    return patterns_df


def modisco_df_to_pdf(patterns_df, output_dir, summary_file_name):
    patterns_df.to_html(
        open(os.path.join(output_dir, f"{summary_file_name}.html"), "w"),
        escape=False,
        formatters=dict(
            modisco_cwm_fwd=path_to_image_html,
            modisco_cwm_rev=path_to_image_html,
            delta_effects=path_to_image_html,
            match0_logo=path_to_image_html,
            match1_logo=path_to_image_html,
            match2_logo=path_to_image_html,
        ),
        index=False,
    )
    from weasyprint import CSS, HTML

    additional_css = """
    	@page {
    			size: 3200mm 1300mm;
    			margin: 0in 0in 0in 0in;
      }
    	@media print {
        /* Set font size for print. You can remove this if not needed. */
        table {
            font-size: 36pt;
            /* Center the table itself within the page. */
            margin-left: auto;
            margin-right: auto;
        }
        /* Center-align text and images in each cell. */
        th, td {
            text-align: center;
            vertical-align: middle;
        }
        }
    	"""
    css = CSS(string=additional_css)
    HTML(os.path.join(output_dir, f"{summary_file_name}.html")).write_pdf(
        os.path.join(output_dir, f"{summary_file_name}.pdf"), stylesheets=[css]
    )


def report_composite_motif(
    modisco_h5pys: Path,
    output_dir: os.PathLike,
    summary_file_name: str,
    img_path_suffixs: os.PathLike,
    meme_motif_db: Union[os.PathLike, None],
    delta_effect_paths: os.PathLike = None,
    motif_prefixs="",
    qval_threshold=1e-3,
    trim_threshold=0.3,
    n_top_hits=1,
    selected_patterns: List[str] = None,
):
    """
    Beta testing don't use
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    pattern_groups = ["pos_patterns", "neg_patterns"]

    if meme_motif_db is not None:
        motifs = read_pfms(meme_motif_db)
        motifs = {key: motif.T for key, motif in motifs.items()}
        logo_dir = os.path.join(output_dir, "ref_motif_logos")
        if not os.path.isdir(logo_dir):
            os.mkdir(logo_dir)
        process_map(
            partial(make_logo, logo_dir=logo_dir),
            motifs.keys(),
            motifs.values(),
            desc="Creating logos for reference motifs",
        )

    df_all = []
    for (
        modisco_h5py,
        img_path_suffix,
        delta_effect_path,
        motif_prefix,
        selected_pattern,
    ) in zip(modisco_h5pys, img_path_suffixs, delta_effect_paths, motif_prefixs, selected_patterns):
        print("Processing", modisco_h5py)
        if delta_effect_path is None:
            delta_effect_path = ""
        selected_pattern = set(selected_pattern) if selected_pattern is not None else None

        match_res = composite_motif_calling(
            modisco_h5py,
            meme_motif_db,
            trim_threshold,
            selected_pattern,
            qval_threshold,
            n_top_hits,
        )
        # match_res.index = match_res['Query_ID']
        selected_pattern = set(match_res["Query_ID"])

        results = {
            "pattern": [],
            "num_seqlets": [],
            "modisco_cwm_fwd": [],
            "modisco_cwm_rev": [],
            "delta_effects": [],
        }
        modisco_logo_dir = os.path.join(img_path_suffix, "trimmed_logos")
        if not os.path.isdir(modisco_logo_dir):
            os.mkdir(modisco_logo_dir)
        create_modisco_logos(
            modisco_h5py,
            modisco_logo_dir,
            trim_threshold,
            pattern_groups,
            selected_patterns=selected_pattern,
        )
        indexs = []
        with h5py.File(modisco_h5py, "r") as modisco_results:
            for name in pattern_groups:
                if name not in modisco_results.keys():
                    continue

                metacluster = modisco_results[name]
                key = lambda x: int(x[0].split("_")[-1])
                for pattern_name, pattern in sorted(metacluster.items(), key=key):
                    num_seqlets = pattern["seqlets"]["n_seqlets"][:][0]
                    pattern_tag = f"{motif_prefix}{name}.{pattern_name}"

                    if selected_pattern is not None:
                        if f"{name}.{pattern_name}" not in selected_pattern:
                            continue
                    indexs.append(f"{name}.{pattern_name}")
                    results["pattern"].append(pattern_tag)
                    results["num_seqlets"].append(num_seqlets)
                    results["modisco_cwm_fwd"].append(
                        os.path.join(
                            img_path_suffix, "trimmed_logos", f"{name}.{pattern_name}.cwm.fwd.png"
                        )
                    )
                    results["modisco_cwm_rev"].append(
                        os.path.join(
                            img_path_suffix, "trimmed_logos", f"{name}.{pattern_name}.cwm.rev.png"
                        )
                    )
                    results["delta_effects"].append(
                        os.path.join(delta_effect_path, f"{pattern_tag}.png")
                    )

        patterns_df = pd.DataFrame(results).reset_index(drop=True)
        patterns_df.index = indexs
        patterns_df = patterns_df.loc[match_res["Query_ID"].values].copy().reset_index(drop=True)
        patterns_df = pd.concat([patterns_df, match_res], axis=1)
        reordered_columns = [
            "pattern",
            "num_seqlets",
            "modisco_cwm_fwd",
            "modisco_cwm_rev",
            "delta_effects",
        ]
        for i in range(1, 3):
            name = f"Target_ID{i}"
            logos = []

            for _, row in patterns_df.iterrows():
                if pandas.isnull(row[name]):
                    logos.append("NA")
                else:

                    logos.append(os.path.join(logo_dir, f"{row[name]}.png"))

            patterns_df[f"{name}_logo"] = logos
            reordered_columns.extend([name, f"q-value{i}", f"{name}_logo"])
        reordered_columns.append("consensus_seq")
        patterns_df = patterns_df[reordered_columns]
        df_all.append(patterns_df)
    patterns_df = pd.concat(df_all, axis=0)
    patterns_df.sort_values("num_seqlets", ascending=False, inplace=True)
    patterns_df.to_html(
        open(os.path.join(output_dir, f"{summary_file_name}.html"), "w"),
        escape=False,
        formatters=dict(
            modisco_cwm_fwd=path_to_image_html,
            modisco_cwm_rev=path_to_image_html,
            delta_effects=path_to_image_html,
            Target_ID1_logo=path_to_image_html,
            Target_ID2_logo=path_to_image_html,
        ),
        index=False,
    )

    from weasyprint import CSS, HTML

    additional_css = """
    	@page {
    			size: 2400mm 1300mm;
    			margin: 0in 0in 0in 0in;
      }
    	@media print {
    		table { font-size: 36pt; }
    	}
    	"""
    css = CSS(string=additional_css)
    HTML(os.path.join(output_dir, f"{summary_file_name}.html")).write_pdf(
        os.path.join(output_dir, f"{summary_file_name}.pdf"), stylesheets=[css]
    )
    return patterns_df


def composite_motif_calling(
    modisco_h5py: Path,
    meme_motif_db: Union[os.PathLike, None],
    trim_threshold=0.3,
    selected_pattern: List[str] = None,
    qval_threshold=1e-3,
    n_top_hits=1,
):
    """
    Beta testing don't use
    """
    query_motif = read_pfms(modisco_h5py, trim_threshold=trim_threshold)
    if selected_pattern is not None:
        query_motif = {key: motif for key, motif in query_motif.items() if key in selected_pattern}
    target_motif = read_pfms(meme_motif_db)
    tomtom_res = tomtom_motif_motif_matrix(
        motifs_1=query_motif, prefix_1="", motifs_2=target_motif, trim_threshold=trim_threshold
    )
    tomtom_res = tomtom_res[tomtom_res["q-value"] < qval_threshold]
    tomtom_res_group = (
        tomtom_res.sort_values("q-value", ascending=True).reset_index(drop=True).groupby("Query_ID")
    )
    query_trim_motif = []
    for i in range(n_top_hits):
        query_trim_motif.append({})
    for group_name, group_df in tomtom_res_group:
        mm = query_motif[group_name].copy()
        for i in range(min(n_top_hits, len(group_df))):
            offset = int(group_df["Optimal_offset"].iloc[i])
            overlap = int(group_df["Overlap"].iloc[i])
            if overlap <= 5:
                continue
            if offset < 0:
                offset = -1 * offset
            else:
                offset = 0
            if overlap == mm.shape[1]:
                continue
            if offset <= 3:
                mm = mm[:, offset + overlap - 2 :]
            elif offset + overlap >= (mm.shape[1] - 3):
                mm = mm[:, : offset - 2]
            else:
                mm[:, offset : offset + overlap] = 0.25

            query_trim_motif[i][group_name] = mm

    final_df = []
    for i in range(n_top_hits):
        if len(query_trim_motif[i]) == 0:
            continue
        tomtom_res_new = tomtom_motif_motif_matrix(
            motifs_1=query_trim_motif[i],
            prefix_1="",  # prefix is added the first time
            motifs_2=target_motif,
            trim_threshold=trim_threshold,
        )
        tomtom_res_new = tomtom_res_new[tomtom_res_new["q-value"] < qval_threshold]
        if len(tomtom_res_new) == 0:
            continue
        tomtom_res_new_group = (
            tomtom_res_new.sort_values("q-value", ascending=True)
            .reset_index(drop=True)
            .groupby("Query_ID")
        )
        for group_name, group_df2 in tomtom_res_new_group:
            group_df1 = tomtom_res_group.get_group(group_name)
            for j in range(min(n_top_hits, len(group_df2))):
                motif1 = group_df1["Target_ID"].iloc[i]
                motif2 = group_df2["Target_ID"].iloc[j]
                qval1 = group_df1["q-value"].iloc[i]
                qval2 = group_df2["q-value"].iloc[j]
                overlap1 = group_df1["Overlap"].iloc[i]
                overlap2 = group_df2["Overlap"].iloc[j]
                offset1 = group_df1["Optimal_offset"].iloc[i]
                offset2 = group_df2["Optimal_offset"].iloc[j]
                orientation1 = group_df1["Orientation"].iloc[i]
                orientation2 = group_df2["Orientation"].iloc[j]
                final_df.append(
                    [
                        group_name,
                        motif1,
                        motif2,
                        qval1,
                        qval2,
                        overlap1,
                        overlap2,
                        offset1,
                        offset2,
                        orientation1,
                        orientation2,
                    ]
                )

    final_df = pd.DataFrame(
        final_df,
        columns=[
            "Query_ID",
            "Target_ID1",
            "Target_ID2",
            "q-value1",
            "q-value2",
            "Overlap1",
            "Overlap2",
            "Optimal_offset1",
            "Optimal_offset2",
            "Orientation1",
            "Orientation2",
        ],
    )
    selected_pattern = set(final_df["Query_ID"])
    final = {key: motif for key, motif in query_motif.items() if key in selected_pattern}
    consensus_seq = []
    nucleotide = np.array(list("ACGT"))
    for key in final_df["Query_ID"]:
        motif = final[key]
        seq = np.argmax(motif, axis=0)
        seq = list(nucleotide[seq])
        consensus_seq.append("".join(seq))
    final_df["consensus_seq"] = consensus_seq

    return final_df
