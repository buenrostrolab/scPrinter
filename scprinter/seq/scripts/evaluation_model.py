import argparse
import gc
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import h5py

import scprinter as scp
from scprinter.seq.dataloader import *
from scprinter.seq.interpretation.attribution_wrapper import *
from scprinter.seq.interpretation.attributions import *
from scprinter.seq.Models import *

parser = argparse.ArgumentParser(description="Run DeepLIFT on trained seq2PRINT model")

parser.add_argument("--pt", type=str, default="config.JSON", help="model.pt")
parser.add_argument("--models", type=str, default=None, help="model")
parser.add_argument("--genome", type=str, default="hg38", help="genome")
parser.add_argument("--peaks", type=str, default=None, help="peaks")
parser.add_argument("--start", type=int, default=0, help="start index of peaks")
parser.add_argument("--end", type=int, default=-1, help="end index of peaks")
parser.add_argument("--method", type=str, default=None, help="method")
parser.add_argument("--wrapper", type=str, default=None, help="method")
parser.add_argument("--nth_output", type=str, default=None, help="nth output")
parser.add_argument("--write_numpy", action="store_true", default=False, help="write numpy")
parser.add_argument("--gpus", nargs="+", type=int, help="gpus")
parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite")
parser.add_argument("--decay", type=float, default=None, help="decay")
parser.add_argument("--extra", type=str, default="", help="extra")
parser.add_argument("--model_norm", type=str, default=None, help="key for model norm")
parser.add_argument("--sample", type=int, default=None, help="sample")
parser.add_argument("--silent", action="store_true", default=False, help="silent")
parser.add_argument(
    "--save_norm", action="store_true", default=False, help="just save the normalization factor"
)
parser.add_argument("--save_key", type=str, default="deepshap", help="key for saving")
parser.add_argument("--save_names", type=str, default="", help="save names")


# ((start != 0) or (end != n_summits))
# After calculating the sequence attrs, save them with npz or bigwig, normalize them if needed
def save_attrs(
    projected_attributions,
    hypo,
    norm_key,
    acc_model,
    id,
    id_str,
    wrapper,
    method,
    extra,
    start,
    end,
    partition_by_region,
    decay,
    regions,
    verbose,
    write_bigwig,
    chrom_size,
    save_dir,
):
    print("saving attrs")
    # slice the center regions of the projected attributions
    vs = projected_attributions[..., 520:-520]

    if verbose:
        print(
            "the estimated norm on prem is",
            np.quantile(vs, 0.05),
            np.quantile(vs, 0.5),
            np.quantile(vs, 0.95),
        )

    # fetch proper normalization factors
    if norm_key is not None:
        print("using pre-calculated norm key", norm_key)
        if norm_key == "count":
            low, median, high = acc_model.count_norm
        elif norm_key == "footprint":
            low, median, high = acc_model.foot_norm
    else:
        print("using calculated norm from the model")
        low, median, high = (
            np.quantile(vs, 0.05),
            np.quantile(vs, 0.5),
            np.quantile(vs, 0.95),
        )

    # Create the proper file name,
    # if we are doing the lora model, we need to add the model_id
    filename_template = f"model_{id_str}." if id[0] is not None else ""
    filename_template += "{type}."
    filename_template += f"{wrapper}.{method}{extra}.{decay}."
    # If we are doing the bulk model with part of the peaks (parallel in terms of peaks)
    # we need to add the start and end
    if (id[0] is None) and partition_by_region:
        filename_template += f"{start}-{end}."
    else:
        # Only normalize the projected attributions if we have all the results already,
        projected_attributions = (projected_attributions - median) / (high - low)
        if verbose:
            print("normalizing", low, median, high)
    print("filename_template", filename_template)
    if write_bigwig:
        attribution_to_bigwig(
            projected_attributions,
            pd.DataFrame(regions),
            chrom_size,
            res=1,
            mode="average",
            output=os.path.join(
                save_dir,
                filename_template.replace("{type}", "attr") + "bigwig",
            ),
            verbose=verbose,
        )
    else:
        np.savez(
            os.path.join(save_dir, filename_template.replace("{type}", "attr") + "npz"),
            projected_attributions,
        )
    np.savez(
        os.path.join(
            save_dir,
            filename_template.replace("{type}", "hypo") + "npz",
        ),
        hypo,
    )
    print(filename_template.replace("{type}", "hypo") + "npz")


def main(
    pt,
    models,
    genome,
    peaks,
    start,
    end,
    method,
    wrapper,
    nth_output,
    write_numpy,
    gpus,
    overwrite,
    decay,
    extra,
    model_norm,
    sample,
    silent,
    save_norm,
    save_key,
    save_names,
):

    torch.set_num_threads(4)
    verbose = not silent
    write_bigwig = not write_numpy
    if models is not None:
        ids = models
        ids = ids.split(",")
        ids = [i.split("-") for i in ids]
        print(ids[:5], ids[-5:])
        ids = [[int(j) for j in i] for i in ids]
        id_strs = save_names.split(",")
        assert len(id_strs) == len(ids)
    else:
        ids = [[None]]
        id_strs = [""]

    print(ids)
    gpus = gpus
    wrapper = wrapper
    method = method
    nth_output = nth_output
    norm_key = model_norm

    print(gpus)
    if len(gpus) == 1:
        torch.cuda.set_device(int(gpus[0]))
    summits = pd.read_table(peaks, sep="\t", header=None)
    summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
    summits["summits"] = (summits[1] + summits[2]) // 2
    summits = summits[[0, "summits"]]
    summits["summits"] = np.array(summits["summits"], dtype=int)

    acc_model = torch.load(pt, map_location="cpu", weights_only=False).cuda()

    # If there's coverage, set it to be the same across all models (so there won't be a coverage bias)
    if acc_model.coverages is not None:
        print("setting coverage to be the same")
        mm = acc_model.coverages.weight.data.mean(dim=0)
        acc_model.coverages.weight.data = (
            torch.ones_like(acc_model.coverages.weight.data) * mm[None]
        )
    acc_model.eval()
    dna_len = acc_model.dna_len

    signal_window = 1000
    print("signal_window", signal_window, "dna_len", dna_len)
    genome_str = deepcopy(genome)
    if genome == "hg38":
        genome = scp.genome.hg38
    elif genome == "mm10":
        genome = scp.genome.mm10
    else:
        print("genome not in ['hg38', mm10'], reading as custom genome")
        try:
            genome = pickle.load(open(genome, "rb"))
        except:
            print("error loading custom genome")
            raise ValueError("genome not supported")

    bias = str(genome.fetch_bias_bw())
    signals = [bias, bias]

    # Either way we iterate over the ids and calculate the attributions
    dataset = seq2PRINTDataset(
        signals=signals,
        ref_seq=genome.fetch_fa(),
        summits=summits,
        DNA_window=dna_len,
        signal_window=signal_window,
        max_jitter=0,
        min_counts=None,
        max_counts=None,
        cached=False,
        reverse_compliment=False,
        device="cpu",
        verbose=verbose,
    )
    summits = dataset.summits
    if end == -1:
        end = summits.shape[0]
    summits = summits[start:end]
    regions = np.array(
        [summits[:, 0], summits[:, 1] - dna_len // 2, summits[:, 1] + dna_len // 2]
    ).T
    acc_model.upsample = False

    save_dir = f"{pt}_{save_key}" + (f"_sample{sample}" if sample is not None else "")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    vs_collection = []
    extra = extra + f"_{nth_output}_"

    if len(gpus) > 1:
        n_split = len(gpus)
        if ids[0][0] is not None:  # This is the lora mode, parallel in terms of lora models
            unfinished_ids = []
            unfinished_id_strs = []
            for id, id_str in zip(ids, id_strs):
                # id_str = "-".join([str(x) for x in id])
                if os.path.exists(
                    os.path.join(
                        save_dir,
                        f"model_{id_str}.hypo.{wrapper}.{method}{extra}.{decay}.npz",
                    )
                ) and (not overwrite):
                    print("exists")
                    continue
                else:
                    unfinished_ids.append(id)
                    unfinished_id_strs.append(id_str)
            unfinished_ids = np.array(unfinished_ids, dtype="object")
            ids_batch = np.array_split(unfinished_ids, n_split)
            ids_strs_batch = np.array_split(unfinished_id_strs, n_split)

            commands = [
                [
                    "seq2print_attr",
                    "--pt",
                    pt,
                    "--genome",
                    genome_str,
                    "--peaks",
                    peaks,
                    "--models",
                    ",".join(["-".join([str(j) for j in i]) for i in id_batch]),
                    "--method",
                    method,
                    "--wrapper",
                    wrapper,
                    "--nth_output",
                    str(nth_output),
                    "--gpus",
                    str(gpu),
                    "--decay",
                    str(decay),
                    "--save_key",
                    save_key,
                    "--save_names",
                    ",".join(list(ids_str_batch)),
                ]
                + (["--overwrite"] if overwrite else [])
                + (["--write_numpy"] if write_numpy else [])
                + (["--model_norm", model_norm] if model_norm is not None else [])
                + (["--silent"] if silent else [])
                for i, (id_batch, ids_str_batch, gpu) in enumerate(
                    zip(ids_batch, ids_strs_batch, gpus)
                )
            ]
            pool = ProcessPoolExecutor(max_workers=len(gpus))
            for gpu, command in zip(gpus, commands):
                pool.submit(subprocess.run, command)
            pool.shutdown(wait=True)
        else:
            # parallel in terms of peaks
            if os.path.exists(
                os.path.join(
                    save_dir,
                    f"hypo.{wrapper}.{method}{extra}.{decay}.npz",
                )
            ) and (not overwrite):
                print("exists")
            else:
                bs = int(math.ceil(len(summits) / n_split))
                start_batches = [i * bs for i in range(n_split)]
                end_batches = [(i + 1) * bs for i in range(n_split)]

                commands = [
                    [
                        "seq2print_attr",
                        "--pt",
                        pt,
                        "--genome",
                        genome_str,
                        "--peaks",
                        peaks,
                        "--start",
                        str(start_),
                        "--end",
                        str(end_),
                        "--method",
                        method,
                        "--write_numpy",
                        "--wrapper",
                        wrapper,
                        "--nth_output",
                        str(nth_output),
                        "--gpus",
                        str(gpu),
                        "--decay",
                        str(decay),
                        "--save_key",
                        save_key,
                    ]
                    + (["--overwrite"] if overwrite else [])
                    + (["--model_norm", model_norm] if model_norm is not None else [])
                    + (["--silent"] if silent else [])
                    for i, (start_, end_, gpu) in enumerate(zip(start_batches, end_batches, gpus))
                ]
                pool = ProcessPoolExecutor(max_workers=len(gpus))
                for gpu, command in zip(gpus, commands):
                    pool.submit(subprocess.run, command)
                pool.shutdown(wait=True)

                # parallel in terms of peaks, merge them:
                attributions = []
                hypo, ohe = [], []
                for start_, end_ in zip(start_batches, end_batches):
                    attributions.append(
                        np.load(
                            os.path.join(
                                save_dir,
                                f"attr.{wrapper}.{method}{extra}.{decay}.{start_}-{end_}.npz",
                            )
                        )["arr_0"]
                    )
                    hypo.append(
                        np.load(
                            os.path.join(
                                save_dir,
                                f"hypo.{wrapper}.{method}{extra}.{decay}.{start_}-{end_}.npz",
                            )
                        )["arr_0"]
                    )

                projected_attributions = np.concatenate(attributions, axis=0)
                hypo = np.concatenate(hypo, axis=0)

                save_attrs(
                    projected_attributions=projected_attributions,
                    hypo=hypo,
                    norm_key=norm_key,
                    acc_model=acc_model,
                    id=[None],  # pass none because it's definitely parallel in peaks mode
                    id_str="",
                    wrapper=wrapper,
                    method=method,
                    extra=extra,
                    start=0,
                    end=regions.shape[0],
                    partition_by_region=False,  # we collected them all
                    decay=decay,
                    regions=regions,
                    verbose=verbose,
                    write_bigwig=write_bigwig,
                    chrom_size=genome.chrom_sizes,
                    save_dir=save_dir,
                )

                for start_, end_ in zip(start_batches, end_batches):
                    os.remove(
                        os.path.join(
                            save_dir,
                            f"attr.{wrapper}.{method}{extra}.{decay}.{start_}-{end_}.npz",
                        )
                    )
                    os.remove(
                        os.path.join(
                            save_dir,
                            f"hypo.{wrapper}.{method}{extra}.{decay}.{start_}-{end_}.npz",
                        )
                    )

    else:
        # single gpu
        bar = trange(len(ids))
        dataset.cache()  # only cache when single gpu and can
        for index in bar:
            id = ids[index]
            id_str = id_strs[index]
            # id_str = "-".join([str(i) for i in id]) if id[0] is not None else None
            bar.set_description(f"working on {id_str}")
            if os.path.exists(
                os.path.join(
                    save_dir,
                    (f"model_{id_str}." if id[0] is not None else "")
                    + f"hypo.{wrapper}.{method}{extra}.{decay}.npz",
                )
            ) and (not overwrite):
                print("exists")
                continue

            model_0 = acc_model if id[0] is None else acc_model.collapse([int(i) for i in id])
            model_0 = model_0.cuda()
            model_0.eval()

            if type(nth_output) is not torch.Tensor:
                if "," in nth_output:
                    nth_output = nth_output.split(",")
                    nth_output = torch.as_tensor([int(i) for i in nth_output])
                elif "-" in nth_output:
                    nth_output_start, nth_output_end = nth_output.split("-")
                    nth_output = torch.as_tensor(
                        [i for i in range(int(nth_output_start), int(nth_output_end))]
                    )
                else:
                    nth_output = torch.as_tensor([int(nth_output)])
                if nth_output[0] < 0:
                    nth_output = None
            if wrapper == "just_sum":
                model = JustSumWrapper(model_0, nth_output=nth_output, threshold=0.301)
            elif wrapper == "count":
                model = CountWrapper(model_0)
            model = model.cuda()

            # if sample, randomly select some regions
            if sample is not None:
                random_ids = np.random.permutation(np.arange(start, min(end, regions.shape[0])))[
                    :sample
                ]
                random_ids = torch.from_numpy(random_ids)
            elif (start == 0) and (end == dataset.summits.shape[0]):
                random_ids = slice(None)
            else:
                random_ids = slice(start, end)

            attributions = calculate_attributions(
                model,
                X=dataset.cache_seqs[random_ids],
                n_shuffles=20,
                method=method,
                verbose=False if len(ids) > 1 else verbose,
            )
            projected_attributions = project_attrs(
                attributions, dataset.cache_seqs[random_ids], 64, "cuda"
            )
            hypo, ohe = (
                attributions.detach().cpu().numpy(),
                dataset.cache_seqs[random_ids].detach().cpu().numpy(),
            )

            if save_norm:
                vs = projected_attributions[..., 520:-520]
                vs_collection.append(
                    vs
                )  # usually when downsampling we just save the normalization factor
                # continue
            save_attrs(
                projected_attributions=projected_attributions,
                hypo=hypo,
                norm_key=norm_key,
                acc_model=acc_model,
                id=id,
                id_str=id_str,
                wrapper=wrapper,
                method=method,
                extra=extra,
                start=start,
                end=end,
                partition_by_region=(
                    (start != 0) or (end != dataset.summits.shape[0])
                ),  # If it's not all regions, it's partitioned
                decay=decay,
                regions=regions[random_ids],  # make sure it matches the random_ids
                verbose=False if len(ids) > 1 else verbose,
                write_bigwig=write_bigwig,
                chrom_size=genome.chrom_sizes,
                save_dir=save_dir,
            )

        if save_norm:
            vs_collection = np.concatenate(vs_collection, axis=0)
            print("sampled signals for normalization", vs_collection.shape)
            vs = vs_collection.reshape((-1))
            # Only trying to figure out the normalization factor
            low, median, high = (
                np.quantile(vs, 0.05),
                np.quantile(vs, 0.5),
                np.quantile(vs, 0.95),
            )
            print("normalizing", low, median, high)
            np.save(
                os.path.join(save_dir, f"norm.{wrapper}.{method}{extra}.{decay}.npy"),
                np.array([low, median, high]),
            )
            gc.collect()
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()


def entrance():
    args = parser.parse_args()
    main(
        args.pt,
        args.models,
        args.genome,
        args.peaks,
        args.start,
        args.end,
        args.method,
        args.wrapper,
        args.nth_output,
        args.write_numpy,
        args.gpus,
        args.overwrite,
        args.decay,
        args.extra,
        args.model_norm,
        args.sample,
        args.silent,
        args.save_norm,
        args.save_key,
        args.save_names,
    )


if __name__ == "__main__":
    entrance()
