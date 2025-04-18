import argparse
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import scprinter as scp
from scprinter.seq.dataloader import *
from scprinter.seq.interpretation.attributions import *
from scprinter.seq.Models import *


@torch.no_grad()
def forward_pass_model(feats, model):
    feats = torch.as_tensor(feats).float().cuda()[:, None, :]
    feats_rev = torch.flip(feats, [2])
    pred = torch.stack(
        [
            torch.sigmoid(model(feats)).cpu().detach()[:, 0, :],
            torch.flip(torch.sigmoid(model(feats_rev)).cpu().detach()[:, 0, :], [1]),
        ],
        axis=0,
    )
    pred = torch.mean(pred, 0)
    return pred


def bigwig_reader(bws, summits, signal_window, batch_size, post_normalize):
    # bws are list of lists:
    feats = []
    bar = trange(len(bws) * len(summits))
    for bw in bws:
        bw = [pyBigWig.open(i, "r") for i in bw]

        for i in range(len(summits)):
            bar.update(1)
            chrom, summit = summits.iloc[i]
            feat = np.nanmean(
                [
                    np.nan_to_num(
                        j.values(chrom, summit - signal_window, summit + signal_window, numpy=True)
                    )
                    for j in bw
                ],
                axis=0,
            )
            feats.append(feat)
            if not post_normalize:
                if len(feats) >= batch_size:
                    yield np.array(feats)
                    feats = []
        if post_normalize:
            feats = np.array(feats)
            pad = (feats.shape[-1] - 800) // 2
            if pad > 0:
                vs = feats[..., pad:-pad]
                low, median, high = (
                    np.quantile(vs, 0.05),
                    np.quantile(vs, 0.5),
                    np.quantile(vs, 0.95),
                )
                feats = (feats - median) / (high - low)
            for chunk in range(0, len(feats), batch_size):
                yield feats[chunk : chunk + batch_size]
            feats = []
        else:
            if len(feats) > 0:
                yield np.array(feats)
                feats = []


def numpy_reader(paths, summits, signal_window, batch_size, post_normalize):
    feats = []
    lengths = 0

    bar = trange(len(paths) * len(summits))
    for path in paths:
        feat = np.nanmean([np.load(p)["arr_0"] for p in path], axis=0)
        pad = feat.shape[-1] // 2 - signal_window
        feat = feat[..., pad:-pad]
        if post_normalize:
            pad = (feat.shape[-1] - 800) // 2
            if pad > 0:
                vs = feat[..., pad:-pad]
                low, median, high = (
                    np.quantile(vs, 0.05),
                    np.quantile(vs, 0.5),
                    np.quantile(vs, 0.95),
                )
            feat = (feat - median) / (high - low)
        feats.append(feat)
        lengths += feat.shape[0]
        if lengths >= batch_size:
            feats = np.concatenate(feats)
            bar.update(len(feats))
            for chunk in range(0, len(feats), batch_size):
                yield feats[chunk : chunk + batch_size]
            feats = []
            lengths = 0
    if len(feats) > 0:
        yield np.concatenate(feats)


def main():
    parser = argparse.ArgumentParser(description="seq2PRINT generate TFBS ")
    parser.add_argument("--count_pt", type=str, default=None, help="model.pt")
    parser.add_argument("--foot_pt", type=str, default=None, help="model.pt")
    parser.add_argument("--seq_count", type=str, nargs="+", help="seq_count", default=None)
    parser.add_argument("--seq_foot", type=str, nargs="+", help="seq_foot", default=None)
    parser.add_argument("--genome", type=str, default="hg38", help="genome")
    parser.add_argument("--peaks", type=str, default=None, help="peaks")
    parser.add_argument("--save_name", type=str, default="/", help="save_dir")
    parser.add_argument("--collection_name", type=str, default=None, help="collection_name")
    parser.add_argument("--gpus", nargs="+", type=int, help="gpus")
    parser.add_argument("--temp", action="store_true", default=False, help="temp")
    parser.add_argument(
        "--avg", action="store_true", default=False, help="just avg without finetune model"
    )
    parser.add_argument("--silent", action="store_true", default=False, help="silent")
    parser.add_argument("--per-peak", action="store_true", default=False, help="per-peak")
    parser.add_argument("--write_numpy", action="store_true", default=False, help="write numpy")
    parser.add_argument("--lora_ids", type=str, default=None, help="lora_ids")
    parser.add_argument("--read_numpy", action="store_true", default=False, help="read numpy")
    parser.add_argument(
        "--post_normalize",
        action="store_true",
        default=False,
        help="post_normalize seq attr per lora id",
    )

    torch.set_num_threads(4)
    args = parser.parse_args()
    collection_name = args.collection_name
    if collection_name is None:
        collection_name = args.save_name
    seq_count = args.seq_count
    seq_foot = args.seq_foot
    count_pt = args.count_pt
    foot_pt = args.foot_pt
    genome = args.genome
    peaks = args.peaks
    gpus = args.gpus
    save_name = args.save_name
    silent = args.silent
    print("gpu", int(gpus[0]))
    torch.cuda.set_device(int(gpus[0]))
    save_name = save_name.split(",")  # if there are multiple save names save them separately.

    if count_pt is not None:
        count_model = torch.jit.load(count_pt, map_location="cpu")
        count_model.with_motif = False
        count_model = count_model.to("cuda")
        count_model.eval()
    else:
        count_model = None
    if foot_pt is not None:
        foot_model = torch.jit.load(foot_pt, map_location="cpu")
        foot_model.with_motif = False
        foot_model = foot_model.to("cuda")
        foot_model.eval()
    else:
        foot_model = None

    if count_model is None and foot_model is None:
        raise ValueError("No model loaded")

    summits = pd.read_table(peaks, sep="\t", header=None)
    summits = summits.drop_duplicates([0, 1, 2])  # drop exact same loci
    summits["summits"] = (summits[1] + summits[2]) // 2
    summits = summits[[0, "summits"]]
    summits["summits"] = np.array(summits["summits"], dtype=int)

    signal_window = 1200 // 2
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
    summits = summits[summits[0].isin(genome.chrom_sizes)]

    # Four cases: with or without lora x read from numpy or bigwig
    if args.lora_ids is None:
        lora_ids = [""]
    else:
        lora_ids = args.lora_ids.split(",")
    if not args.write_numpy:
        assert len(save_name) == len(
            lora_ids
        ), "When writing a bigwig file, the number of savename must be the same as lora_ids"

    if len(gpus) > 1 and args.lora_ids is not None:
        # Split lora_ids to different gpus
        ids_batch = np.array_split(lora_ids, len(gpus))
        if len(save_name) < len(ids_batch):
            save_name = [save_name[0]] * len(
                ids_batch
            )  # this is fine, because we asserted above that the length has to be the same if writing bigwig
        save_name_batch = np.array_split(save_name, len(gpus))
        commands = [
            [
                "seq2print_tfbs",
                "--count_pt",
                args.count_pt,
                "--foot_pt",
                args.foot_pt,
                "--seq_count",
                " ".join(args.seq_count),
                "--seq_foot",
                " ".join(args.seq_foot),
                "--genome",
                args.genome,
                "--peaks",
                args.peaks,
                "--gpus",
                str(gpu),
                "--lora_ids",
                ",".join([str(i) for i in ids]),
            ]
            + (["--silent"] if args.silent else [])
            + (["--per-peak"] if args.per_peak else [])
            + (["--read_numpy"] if args.read_numpy else [])
            + (["--write_numpy"] if args.write_numpy else [])
            + (
                ["--collection_name", f"{collection_name}_temp_TFBS_part{i}"]
                if args.write_numpy
                else ["--save_name", ",".join(list(save_name_batch[i]))]
            )
            + (["--post_normalize"] if args.post_normalize else [])
            for i, (gpu, ids) in enumerate(zip(gpus, ids_batch))
        ]

        pool = ProcessPoolExecutor(max_workers=len(gpus))
        for gpu, command in zip(gpus, commands):
            print(command)
            pool.submit(subprocess.run, command)

        pool.shutdown(wait=True)
        # Only load and concat if in save numpy mode
        if args.write_numpy:
            results = []
            for i in range(len(gpus)):
                results.append(np.load(f"{collection_name}_temp_TFBS_part{i}_TFBS.npz")["tfbs"])
            avg = np.concatenate(results)
            np.savez(
                collection_name + "_TFBS.npz",
                tfbs=avg,
            )
            for i in range(len(gpus)):
                os.remove(collection_name + f"_temp_TFBS_part{i}_TFBS.npz")

    else:
        # only one gpu
        input_readers = []
        models = []
        reader = bigwig_reader if not args.read_numpy else numpy_reader

        if seq_foot is not None:
            input_readers.append(
                reader(
                    [
                        [sc.replace("{lora_id}", str(lora_id)) for sc in seq_count]
                        for lora_id in lora_ids
                    ],
                    summits,
                    signal_window,
                    512,
                    args.post_normalize,
                )
            )
            models.append(count_model)
        if seq_count is not None:
            input_readers.append(
                reader(
                    [
                        [sf.replace("{lora_id}", str(lora_id)) for sf in seq_foot]
                        for lora_id in lora_ids
                    ],
                    summits,
                    signal_window,
                    512,
                    args.post_normalize,
                )
            )
            models.append(foot_model)

        output_all = []
        for reader, model in zip(input_readers, models):
            output = []
            for feats in reader:
                output.append(forward_pass_model(feats, model))
            output = torch.cat(output, 0).numpy()
            output_all.append(output)
        # Get the average of all models
        avg = np.mean(output_all, axis=0)
        avg = avg.reshape((len(lora_ids), len(summits), -1))
        avg = avg[..., 100:-100]
        print(avg.shape)

        if args.write_numpy:
            np.savez(
                collection_name + "_TFBS.npz",
                tfbs=avg,
            )
        else:

            regions = summits.copy()
            regions[1] = regions["summits"] - signal_window + 200
            regions[2] = regions["summits"] + signal_window - 200
            regions = regions[[0, 1, 2]]
            for i, (lora_id, name) in enumerate(zip(lora_ids, save_name)):
                print(name, lora_id)
                attribution_to_bigwig(
                    avg[i],
                    regions,
                    genome.chrom_sizes,
                    res=1,
                    mode="average",
                    output=name + "_TFBS.bigwig",
                    verbose=not silent,
                )


if __name__ == "__main__":
    main()
