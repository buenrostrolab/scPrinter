import argparse
import gc
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm, trange

import scprinter as scp
from scprinter.utils import DNA_one_hot, regionparser
from scprinter.utils import zscore2pval_torch as z2p

parser = argparse.ArgumentParser(description="Calculate delta effects of motifs on models")

parser.add_argument("--pt", nargs="+", type=str, default="config.JSON", help="model.pt")
parser.add_argument("--ids", type=str, default=None, help="lora ids")
parser.add_argument("--genome", type=str, default="hg38", help="genome")
parser.add_argument("--peaks", type=str, default=None, help="peaks")
parser.add_argument("--motifs", type=str, default=None, help="motifs")
parser.add_argument(
    "--mode",
    type=str,
    default="argmax",
    help="mode to fetch concensus from motif argmax or multinomial",
)
parser.add_argument("--gpus", nargs="+", type=int, help="gpus")
parser.add_argument("--sample_num", type=int, default=1000, help="sample this many peak")
parser.add_argument("--output", type=str, default="delta_effects", help="output file")
parser.add_argument("--collapse_bins", default=False, action="store_true", help="collapse bins")
parser.add_argument("--random_seq", default=False, action="store_true", help="random seq")
parser.add_argument("--verbose", default=False, action="store_true", help="verbose")

nucleotide = np.array(list("ACGT"))


# Function to get model outputs in batches and concatenate them separately
def get_model_outputs(
    models, seq_list, batch_size=64, device="cuda", verbose=False, summary_func_footprint=None
):
    all_outputs1 = []
    all_outputs2 = []
    # print (summary_func_footprint)
    for seq in tqdm(seq_list, disable=not verbose):
        seq = seq  # Move input to GPU
        # n_motif = seq.shape[0]
        n_CRE = seq.shape[0]
        # seq = seq.flatten(start_dim=0, end_dim=1)
        # print("seq prepared v2")

        outputs1 = 0
        outputs2 = 0
        # bar = tqdm(seq, desc="get_model_outputs")
        with torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=torch.bfloat16,
            enabled=device != "cpu",
        ):
            with torch.no_grad():  # Disable gradient calculation
                for i in range(0, len(seq), batch_size):
                    batch = seq[i : i + batch_size].to(device).float()
                    output1, output2 = 0, 0
                    for model in models:
                        a, b = model(batch)
                        output1 += a
                        output2 += b
                    output1 = z2p(output1).detach().cpu()
                    output2 = output2.float().detach().cpu()
                    if summary_func_footprint is not None:
                        # print (output1.shape)
                        output1 = summary_func_footprint(output1)
                        # print(output1.shape)
                    # motif_idx = torch.arange(i, i+batch.shape[0]) // n_CRE
                    # if outputs1 is None:
                    #     # shape of n_motif + output1.shape
                    #     outputs1 = torch.zeros([n_motif] + list(output1.shape)[1:])
                    #     outputs2 = torch.zeros([n_motif] + list(output2.shape)[1:])
                    #     print (outputs1.shape, outputs2.shape)
                    # # scatter output1 / output2 to the right place
                    # outputs1.index_add_(0, motif_idx, output1)
                    # outputs2.index_add_(0, motif_idx, output2)
                    outputs1 += output1.sum(axis=0) / n_CRE
                    outputs2 += output2.sum(axis=0) / n_CRE

            all_outputs1.append(outputs1)
            all_outputs2.append(outputs2)
    return all_outputs1, all_outputs2


def fast_sample(motif):
    nucleotide = np.array(list("ACGT"))
    rg = np.random.default_rng()
    cumulative_probs = np.cumsum(motif, axis=0)
    random_values = rg.random((1, cumulative_probs.shape[1]))
    id = np.argmax(cumulative_probs > random_values, axis=0)
    return nucleotide[id]


def add_motif_to_seq(seq_one_hot, motif, args):
    seq_copy = deepcopy(seq_one_hot)
    pad = (seq_one_hot.shape[-1] - motif.shape[-1]) // 2
    seq_to_change = motif + seq_copy[:, pad : pad + motif.shape[1]] * 1e-3  # add motif,
    if args.mode == "argmax":
        concensus = nucleotide[np.argmax(seq_to_change, axis=0)]
    elif args.mode == "multinomial":
        concensus_motif = seq_to_change / seq_to_change.sum(axis=0, keepdims=True)
        concensus = fast_sample(concensus_motif)
    seq_copy[:, pad : pad + motif.shape[1]] = DNA_one_hot(concensus).detach().numpy()
    return seq_copy


def plant_motifs_in_seq(seq, motifs, args):
    # rg = np.random.default_rng()
    sequences = []
    seq_one_hot = DNA_one_hot(seq).float().detach().numpy()  # make N some small values
    for motif_idx, motif in enumerate(motifs):
        # for m in motif:
        sequences.append(add_motif_to_seq(seq_one_hot, motif, args))
        # if motif.shape[0] > 1:
        #     composite = motif.sum(axis=0)
        #     composite = composite / composite.sum(axis=0, keepdims=True)
        #     composite = np.nan_to_num(composite)
        #     sequences.append(add_motif_to_seq(seq_one_hot, composite, args))
        # sequences.append(seq_one_hot)

    sequences = np.stack(sequences, axis=0).astype("int")  # nmotif, xxx
    sequences = torch.tensor(sequences)  #
    return sequences


def main():
    torch.set_num_threads(4)
    args = parser.parse_args()
    gpus = args.gpus
    print("gpus", gpus)
    if len(gpus) == 1:
        print("setting device", int(gpus[0]))
        torch.cuda.set_device(int(gpus[0]))
    CREs = regionparser(args.peaks, None)
    if len(CREs) > args.sample_num:
        CREs = CREs.sample(args.sample_num).reset_index(drop=True)
    models = args.pt
    if args.ids is not None:
        ids = args.ids
        ids = ids.split(",")
        ids = [i.split("-") for i in ids]
        print(ids[:5], ids[-5:])
        ids = [[int(j) for j in i] for i in ids]
    else:
        ids = [[None]]
    genome = args.genome
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

    parallel_lora = len(ids) > len(models)
    gpus = gpus[: max(len(models), len(ids))]
    if len(gpus) > 1:
        # parallelize
        CREs.to_csv(f"{args.output}_sampled_CREs.txt", sep="\t", header=False, index=False)
        if parallel_lora:
            n_split = len(gpus)
            ids_batch = np.array_split(ids, n_split)
            commands = [
                ["seq2print_delta", "--pt"]
                + models
                + [
                    "--ids",
                    ",".join(["-".join([str(yy) for yy in xx]) for xx in ids_batch[i]]),
                    "--genome",
                    args.genome,
                    "--peaks",
                    f"{args.output}_sampled_CREs.txt",
                    "--motifs",
                    args.motifs,
                    "--mode",
                    args.mode,
                    "--gpus",
                    str(gpu),
                    "--sample_num",
                    str(args.sample_num),
                    "--output",
                    args.output + f"_part{i}",
                ]
                + (["--collapse_bins"] if args.collapse_bins else [])
                + (["--random_seq"] if args.random_seq else [])
                + (["--verbose"] if args.verbose else [])
                for i, gpu in enumerate(gpus)
            ]
        else:
            model_batch = np.array_split(models, len(gpus))
            commands = [
                ["seq2print_delta", "--pt"]
                + list(model)
                + (["ids", args.ids] if args.ids is not None else [])
                + [
                    "--genome",
                    args.genome,
                    "--peaks",
                    f"{args.output}_sampled_CREs.txt",
                    "--motifs",
                    args.motifs,
                    "--mode",
                    args.mode,
                    "--gpus",
                    str(gpu),
                    "--sample_num",
                    str(args.sample_num),
                    "--output",
                    args.output + f"_part{i}",
                ]
                + (["--collapse_bins"] if args.collapse_bins else [])
                + (["--random_seq"] if args.random_seq else [])
                + (["--verbose"] if args.verbose else [])
                for i, (model, gpu) in enumerate(zip(model_batch, gpus))
            ]

        pool = ProcessPoolExecutor(max_workers=len(gpus))
        for gpu, command in zip(gpus, commands):
            # print (command)
            # os.system(" ".join(command))
            pool.submit(subprocess.run, command)
        pool.shutdown(wait=True)

        # merge
        results_fp = []
        results_count = []
        for i in range(len(gpus)):
            result = pickle.load(open(args.output + f"_part{i}", "rb"))
            results_fp.append(result[0])
            results_count.append(result[1])
        if parallel_lora:
            results_fp = np.concatenate(results_fp, axis=1)
            results_count = np.concatenate(results_count, axis=1)
        else:
            results_fp = np.concatenate(results_fp, axis=0)
            results_count = np.concatenate(results_count, axis=0)
        pickle.dump([results_fp, results_count], open(args.output, "wb"))
        for i in range(len(gpus)):
            subprocess.run(["rm", args.output + f"_part{i}"])
    else:
        torch.cuda.set_device(int(gpus[0]))

        motifs = np.load(args.motifs, allow_pickle=True)
        results_fp_all = []
        results_count_all = []

        for model_id, model in enumerate(models):
            if isinstance(model, str):
                model = torch.load(model, map_location="cpu", weights_only=False)

            if model.coverages is not None:
                # print("setting coverage to be the same")
                mm = model.coverages.weight.data.mean(dim=0)
                model.coverages.weight.data = (
                    torch.ones_like(model.coverages.weight.data) * mm[None]
                )
            dna_len = model.dna_len
            results_fp = []
            results_count = []

            seq_motif = torch.zeros((len(motifs), len(CREs), 4, dna_len), dtype=torch.int8)
            seq_bg = torch.zeros((len(CREs), 4, dna_len), dtype=torch.int8)
            # bar = trange(args.sample_num, desc="sampling")
            ct = 0
            bar = trange(len(CREs), desc="sampling")
            for i, region in CREs.iterrows():
                bar.update(1)
                chr, start, end = region.iloc[0], region.iloc[1], region.iloc[2]

                center = int((start + end) // 2)
                pad = dna_len // 2
                seq = genome.fetch_seq(chr, center - pad, center + pad)
                if len(seq) != dna_len:
                    print("seq length not dna_len", len(seq))
                    continue
                # random shuffle seq
                if args.random_seq:
                    seq = list(seq)
                    np.random.shuffle(seq)
                    seq = "".join(list(seq))
                if "N" in seq:
                    continue

                # bar.update(1)
                sequences = plant_motifs_in_seq(seq, motifs, args)
                seq_motif[:, i] = sequences.type(torch.int8)
                # if ct == 0:
                #     nucleotide = np.array(list("ACGT"))
                # for j in range(len(tmp1)):
                #     # print (''.join(list(nucleotide[np.argmax(tmp1[j].numpy(), axis=0)])))
                #     print (''.join(list(nucleotide[np.argmax(tmp2[j].numpy(), axis=0)])))
                # raise EOFError
                ct += 1
                seq_bg[i] = DNA_one_hot(seq).detach().type(torch.int8)
                # seq_motif.append(sequences)
                # seq_bg.append(DNA_one_hot(seq).detach()[None]) # to match the #motif
                # break

            input_sequences = [x for x in seq_motif] + [seq_bg]
            import time

            # print("seq prepared")
            # time.sleep(100)
            #
            # for seq in [seq_motif, seq_bg]:
            #     seq = torch.stack(seq, axis=1) # nmotif, nCRE, xxx
            #     # seq = seq.flatten(start_dim=0, end_dim=1) # collapse first two dim
            #     input_sequences.append(seq)
            bar = tqdm(ids, desc="lora model", disable=not args.verbose)
            for name in bar:
                bar.set_description(f"working on {name} with model {model_id}")
                # Get model outputs for all sequences
                if name is not None:
                    md = model.collapse(name).to("cuda")
                else:
                    md = model.to("cuda")
                # print (md)
                # print (args.collapse_bins)
                outputs1, outputs2 = get_model_outputs(
                    [md],
                    input_sequences,
                    64,
                    "cuda",
                    verbose=((len(ids) == 1) and (args.verbose)),
                    summary_func_footprint=(
                        partial(torch.sum, dim=-1) if args.collapse_bins else None
                    ),
                )
                # Now, concatenated_outputs1 and concatenated_outputs2 hold the concatenated results of the first and second outputs of the model
                # delta = z2p(outputs1[1]) - z2p(outputs1[0])
                # motif minus baseline
                # nmotif, nCRE, output_dim
                outputs1 = torch.stack(outputs1[:-1], axis=0) - outputs1[-1][None]
                outputs2 = torch.stack(outputs2[:-1], axis=0) - outputs2[-1][None]

                # delta = torch.stack(outputs1, axis=0)
                results_fp.append(outputs1.detach().cpu().numpy())

                # delta = torch.stack(outputs2, axis=0)
                # delta = outputs2[0] - outputs2[1] - outputs2[2] + outputs2[3]
                results_count.append(outputs2.detach().cpu().numpy())
                del md
            # results_fp = np.array(results_fp).mean(axis=2) # average over CREs
            # results_count = np.array(results_count).mean(axis=2) # average over CREs
            # print(results_count.shape)
            results_fp_all.append(results_fp)
            results_count_all.append(results_count)
        pickle.dump(
            [np.array(results_fp_all), np.array(results_count_all)], open(args.output, "wb")
        )


if __name__ == "__main__":
    main()
