import argparse
import gc
import json
import os.path
import pickle
import shutil
import socket
from pathlib import Path

import pandas as pd
import transformers
import wandb
from ema_pytorch import EMA
from wandb.sdk.lib.runid import generate_id

import scprinter as scp
from scprinter.seq.dataloader import *
from scprinter.seq.Models import *
from scprinter.utils import loadDispModel

torch.backends.cudnn.benchmark = True


def construct_model_from_config(config):
    n_filters = config["n_filters"]
    n_layers = config["n_layers"]
    activation = config["activation"]
    head_kernel_size = config["head_kernel_size"]
    kernel_size = config["kernel_size"]
    if activation == "relu":
        activation = torch.nn.ReLU()
    elif activation == "gelu":
        activation = torch.nn.GELU()
    batch_norm = config["batch_norm"]
    dilation_base = config["dilation_base"]
    bottleneck_factor = config["bottleneck_factor"]
    bottleneck = int(n_filters * bottleneck_factor)
    rezero = config["rezero"]
    batch_norm_momentum = config["batch_norm_momentum"]
    groups = config["groups"]
    no_inception = config["no_inception"]
    n_inception_layers = config["n_inception_layers"]
    inception_layers_after = config["inception_layers_after"]
    if no_inception:
        n_inception_layers = 0

    inception_version = config["inception_version"]
    if inception_layers_after:
        inception_bool = [False] * (n_layers - n_inception_layers) + [True] * (n_inception_layers)
    else:
        inception_bool = [True] * n_inception_layers + [False] * (n_layers - n_inception_layers)

    acc_dna_cnn = DNA_CNN(
        n_filters=n_filters,
    )
    dilation_func = lambda x: 2 ** (x + dilation_base)
    acc_hidden = DilatedCNN(
        n_filters=n_filters,
        bottleneck=bottleneck,
        n_layers=n_layers,
        kernel_size=kernel_size,
        groups=groups,
        activation=activation,
        batch_norm=batch_norm,
        residual=True,
        rezero=rezero,
        dilation_func=dilation_func,
        batch_norm_momentum=batch_norm_momentum,
        inception=inception_bool,
        inception_version=inception_version,
    )

    acc_head = Footprints_head(
        n_filters,
        kernel_size=head_kernel_size,
        n_scales=99,
    )
    output_len = 800
    dna_len = output_len + acc_dna_cnn.conv.weight.shape[2] - 1
    for i in range(n_layers):
        dna_len = dna_len + 2 * (kernel_size // 2) * dilation_func(i)
    print("dna_len", dna_len)

    acc_model = seq2PRINT(
        dna_cnn_model=acc_dna_cnn,
        hidden_layer_model=acc_hidden,
        profile_cnn_model=acc_head,
        dna_len=dna_len,
        output_len=output_len,
    )
    return acc_model, dna_len, output_len


def entry(config=None, wandb_run_name="", enable_wandb=True, resume=False):
    # Initialize a new wandb run
    print("start a new run!!!")
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    disp_path = (
        scp.datasets.pretrained_dispersion_model
        if "dispersion_model" not in config
        else config["dispersion_model"]
    )
    print("using dispersion model", disp_path)
    dispmodel = loadDispModel(disp_path)
    dispmodel = dispModel(deepcopy(dispmodel)).cuda()

    data_dir = config["data_dir"]
    temp_dir = config["temp_dir"]
    model_dir = config["model_dir"]

    split = config["split"]
    peaks = os.path.join(data_dir, config["peaks"])
    lora_mode = False

    if "cells" in config:
        cells_of_interest = np.array(config["cells"])
        pretrain_lora_model = os.path.join(data_dir, config["pretrain_lora_model"])
    else:
        cells_of_interest = None
        pretrain_lora_model = None

    if "pretrain_model" in config:
        pretrain_model = os.path.join(data_dir, config["pretrain_model"])
        insertion = os.path.join(data_dir, config["insertion"])
        grp2barcodes = os.path.join(data_dir, config["grp2barcodes"])
        grp2embeddings = os.path.join(data_dir, config["grp2embeddings"])
        grp2covs = os.path.join(data_dir, config["grp2covs"])
        cell_sample = config["cell_sample"]
        insertion = pickle.load(open(insertion, "rb"))
        grp2barcodes = np.array(pickle.load(open(grp2barcodes, "rb")), dtype="object")
        grp2embeddings = np.array(pickle.load(open(grp2embeddings, "rb")))
        grp2covs = np.array(pickle.load(open(grp2covs, "rb")))
        if cells_of_interest is not None:
            grp2barcodes = grp2barcodes[cells_of_interest]

        lora_mode = True
    else:
        pretrain_model = None

    peaks = pd.read_table(peaks, sep="\t", header=None)
    summits = peaks
    summits["summits"] = (summits[1] + summits[2]) // 2

    summits = summits[[0, "summits"]]
    summits["index"] = np.arange(len(summits))
    print(summits)
    modes = np.arange(2, 101, 1)

    max_jitter = config["max_jitter"]
    ema_flag = config["ema"]
    amp_flag = config["amp"]
    batch_size = config["batch_size"]
    weight_decay = config["weight_decay"]
    genome = config["genome"]
    if "accumulate_grad_batches" in config:
        accumulate_grad = config["accumulate_grad_batches"]
    else:
        accumulate_grad = 1

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

    lr = config["lr"]
    if lora_mode:
        acc_model = torch.load(pretrain_model, map_location="cpu", weights_only=False)
    else:
        acc_model, dna_len, output_len = construct_model_from_config(config)
        acc_model.dna_len = dna_len
        acc_model.signal_len = output_len

    acc_model.cuda()
    if lora_mode:
        for p in acc_model.parameters():
            p.requires_grad = False
    # acc_model.profile_cnn_model.conv_layer.weight.requires_grad = True
    # acc_model.profile_cnn_model.conv_layer.bias.requires_grad = True
    # acc_model.profile_cnn_model.linear.weight.requires_grad = True
    # acc_model.profile_cnn_model.linear.bias.requires_grad = True
    # acc_model, dna_len, output_len = construct_model_from_config(config)
    acc_model.cuda()

    total_params = sum(p.numel() for p in acc_model.parameters())
    print("total params - pretrained model", total_params)

    print("output len", acc_model.output_len)
    print("dna len", acc_model.dna_len)

    cov = {k: None for k in split}
    if not lora_mode:
        signals = os.path.join(data_dir, config["signals"])
        bias = str(genome.fetch_bias_bw(verify=True))
        signals = [signals, bias]
        datasets = {
            k: seq2PRINTDataset(
                signals=signals,
                ref_seq=genome.fetch_fa(),
                summits=summits[summits[0].isin(split[k])],
                DNA_window=dna_len,
                signal_window=output_len + 200,
                max_jitter=max_jitter if k in ["train"] else 0,
                min_counts=None,
                max_counts=None,
                cached=True,
                lazy_cache=True,
                reverse_compliment=(config["reverse_compliment"] if k in ["train"] else False),
                device="cpu",
            )
            for k in split
        }

        coverage = datasets["train"].coverage

        min_, max_ = np.quantile(coverage, 0.0001), np.quantile(coverage, 0.9999)
        min_ = max(min_, 10)
        print("coverage cutoff", min_, max_)

        for k in split:
            datasets[k].filter_by_coverage(min_, max_)
        dataloader = {
            k: seq2PRINTDataLoader(
                dataset=datasets[k],
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
            )
            for k in split
        }
    else:

        datasets = {
            k: scseq2PRINTDataset(
                insertion_dict=insertion,
                bias=str(genome.fetch_bias_bw(verify=True)),
                group2idx=grp2barcodes,
                ref_seq=genome.fetch_fa(),
                summits=summits[summits[0].isin(split[k])],
                DNA_window=acc_model.dna_len,
                signal_window=acc_model.output_len + 200,
                max_jitter=max_jitter if k in ["train"] else 0,
                cached=True,
                lazy_cache=True,
                reverse_compliment=(config["reverse_compliment"] if k in ["train"] else False),
                device="cpu",
                coverages=cov[k],
                data_augmentation=True if k in ["train"] else False,
                mode=config["dataloader_mode"] if k in ["train"] else "uniform",
                cell_sample=cell_sample,
            )
            for k in split
        }
        data = datasets["train"]
        coverage_in_peak = np.log10(
            data.coverages.reshape((len(data.summits), -1)).sum(axis=0, keepdims=True).T
        )
        print("coverage_in_peak", coverage_in_peak.shape)
        dataloader = {
            k: seq2PRINTDataLoader(
                dataset=datasets[k],
                batch_size=(
                    (batch_size // cell_sample) * 2
                    if (config["dataloader_mode"] == "peak") and (k in ["train"])
                    else batch_size
                ),
                num_workers=4,
                pin_memory=True,
                shuffle=True,
                collate_fn=(
                    collate_fn_singlecell
                    if (config["dataloader_mode"] == "peak") and (k in ["train"])
                    else None
                ),
            )
            for k in split
        }

    val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
        acc_model,
        dataloader["valid"],
        (len(datasets["valid"].summits) // batch_size),
        dispmodel,
        modes,
    )
    print("before training", val_loss, profile_pearson, across_pearson_fp, across_pearson_cov)
    # optimizer = torch.optim.AdamW(acc_model.parameters(), lr=1e-3, weight_decay=weight_decay)
    # if "scheduler" in config:
    #     scheduler = config["scheduler"]
    # else:
    #     scheduler = False
    # if scheduler:
    #     scheduler = transformers.get_cosine_schedule_with_warmup(
    #         optimizer, num_warmup_steps=3000, num_training_steps=100000
    #     )
    # else:
    #     scheduler = None
    #
    # acc_model.cuda()
    # ema = None
    # if ema_flag:
    #     update_after_step = 100
    #     print("update after step", update_after_step)
    #     ema = EMA(
    #         acc_model,
    #         beta=0.9999,  # exponential moving average factor
    #         update_after_step=update_after_step,  # only after this number of .update() calls will it start updating
    #         update_every=10,
    #     )  # how often to actually update, to save on compute (updates every 10th .update() call)
    #
    # acc_model.fit(
    #     dispmodel,
    #     dataloader["train"],
    #     validation_data=dataloader["valid"],
    #     validation_size=(len(datasets["valid"].summits) // batch_size),
    #     max_epochs=3,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     validation_freq=(len(datasets["train"].summits) // batch_size),
    #     early_stopping=(3 if "early_stopping" not in config else config["early_stopping"]),
    #     return_best=True,
    #     savename=f"{temp_dir}/{wandb_run_name}",
    #     modes=modes,
    #     downsample=None if "downsample" not in config else config["downsample"],
    #     ema=ema,
    #     use_amp=amp_flag,
    #     use_wandb=enable_wandb,
    #     batch_size=batch_size,
    # )
    #
    # if ema:
    #     del acc_model
    #     acc_model = torch.load(f"{temp_dir}/{wandb_run_name}.ema_model.pt").cuda()
    #
    # val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
    #     acc_model,
    #     dataloader["valid"],
    #     (len(datasets["valid"].summits) // batch_size),
    #     dispmodel,
    #     modes,
    # )
    #
    # print("after adjusted output head", val_loss, profile_pearson, across_pearson_fp, across_pearson_cov)
    #
    # for p in acc_model.parameters():
    #     p.requires_grad = False
    if pretrain_model is not None:
        acc_model = acc_model.cpu()
        if grp2covs.shape[0] != len(coverage_in_peak):
            print("grp2covs", grp2covs.shape, "coverage_in_peak", coverage_in_peak.shape)
            # slicing cells of interest so mismatch in shape:
            grp2covs = np.concatenate([grp2covs, np.ones_like(grp2covs)], axis=1)
            if cells_of_interest is not None:
                from scipy.stats import pearsonr

                print(
                    "pearsonr",
                    pearsonr(
                        grp2covs[cells_of_interest, 0].reshape((-1)), coverage_in_peak.reshape((-1))
                    ),
                )

                grp2covs[cells_of_interest, 1] = coverage_in_peak.reshape((-1))

        else:
            from scipy.stats import pearsonr

            print("pearsonr", pearsonr(grp2covs.reshape((-1)), coverage_in_peak.reshape((-1))))
            grp2covs = np.concatenate([grp2covs, coverage_in_peak], axis=1)

        print("grp2covs", grp2covs.shape)
        acc_model = seq2PRINT(
            dna_cnn_model=acc_model.dna_cnn_model,
            hidden_layer_model=acc_model.hidden_layer_model,
            profile_cnn_model=acc_model.profile_cnn_model,
            dna_len=acc_model.dna_len,
            output_len=acc_model.output_len,
            embeddings=grp2embeddings,
            coverages=grp2covs,
            rank=config["lora_rank"],
            hidden_dim=config["lora_hidden_dim"],
            lora_dna_cnn=config["lora_dna_cnn"],
            lora_dilated_cnn=config["lora_dilated_cnn"],
            lora_pff_cnn=config["lora_pff_cnn"],
            lora_profile_cnn=config["lora_profile_cnn"],
            lora_count_cnn=config["lora_count_cnn"],
            n_lora_layers=config["n_lora_layers"],
            coverage_in_lora=config["coverage_in_lora"],
        )
        acc_model = acc_model.cuda()

    if pretrain_lora_model is not None:
        pretrain_lora_model = torch.load(
            pretrain_lora_model, map_location="cpu", weights_only=False
        )
        acc_model.load_state_dict(pretrain_lora_model.state_dict())
        if cells_of_interest is not None:
            acc_model.embeddings = nn.Embedding.from_pretrained(
                torch.tensor(grp2embeddings[cells_of_interest]).float()
            )
            acc_model.coverages = nn.Embedding.from_pretrained(
                torch.tensor(grp2covs[cells_of_interest]).float()
            )
        acc_model = acc_model.cuda()

        # pretrain_lora_model = torch.load(pretrain_lora_model, map_location="cpu")
        # if isinstance(acc_model.profile_cnn_model, Footprints_head):
        #     acc_model.profile_cnn_model.conv_layer.layer.load_state_dict(
        #         pretrain_lora_model.profile_cnn_model.conv_layer.layer.state_dict()
        #     )
        #     acc_model.profile_cnn_model.linear.layer.load_state_dict(
        #         pretrain_lora_model.profile_cnn_model.linear.layer.state_dict()
        #     )
        # elif isinstance(acc_model.profile_cnn_model, BiasAdjustedFootprintsHead):
        #     acc_model.profile_cnn_model.adjustment_footprint.load_state_dict(
        #         pretrain_lora_model.profile_cnn_model.adjustment_footprint.state_dict()
        #     )
        #     acc_model.profile_cnn_model.adjustment_count.load_state_dict(
        #         pretrain_lora_model.profile_cnn_model.adjustment_count.state_dict()
        #     )
        #
        #     acc_model.profile_cnn_model.footprints_head.conv_layer.layer.load_state_dict(
        #         pretrain_lora_model.profile_cnn_model.footprints_head.conv_layer.layer.state_dict()
        #     )
        #     acc_model.profile_cnn_model.footprints_head.linear.layer.load_state_dict(
        #         pretrain_lora_model.profile_cnn_model.footprints_head.linear.layer.state_dict()
        #     )
        # else:
        #     print("profile cnn model not supported", type(acc_model.profile_cnn_model))
        # #
        # # acc_model.profile_cnn_model.adjustment_footprint.load_state_dict(
        # #     pretrain_lora_model.profile_cnn_model.adjustment_footprint.state_dict()
        # # )
        # # acc_model.profile_cnn_model.adjustment_count.load_state_dict(
        # #     pretrain_lora_model.profile_cnn_model.adjustment_count.state_dict()
        # # )
        # #
        # # acc_model.profile_cnn_model.footprints_head.conv_layer.layer.load_state_dict(
        # #     pretrain_lora_model.profile_cnn_model.footprints_head.conv_layer.layer.state_dict()
        # # )
        # # acc_model.profile_cnn_model.footprints_head.linear.layer.load_state_dict(
        # #     pretrain_lora_model.profile_cnn_model.footprints_head.linear.layer.state_dict()
        # # )
        #
        # for flag, module, pretrain_module in [
        #     (
        #         config["lora_dna_cnn"],
        #         [acc_model.dna_cnn_model.conv],
        #         [pretrain_lora_model.dna_cnn_model.conv],
        #     ),
        #     (
        #         config["lora_dilated_cnn"],
        #         [l.module.conv1 for l in acc_model.hidden_layer_model.layers],
        #         [l.module.conv1 for l in pretrain_lora_model.hidden_layer_model.layers],
        #     ),
        #     (
        #         config["lora_pff_cnn"],
        #         [l.module.conv2 for l in acc_model.hidden_layer_model.layers],
        #         [l.module.conv2 for l in pretrain_lora_model.hidden_layer_model.layers],
        #     ),
        #     (
        #         config["lora_profile_cnn"],
        #         [
        #             (
        #                 acc_model.profile_cnn_model.footprints_head.conv_layer
        #                 if isinstance(acc_model.profile_cnn_model, BiasAdjustedFootprintsHead)
        #                 else acc_model.profile_cnn_model.conv_layer
        #             )
        #         ],
        #         [
        #             (
        #                 pretrain_lora_model.profile_cnn_model.footprints_head.conv_layer
        #                 if isinstance(
        #                     pretrain_lora_model.profile_cnn_model, BiasAdjustedFootprintsHead
        #                 )
        #                 else pretrain_lora_model.profile_cnn_model.conv_layer
        #             )
        #         ],
        #     ),
        #     (
        #         config["lora_count_cnn"],
        #         [
        #             (
        #                 acc_model.profile_cnn_model.footprints_head.linear
        #                 if isinstance(acc_model.profile_cnn_model, BiasAdjustedFootprintsHead)
        #                 else acc_model.profile_cnn_model.linear
        #             )
        #         ],
        #         [
        #             (
        #                 pretrain_lora_model.profile_cnn_model.footprints_head.linear
        #                 if isinstance(
        #                     pretrain_lora_model.profile_cnn_model, BiasAdjustedFootprintsHead
        #                 )
        #                 else pretrain_lora_model.profile_cnn_model.linear
        #             )
        #         ],
        #     ),
        # ]:
        #     if flag:
        #         for m, prem in zip(module, pretrain_module):
        #             for i in range(len(m.A_embedding)):
        #                 if not isinstance(m.A_embedding[i], nn.Embedding):
        #                     m.A_embedding[i].load_state_dict(prem.A_embedding[i].state_dict())
        #             for i in range(len(m.B_embedding)):
        #                 if not isinstance(m.B_embedding[i], nn.Embedding):
        #                     m.B_embedding[i].load_state_dict(prem.B_embedding[i].state_dict())
        #
        # acc_model.cuda()

    print("model")
    print(acc_model)
    total_params = sum(p.numel() for p in acc_model.parameters() if p.requires_grad)
    print("total trainable params", total_params)
    total_params = sum(p.numel() for p in acc_model.parameters())
    print("total params", total_params)

    ema = None
    if ema_flag:
        update_after_step = 100
        print("update after step", update_after_step)
        ema = EMA(
            acc_model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=update_after_step,  # only after this number of .update() calls will it start updating
            update_every=10,
        )  # how often to actually update, to save on compute (updates every 10th .update() call)

    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(acc_model.parameters(), lr=lr, weight_decay=weight_decay)
    if "scheduler" in config:
        scheduler = config["scheduler"]
    else:
        scheduler = False
    if scheduler:
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=3000, num_training_steps=100000
        )
    else:
        scheduler = None

    val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
        acc_model,
        dataloader["valid"],
        (len(datasets["valid"].summits) // batch_size),
        dispmodel,
        modes,
    )
    print("before training", val_loss, profile_pearson, across_pearson_fp, across_pearson_cov)
    # raise EOFError
    acc_model.train()
    acc_model.fit(
        dispmodel,
        dataloader["train"],
        validation_data=dataloader["valid"],
        validation_size=(len(datasets["valid"].summits) // batch_size),
        max_epochs=300 if "epochs" not in config else config["epochs"],
        optimizer=optimizer,
        scheduler=scheduler,
        validation_freq=(len(datasets["train"].summits) // batch_size),
        early_stopping=(5 if "early_stopping" not in config else config["early_stopping"]),
        return_best=True,
        savename=f"{temp_dir}/{wandb_run_name}",
        modes=modes,
        downsample=None if "downsample" not in config else config["downsample"],
        ema=ema,
        use_amp=amp_flag,
        use_wandb=enable_wandb,
        accumulate_grad=accumulate_grad,
        batch_size=batch_size,
        coverage_warming=10 if lora_mode else 0,
    )
    if ema:
        del acc_model
        acc_model = torch.load(
            f"{temp_dir}/{wandb_run_name}.ema_model.pt", map_location="cpu", weights_only=False
        ).cuda()
    acc_model.eval()
    savename = config["savename"]
    acc_model.count_norm = [0, 0, 1]
    acc_model.foot_norm = [0, 0, 1]

    torch.save(acc_model, f"{model_dir}/{savename}-{wandb_run_name}.pt")
    val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = validation_step_footprint(
        acc_model,
        dataloader["valid"],
        (len(datasets["valid"].summits) // batch_size),
        dispmodel,
        modes,
    )

    test_loss, test_profile_pearson, test_across_pearson_fp, test_across_pearson_cov = (
        validation_step_footprint(
            acc_model,
            dataloader["test"],
            (len(datasets["valid"].summits) // batch_size),
            dispmodel,
            modes,
        )
    )
    acc_model.cpu()
    del acc_model
    if ema:
        del ema
    torch.cuda.empty_cache()
    gc.collect()

    # Now that the model is trained we use it to estimate the normalization range of the shap values
    # For LoRA model the tricky part is that we also need to sample across multiple cells.
    model_type = "lora" if lora_mode else "seq2print"
    model = f"{model_dir}/{savename}-{wandb_run_name}.pt"
    gpus = [0]
    peak_file = os.path.join(data_dir, config["peaks"])
    if lora_mode:
        num_cell = len(grp2barcodes)
    else:
        num_cell = 1
    sample_num = 30000
    count_hypo = scp.tl.seq_attr_seq2print(
        region_path=peak_file,
        model_type=model_type,
        model_path=model,
        genome=genome,
        gpus=gpus,
        preset="count",
        sample=max(sample_num // num_cell, 2),
        save_norm=True,
        overwrite=True,
        verbose=True,
        launch=True,
        # lora_ids=[i for i in range(num_cell)],
        lora_config=config["config_path"],
        group_names=list(config["group_names"]) if "group_names" in config else None,
    )
    foot_hypo = scp.tl.seq_attr_seq2print(
        region_path=peak_file,
        model_type=model_type,
        model_path=model,
        genome=genome,
        gpus=gpus,
        preset="footprint",
        sample=max(sample_num // num_cell, 2),
        save_norm=True,
        overwrite=True,
        verbose=True,
        launch=True,
        # lora_ids=[i for i in range(num_cell)],
        lora_config=config["config_path"],
        group_names=list(config["group_names"]) if "group_names" in config else None,
    )
    acc_model = torch.load(
        f"{model_dir}/{savename}-{wandb_run_name}.pt", map_location="cpu", weights_only=False
    )
    count_hypo = np.load(count_hypo)
    foot_hypo = np.load(foot_hypo)
    low, median, high = count_hypo
    print("count head normalization factor", low, median, high)
    acc_model.count_norm = [low, median, high]
    low, median, high = foot_hypo
    print("footprint head normalization factor", low, median, high)
    acc_model.foot_norm = [low, median, high]
    torch.save(acc_model, f"{model_dir}/{savename}-{wandb_run_name}.pt")
    # shutil.rmtree(f"{model_dir}/{savename}-{wandb_run_name}.pt_deepshap")

    if enable_wandb:
        wandb.summary["final_valid_loss"] = val_loss
        wandb.summary["final_valid_within"] = profile_pearson
        wandb.summary["final_valid_across"] = across_pearson_fp
        wandb.summary["final_valid_cov"] = across_pearson_cov
        wandb.summary["final_test_loss"] = test_loss
        wandb.summary["final_test_within"] = test_profile_pearson
        wandb.summary["final_test_across"] = test_across_pearson_fp
        wandb.summary["final_test_cov"] = test_across_pearson_cov
        wandb.finish()
        del acc_model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="seq2PRINT (LoRA) model")
    parser.add_argument("--config", type=str, default="config.JSON", help="config file")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/",
        help="data directory, will be append to all data path in config",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="/",
        help="temp directory, will be used to store working models",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/", help="will be used to store final models"
    )
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb")
    parser.add_argument("--project", type=str, default="scPrinterSeq_v3_lora", help="project name")
    parser.add_argument("--run_id", type=str, default=None, help="run id")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")

    torch.set_num_threads(4)
    args = parser.parse_args()
    config = json.load(open(args.config))

    for path in [args.data_dir, args.temp_dir, args.model_dir]:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    config["data_dir"] = args.data_dir
    config["temp_dir"] = args.temp_dir
    config["model_dir"] = args.model_dir
    config["config_path"] = args.config

    if args.enable_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project,
            notes=socket.gethostname() if "notes" not in config else config["notes"],
            # track hyperparameters and run metadata
            tags=config["tags"] if "tags" in config else [],
            config=config,
            job_type="training",
            reinit=True,
            id=args.run_id,
            resume="must" if args.resume else None,
        )

    # Initialize a new wandb run
    if args.enable_wandb:
        with wandb.init(config=config):
            config = wandb.config
            print("run id", wandb.run.id)
            print("run name", wandb.run.name)
            wandb_run_name = wandb.run.name
            entry(config, wandb_run_name, args.enable_wandb, args.resume)
    else:
        config = config
        random_name = generate_id() if args.run_id is None else args.run_id
        print(random_name)  # Example: "bold-otter-17"
        wandb_run_name = random_name
        entry(config, wandb_run_name, args.enable_wandb, args.resume)


if __name__ == "__main__":
    main()
