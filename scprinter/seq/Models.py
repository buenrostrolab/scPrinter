from copy import deepcopy

import torch
import wandb
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm, trange

from .minimum_footprint import *
from .Modules import *


def predict(model, X, batch_size=64, verbose=False):
    """
    This is the predict function

    Parameters
    ----------
    model: torch.nn.Module
        The model to predict with
    X: torch.Tensor
        The input data
    batch_size: int, optional
        batch size to use for the forward pass
    verbose: bool, optional
        Whether to display a progress bar. Default is False.

    """
    pred_footprints = []
    pred_scores = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in trange(0, len(X), batch_size, disable=not verbose):
            X_batch = X[i : i + batch_size].to(device)
            X_foot, X_score = model(X_batch)
            pred_footprints.append(X_foot.detach().cpu())
            pred_scores.append(X_score.detach().cpu())
        pred_footprints = torch.cat(pred_footprints, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)
    return pred_footprints, pred_scores


@torch.no_grad()
def validation_step_footprint(model, validation_data, validation_size, dispmodel, modes):
    """
    This is the validation step for the seq2PRINT model

    Parameters
    ----------
    model: torch.nn.Module
        The model to validate
    validation_data: torch.utils.data.DataLoader
        a dataloader for the validation data
    validation_size: int | None
        the number of batches to validate on
    dispmodel: torch.nn.Module
        the dispersion model to use for the footprint prediction
    modes: np.ndarray | None | torch.Tensor
        the modes to use for the footprint prediction
    Returns
    -------

    """
    device = next(model.parameters()).device
    size = 0
    val_loss = 0

    profile_pearson_counter = CumulativeCounter()
    across_batch_pearson_fp = CumulativePearson()
    across_batch_pearson_cov = CumulativePearson()

    if validation_size is None:
        validation_size = len(validation_data)

    bar = trange(validation_size, desc=" - (Validation)", leave=False, dynamic_ncols=True)
    total_len = 0
    for data in validation_data:
        if len(data) == 2:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            cell = None
            norm_cov = None
        else:
            X, y, cell, _, norm_cov = data
            cell = cell[..., 0]
            X = X.to(device)
            y = y.to(device)
            cell = cell.to(device)
            # norm_cov = norm_cov.to(device)
            norm_cov = None

        pred_footprint, pred_coverage = model(X, cell)
        y_coverage = (
            torch.log1p(y[:, 0].sum(dim=-1)) if norm_cov is None else norm_cov
        )  # calculating coverage on the fly
        # print (coverage.min(), coverage.max())
        y_footprint = multiscaleFoot(y[:, 0], y[:, 1], modes, dispmodel)
        mask = ~torch.isnan(y_footprint)
        y_footprint = torch.nan_to_num(y_footprint, nan=0)
        loss_ = F.mse_loss(pred_footprint[mask], y_footprint[mask])
        pred_footprint, y_footprint = pred_footprint.reshape(
            (len(pred_footprint), -1)
        ), y_footprint.reshape((len(y), -1))
        val_loss += loss_.item()

        # Now calculate the batch pearson correlation scores
        corr = batch_pearson_correlation(pred_footprint, y_footprint).detach().cpu()[:, None]
        profile_pearson_counter.update(corr)
        # save for across batch pearson
        across_batch_pearson_fp.update(pred_footprint, y_footprint)
        across_batch_pearson_cov.update(pred_coverage, y_coverage)

        size += 1
        bar.update(1)
        if validation_size is not None and size > validation_size:
            break

    val_loss /= size

    return (
        val_loss,
        profile_pearson_counter.mean(),
        across_batch_pearson_fp.corr(),
        across_batch_pearson_cov.corr(),
    )


def adjust_embedding_scale(rank, embedding, coverages, coverage_in_lora, A_embedding, B_embedding):
    # test A_output distribution
    with torch.no_grad():

        embedding.eval()
        A_embedding.eval()
        B_embedding.eval()
        test_cell_num = min(100, embedding.weight.shape[0])

        A_cells = embedding(torch.arange(test_cell_num).long())
        if coverages is not None:
            coverages_cells = coverages(torch.arange(test_cell_num).long())
        else:
            coverages_cells = None
        if coverage_in_lora:
            A_cells = torch.cat([A_cells, coverages_cells], dim=-1)
        A_output = A_embedding(A_cells)
        B_output = B_embedding(A_cells)
        mean, std = A_output.mean(), A_output.std()
        print("A_output mean: {}, std: {}".format(mean, std))
        mean, std = B_output.mean(), B_output.std()
        print("B_output mean: {}, std: {}".format(mean, std))

        A_output = A_output.reshape((test_cell_num, -1, rank))
        B_output = B_output.reshape((test_cell_num, rank, -1))
        lora_out = torch.bmm(A_output, B_output)
        mean, std = lora_out.mean(), lora_out.std()
        print("lora_out mean: {}, std: {}".format(mean, std))
        # self.scale *= 1 / (std * r)
        rescale_factor = math.sqrt(1 / (std))
        embedding.weight.data[...] *= rescale_factor
        if coverage_in_lora:
            coverages.weight.data[...] *= rescale_factor
        return rescale_factor  # rescale the embedding matrix


class seq2PRINT(nn.Module):
    """
    This is the seq2PRINT model

    Parameters
    ----------
    dna_cnn_model: torch.nn.Module
        The DNA CNN model
    hidden_layer_model: torch.nn.Module
        The hidden layer model
    profile_cnn_model: torch.nn.Module
        The profile CNN model
    dna_len: int
        The length of the input DNA sequence
    output_len: int
        The length of the output peak window
    embeddings: np.ndarray | None
        The embeddings to use for single cells or pseudobulks
    coverages: np.ndarray | None
        The bias terms to regress out when doing LoRA
    rank: int
        The rank to use for the LoRA model
    hidden_dim: int
        The hidden dimension to use for the LoRA model
    n_lora_layers: int
        The number of hidden layers from embedding -> lora params
    lora_dna_cnn: bool
        Whether to use LoRA on the DNA CNN model
    lora_dilated_cnn: bool
        Whether to use LoRA on the dilated CNN model (the dilated part)
    lora_pff_cnn: bool
        Whether to use LoRA on the dilated CNN model (the pff part)
    lora_profile_cnn: bool
        Whether to use LoRA on the output CNN model the footprint part
    lora_count_cnn: bool
        Whether to use LoRA on the output CNN model the count part
    coverage_in_lora: bool
        Whether to use the coverage information in the LoRA finetuning weights
    """

    def __init__(
        self,
        dna_cnn_model=None,
        hidden_layer_model=None,
        profile_cnn_model=None,
        dna_len=2114,
        output_len=1000,
        embeddings=None,
        coverages=None,
        rank=8,
        hidden_dim=None,
        n_lora_layers=0,
        lora_dna_cnn=False,
        lora_dilated_cnn=False,
        lora_pff_cnn=False,
        lora_profile_cnn=False,
        lora_count_cnn=False,
        coverage_in_lora=False,
    ):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.dna_len = dna_len
        self.output_len = output_len

        if coverages is not None:
            self.coverages = nn.Embedding(coverages.shape[0], coverages.shape[1])
            self.coverages.weight.data = torch.from_numpy(coverages).float()
            self.coverages.weight.requires_grad = False
        else:
            self.coverages = None
        if embeddings is not None:
            self.embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.embeddings.weight.data = torch.from_numpy(embeddings).float()
            self.embeddings.weight.requires_grad = False
            lora_embedding_dim = (
                embeddings.shape[-1] + coverages.shape[-1]
                if coverage_in_lora
                else embeddings.shape[-1]
            )
        else:
            self.embeddings = None
            self.coverages = None
        self.coverage_in_lora = coverage_in_lora

        # make the LoRA models for the DNA CNN, hidden layer, and profile CNN
        if lora_dna_cnn:
            assert self.embeddings is not None, "Embeddings must be provided for LoRA"
            self.dna_cnn_model.conv = Conv1dLoRA(
                self.dna_cnn_model.conv,
                A_embedding_dim=lora_embedding_dim,
                B_embedding_dim=lora_embedding_dim,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )
            rescale_factor = adjust_embedding_scale(
                rank,
                self.embeddings,
                self.coverages,
                self.coverage_in_lora,
                self.dna_cnn_model.conv.A_embedding,
                self.dna_cnn_model.conv.B_embedding,
            )
            self.dna_cnn_model.conv.reset_B()

        hidden_layers = self.hidden_layer_model.layers
        for i in range(len(hidden_layers)):
            if lora_dilated_cnn:
                assert self.embeddings is not None, "Embeddings must be provided for LoRA"
                hidden_layers[i].module.conv1 = Conv1dLoRA(
                    hidden_layers[i].module.conv1,
                    A_embedding_dim=lora_embedding_dim,
                    B_embedding_dim=lora_embedding_dim,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )
                rescale_factor = adjust_embedding_scale(
                    rank,
                    self.embeddings,
                    self.coverages,
                    self.coverage_in_lora,
                    hidden_layers[i].module.conv1.A_embedding,
                    hidden_layers[i].module.conv1.B_embedding,
                )
                hidden_layers[i].module.conv1.reset_B()

            if lora_pff_cnn:
                assert self.embeddings is not None, "Embeddings must be provided for LoRA"
                hidden_layers[i].module.conv2 = Conv1dLoRA(
                    hidden_layers[i].module.conv2,
                    A_embedding_dim=lora_embedding_dim,
                    B_embedding_dim=lora_embedding_dim,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )
                rescale_factor = adjust_embedding_scale(
                    rank,
                    self.embeddings,
                    self.coverages,
                    self.coverage_in_lora,
                    hidden_layers[i].module.conv2.A_embedding,
                    hidden_layers[i].module.conv2.B_embedding,
                )
                hidden_layers[i].module.conv2.reset_B()

        if lora_profile_cnn:
            assert self.embeddings is not None, "Embeddings must be provided for LoRA"
            self.profile_cnn_model.conv_layer = Conv1dLoRA(
                self.profile_cnn_model.conv_layer,
                A_embedding_dim=lora_embedding_dim,
                B_embedding_dim=lora_embedding_dim,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )
            rescale_factor = adjust_embedding_scale(
                rank,
                self.embeddings,
                self.coverages,
                self.coverage_in_lora,
                self.profile_cnn_model.conv_layer.A_embedding,
                self.profile_cnn_model.conv_layer.B_embedding,
            )
            self.profile_cnn_model.conv_layer.reset_B()

        # Historical code
        # if isinstance(self.profile_cnn_model.linear, nn.Linear):
        #     print("translating linear into conv1d")
        #     weight = self.profile_cnn_model.linear.weight.data
        #     print(weight.shape)
        #     bias = self.profile_cnn_model.linear.bias.data
        #     self.profile_cnn_model.linear = Conv1dWrapper(weight.shape[1], weight.shape[0], 1)
        #     print(self.profile_cnn_model.linear.conv.weight.shape)
        #     self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
        #     self.profile_cnn_model.linear.conv.bias.data = bias

        if lora_count_cnn:
            assert self.embeddings is not None, "Embeddings must be provided for LoRA"
            self.profile_cnn_model.linear = Conv1dLoRA(
                self.profile_cnn_model.linear,
                A_embedding_dim=lora_embedding_dim,
                B_embedding_dim=lora_embedding_dim,
                r=1,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )
            rescale_factor = adjust_embedding_scale(
                1,
                self.embeddings,
                self.coverages,
                self.coverage_in_lora,
                self.profile_cnn_model.linear.A_embedding,
                self.profile_cnn_model.linear.B_embedding,
            )
            self.profile_cnn_model.linear.reset_B()

        # Bias adjusted footprints head should come after the LoRA model
        if self.coverages is not None:
            self.profile_cnn_model = BiasAdjustedFootprintsHead(
                self.profile_cnn_model, self.coverages.weight.data.shape[1]
            )

    def return_origin(self):
        """
        This function returns the original bulk model not finetuned with LoRA
        Returns
        -------

        """
        self = self.to("cpu")
        model_clone = deepcopy(self)

        # If the conv part is not Conv1dWrapper, it would be a LoRA model, then we just return the layer of it
        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = model_clone.dna_cnn_model.conv.layer
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv1, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv1 = layer.module.conv1.layer
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv2, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv2 = layer.module.conv2.layer

        if isinstance(model_clone.profile_cnn_model, BiasAdjustedFootprintsHead):
            model_clone.profile_cnn_model = model_clone.profile_cnn_model.footprints_head

        if not isinstance(model_clone.profile_cnn_model.conv_layer, Conv1dWrapper):
            model_clone.profile_cnn_model.conv_layer = (
                model_clone.profile_cnn_model.conv_layer.layer
            )
        if not isinstance(model_clone.profile_cnn_model.linear, Conv1dWrapper):
            model_clone.profile_cnn_model.linear = model_clone.profile_cnn_model.linear.layer

        return model_clone

    def collapse(self, cell=None, turn_on_grads=True, A_cells=None, B_cells=None, coverages=None):
        """
        This function collapses the LoRA model to a model for one cell or subset of cells.

        Parameters
        ----------
        cell: int | torch.Tensor
            The cell to collapse the model to
        turn_on_grads: bool, optional
            Whether to turn on gradients for the model. Default is True.

        Returns
        -------

        """

        # self = self.to('cpu')
        model_clone = deepcopy(self)

        if cell is not None:
            if type(cell) not in [int, list, np.ndarray, torch.Tensor]:
                raise ValueError("cell must be integer(s)")
            if type(cell) is int:
                cell = [cell]
            if self.embeddings is not None:
                cell = torch.tensor(cell).long().to(self.embeddings.weight.data.device)
                A_cells = self.embeddings(cell)
                B_cells = self.embeddings(cell)
            if self.coverages is not None:
                coverages = self.coverages(cell)
            else:
                coverages = None
            if self.coverage_in_lora:
                A_cells = torch.cat([A_cells, coverages], dim=-1)
                B_cells = torch.cat([B_cells, coverages], dim=-1)

        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = model_clone.dna_cnn_model.conv.collapse_layer(
                A_cells, B_cells
            )
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv1, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv1 = layer.module.conv1.collapse_layer(A_cells, B_cells)
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv2, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv2 = layer.module.conv2.collapse_layer(A_cells, B_cells)

        if isinstance(model_clone.profile_cnn_model, BiasAdjustedFootprintsHead):
            model = model_clone.profile_cnn_model.footprints_head
            model_clone.profile_cnn_model.collapse_layer(coverages)
        else:
            model = model_clone.profile_cnn_model

        if not isinstance(model.conv_layer, Conv1dWrapper):
            model.conv_layer = model.conv_layer.collapse_layer(A_cells, B_cells)
        if not isinstance(model.linear, Conv1dWrapper):
            model.linear = model.linear.collapse_layer(A_cells, B_cells)
        if turn_on_grads:
            for p in model_clone.parameters():
                p.requires_grad = True

        return model_clone

    def forward(self, X, cells=None, output_len=None, modes=None):
        """
        This is the forward function for the seq2PRINT model

        Parameters
        ----------
        X: torch.Tensor
            The input data
        cells: torch.Tensor | None
            The cell to use for the model
        output_len: int | None
            The output length of the model
        modes: np.ndarray | None | torch.Tensor
            The modes to use for the model

        Returns
        -------
        score: tuple[torch.Tensor]
            The output of the model (predicted footprints and coverage)

        """
        if output_len is None:
            output_len = self.output_len

        A_cells, B_cells, coverages = None, None, None
        if (cells is not None) and (self.embeddings is not None):
            A_cells = self.embeddings(cells)
            B_cells = self.embeddings(cells)

            if self.coverages is not None:
                coverages = self.coverages(cells)
            if self.coverage_in_lora:
                A_cells = torch.cat([A_cells, coverages], dim=-1)
                B_cells = torch.cat([B_cells, coverages], dim=-1)
        # get the motifs
        X = self.dna_cnn_model(X, A_cells=A_cells, B_cells=B_cells)
        # get the hidden layer
        X = self.hidden_layer_model(X, A_cells=A_cells, B_cells=B_cells)
        # get the profile
        output = self.profile_cnn_model(
            X,
            A_cells=A_cells,
            B_cells=B_cells,
            coverages=coverages,
            output_len=output_len,
            modes=modes,
        )
        return output

    def load_train_state_dict(self, ema, optimizer, scaler, savename):
        """
        This function loads the training state dict
        """

        print("Nan training loss, load last OK-ish checkpoint")
        device = next(self.parameters()).device
        print(device)
        self.cpu()
        checkpoint = torch.load(savename)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        self.to(device)
        optimizer = optimizer.to(device)
        scaler = scaler.to(device)

        del checkpoint
        torch.cuda.empty_cache()
        if ema:
            # ema = ema.cpu()
            ema = torch.load(savename + ".ema.pt")
        ema = ema.to(device)
        return ema, optimizer, scaler

    def fit(
        self,
        dispmodel,
        training_data,
        validation_data=None,
        validation_size=None,
        max_epochs=100,
        optimizer=None,
        scheduler=None,
        validation_freq=100,
        early_stopping=False,
        return_best=True,
        savename="model",
        modes=np.arange(2, 101, 1),
        downsample=None,
        ema=None,
        use_amp=False,
        use_wandb=False,
        accumulate_grad=1,
        batch_size=None,
        coverage_warming=0,
        training_num_modes=30,
    ):
        """
        This is the fit function for seq2PRINT

        Parameters
        ----------
        dispmodel: torch.nn.Module
            The dispersion model to use for the footprint prediction
        training_data: torch.utils.data.DataLoader
            The training data loader
        validation_data: torch.utils.data.DataLoader | None
            The validation data loader
        validation_size: int | None
            The number of batches to validate on
        max_epochs: int
            The maximum number of epochs to train for
        optimizer: torch.optim.Optimizer
            The optimizer to use
        scheduler: torch.optim.lr_scheduler
            The scheduler to use
        validation_freq: int
            The frequency of validation (after how many batches of training data)
        early_stopping: bool | int
            Whether to use early stopping, if int, then the number of epochs to wait before stopping
        return_best: bool
            Whether to return the best model
        savename: str
            The name to save the model to
        modes: np.ndarray
            The modes to use for the footprint training and prediction
        downsample: float | None
            The amount to downsample the data, think it as a dropout rate for the input data
        ema: torch.nn.Module
            The exponential moving average model
        use_amp: bool
            Whether to use automatic mixed precision
        use_wandb: bool
            Whether to use wandb for logging
        accumulate_grad: int
            The number of gradients to accumulate before stepping
        batch_size: int | None
            The batch size to use
        coverage_warming: int
            The number of epochs to warm up to get used to the coverage shift
        training_num_modes: int
            The number of modes to use for training

        """

        batch_size = training_data.batch_size if batch_size is None else batch_size
        early_stopping_counter = 0

        best_loss = np.inf

        modes_all = list(np.arange(2, 101, 1))
        index_all = np.zeros(int(np.max(modes_all) + 1), dtype=int)
        for i, mode in enumerate(modes_all):
            index_all[mode] = i

        device = next(self.parameters()).device
        loss_history = []
        assert (
            validation_size is not None or validation_data is not None
        ), "Either validation_size or validation_data should be provided"

        if validation_data is None:
            validation_data = training_data
        if validation_size is None:
            validation_size = len(validation_data)
        if validation_freq is None:
            validation_freq = len(training_data)
        if use_amp:
            print("Using amp")

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if coverage_warming > 0:
            # This is a dict to store the requires_grads for each parameter so we can revert back to the original state when coverage warming is done
            requires_grads = {}
            for name, param in self.named_parameters():
                if name in requires_grads:
                    if requires_grads[name] != param.requires_grad:
                        print("duplicated name", name, param.requires_grad, requires_grads[name])
                        raise ValueError("Duplicated name")
                requires_grads[name] = param.requires_grad
                param.requires_grad = False
            for p in self.profile_cnn_model.adjustment_count.parameters():
                p.requires_grad = True
            for p in self.profile_cnn_model.adjustment_footprint.parameters():
                p.requires_grad = True
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print("total trainable params", total_params)
            for group in optimizer.param_groups:
                lr = group["lr"]
                print("old lr", lr)
                group["lr"] = min(1e-3, group["lr"] * 10)
                print("new lr", group["lr"])

        for epoch in range(max_epochs):
            moving_avg_loss = 0
            iteration = 0

            bar = trange(
                validation_freq,
                desc=(
                    " - (Training) {epoch}".format(epoch=epoch + 1) + " warming"
                    if (coverage_warming > 0 and epoch <= coverage_warming)
                    else ""
                ),
                leave=False,
                dynamic_ncols=True,
            )

            if (epoch == coverage_warming) and coverage_warming > 0:
                for name, param in self.named_parameters():
                    param.requires_grad = requires_grads[name]
                for group in optimizer.param_groups:
                    print("revert lr", lr)
                    group["lr"] = lr
                # for p in self.profile_cnn_model.adjustment_count.parameters():
                #     p.requires_grad = False
                # for p in self.profile_cnn_model.adjustment_footprint.parameters():
                #     p.requires_grad = False

            # At the beginning of each epoch, we resample the training data
            training_data_epoch_loader = training_data.resample()
            for data in training_data_epoch_loader:

                try:
                    autocast_context = torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                    )
                except RuntimeError:
                    autocast_context = torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=use_amp
                    )  # For older GPUs
                with autocast_context:
                    random_modes = np.random.permutation(modes)[
                        :training_num_modes
                    ]  # randomly select 30 modes for training
                    select_index = torch.as_tensor(index_all[random_modes])
                    if len(data) == 2:
                        X, y = data
                        cell = None
                        norm_cov = None
                    else:
                        X, y, cell, peak, norm_cov = data
                        X = X[:batch_size]
                        y = y[:batch_size]
                        cell = cell[:batch_size, 0]
                        cell = cell.to(device)
                        # norm_cov = norm_cov[:batch_size].to(device)
                        norm_cov = None

                    X = X.to(device)
                    y = y.to(device)

                    atac = y[:, 0]
                    if downsample is not None:
                        atac = F.dropout(atac, 1 - downsample, training=self.training)

                    footprints = multiscaleFoot(atac, y[:, 1], random_modes, dispmodel)
                    mask = ~torch.isnan(footprints)

                    if norm_cov is not None:
                        coverage = norm_cov
                    else:
                        coverage = y[:, 0].sum(dim=-1)
                        coverage = torch.log1p(coverage)

                    pred_footprint, pred_coverage = self.forward(X, cell, modes=select_index)

                    desc_str = " - (Training) {epoch}".format(epoch=epoch + 1)

                    loss_footprint = F.mse_loss(pred_footprint[mask], footprints[mask])
                    desc_str += " Footprint Loss: {loss:.2f}".format(loss=loss_footprint.item())

                    loss_coverage = F.mse_loss(coverage, pred_coverage)
                    desc_str += " Coverage Loss: {loss:.2f}".format(loss=loss_coverage.item())
                    desc_str += (
                        " warming" if (coverage_warming > 0 and epoch <= coverage_warming) else ""
                    )
                    loss = (loss_footprint + loss_coverage) / accumulate_grad
                    if np.isnan(loss.item()):
                        ema, optimizer, scaler = self.load_train_state_dict(
                            ema, optimizer, scaler, savename
                        )
                        continue

                scaler.scale(loss).backward()
                moving_avg_loss += loss_footprint.item()
                if (iteration + 1) % accumulate_grad == 0:
                    scaler.unscale_(
                        optimizer
                    )  # Unscale gradients for clipping without inf/nan gradients affecting the model
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=5.0
                    )  # Adjust max_norm accordingly

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # if scaler._scale < min_scale:
                    #     scaler._scale = torch.tensor(min_scale).to(scaler._scale)
                    if (epoch >= coverage_warming) and ema:
                        ema.update()

                    if scheduler:
                        scheduler.step()

                bar.set_description(desc_str)
                bar.update(1)
                iteration += 1
                if iteration >= validation_freq:
                    break

            print(
                " - (Training) {epoch} Loss: {loss:.2f}".format(
                    epoch=epoch + 1, loss=moving_avg_loss / iteration
                )
            )

            bar.close()
            self.eval()

            val_loss, profile_pearson, across_pearson_fp, across_pearson_cov = (
                validation_step_footprint(self, validation_data, validation_size, dispmodel, modes)
            )

            if np.isnan(val_loss):
                print("Nan loss, load last OK-ish checkpoint")
                ema, optimizer, scaler = self.load_train_state_dict(
                    ema, optimizer, scaler, savename
                )

            print(
                " - (Validation) {epoch} \
                        Loss: {loss:.5f}".format(
                    epoch=epoch + 1, loss=val_loss
                )
            )
            print("Profile pearson", profile_pearson)
            print("Across peak pearson fp", across_pearson_fp)
            print("Across peak pearson cov", across_pearson_cov)

            if (epoch >= coverage_warming) and ema:
                ema.eval()
                ema.ema_model.eval()
                val_loss_ema, profile_pearson, across_pearson_fp, across_pearson_cov = (
                    validation_step_footprint(
                        ema.ema_model, validation_data, validation_size, dispmodel, modes
                    )
                )
                if val_loss_ema > val_loss:
                    # ema not converged yet:
                    early_stopping_counter = 0

                if np.isnan(val_loss_ema):
                    print("Nan loss, load last OK-ish checkpoint")
                    ema, optimizer, scaler = self.load_train_state_dict(
                        ema, optimizer, scaler, savename
                    )

                print(
                    " - (Validation) {epoch} \
                Loss: {loss:.5f}".format(
                        epoch=epoch + 1, loss=val_loss_ema
                    )
                )
                print("EMA Profile pearson", profile_pearson)
                print("EMA Across peak pearson fp", across_pearson_fp)
                print("EMA Across peak pearson cov", across_pearson_cov)
                ema.train()
                val_loss = val_loss_ema

            self.train()

            loss_history.append([moving_avg_loss / iteration, val_loss])

            if val_loss < best_loss:
                print("best loss", val_loss)
                best_loss = val_loss
                early_stopping_counter = 0
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema else None,
                }
                torch.save(checkpoint, savename)
                torch.save(self, savename + ".model.pt")
                if ema:
                    torch.save(ema, savename + ".ema.pt")
                    torch.save(ema, savename + ".ema.pt")
                    torch.save(ema.ema_model, savename + ".ema_model.pt")
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    if epoch < coverage_warming:
                        print("Early stopping on coverage warming")
                        coverage_warming = epoch
                    else:
                        print("Early stopping")
                        break

            if use_wandb:
                wandb.log(
                    {
                        "train/train_loss": moving_avg_loss / iteration,
                        "val/val_loss": val_loss,
                        "val/best_val_loss": best_loss,
                        "val/profile_pearson": profile_pearson,
                        "val/across_pearson_footprint": across_pearson_fp,
                        "val/across_pearson_coverage": across_pearson_cov,
                        "epoch": epoch,
                    }
                )

        if return_best:
            self.load_state_dict(torch.load(savename, weights_only=False)["state_dict"])
            print("loaded best model")

        return epoch, loss_history

    def predict(self, *args, **kwargs):
        return predict(self, *args, **kwargs)
