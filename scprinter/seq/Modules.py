import copy

import numpy as np
import torch.nn as nn

from .Functions import *


class Residual(nn.Module):
    """
    This is a simple residual connection
    return x + module(x)

    Parameters
    ----------
    module: nn.Module
        the module to be wrapped
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)


# I assume the lr would be at 1e-3 scale, thus, the times 10 rescale the lr to 1e-2 scale
class ReZero(nn.Module):
    """
    This is a residual connection with the rezero trick with a learnable weight

    Parameters
    ----------
    module: nn.Module
        the module to be wrapped

    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.weight1 = nn.Parameter(torch.Tensor([0.001]))

    def forward(self, X, *args, **kwargs):
        return torch.add(X, self.module(X, *args, **kwargs) * (self.weight1 * 10))


# Do nothing just pass,
class Pass(nn.Module):
    """
    Just pass the input through the module. This is used as a placeholder class

    Parameters
    ----------
    module: nn.Module
        the module to be wrapped

    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X, *args, **kwargs):
        return self.module(X, *args, **kwargs)


class Conv1dWrapper(nn.Module):
    """
    This is a wrapper for the Conv1d layer, it can be used to select a subset of the weights (modes for footprint).
    All the parameters are passed directly to the Conv1d layer


    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, X, *args, modes=None, **kwargs):
        """

        Parameters
        ----------
        X: torch.Tensor
            shape=(batch_size, n_filters, seq_len)
        args:
            Just a placeholder
        modes: torch.Tensor | None
            The filters to be selected
        kwargs:
            Just a placeholder

        Returns
        -------

        """
        # The args, and kwargs are just placeholders
        # Use modes to select a subset of the weights
        return (
            self.conv(X)
            if modes is None
            else F.conv1d(
                X,
                self.conv.weight[modes],
                self.conv.bias[modes],
                stride=self.conv.stride[0],
                padding=self.conv.padding[0],
                dilation=self.conv.dilation[0],
                groups=self.conv.groups,
            )
        )


class DNA_CNN(nn.Module):
    """
    This is actually as simple as one CNN layer,
    It's used to extract the DNA sequence features (the first layer)
    just to make it easy to construct a seq2PRINT model
    """

    def __init__(self, n_filters=64, kernel_size=21, activation=nn.ReLU(), in_channels=4):
        """

        Parameters
        ----------
        n_filters: int
            number of filters
        kernel_size: int
            kernel size
        activation: nn.Module
        in_channels: int
            number of input channels
        """
        super().__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv = Conv1dWrapper(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = copy.deepcopy(activation)

    def forward(self, X, *args, **kwargs):
        X = self.conv(X, *args, **kwargs)
        X = self.activation(X)
        return X


class ConvBlockModule(nn.Module):
    """
    A Convolutional Block with grouped convolution followed by 1x1 convolution (PFF)

    Parameters
    ----------
    n_filters: int
        number of kernels
    bottleneck: int
        number of output kernels in the first layer
    kernel_size: int
        kernel size
    dilation: int
        dilation rate
    activation: nn.Module
        activation function
    batch_norm: bool
        batch normalization in between layers
    batch_norm_momentum: float
        batch normalization momentum
    groups: int
        number of groups in the conv layer
    inception_version: int
        2 or 3, 2 stands for 2 layers, 3 stands for 3 layers
    empty: bool
        if True, don't actually initialize the model, and one can do post-hoc initalization. This is only used to transform some old trained models to the latest format.
    """

    def __init__(
        self,
        n_filters=64,
        bottleneck=None,
        kernel_size=3,
        dilation=1,
        activation=nn.ReLU(),
        batch_norm=False,
        batch_norm_momentum=0.1,
        groups=8,
        inception_version=2,
        empty=False,
    ):

        super().__init__()
        if empty:
            return
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.batch_norm = batch_norm
        self.inception_version = inception_version

        bottleneck = n_filters // 2 if bottleneck is None else bottleneck
        self.bottleneck = bottleneck

        if inception_version == 2:
            # Model contains
            # grouped conv (n_filters, bottleneck, kernel_size) -> (batch norm) -> (bottleneck, n_filters, 1) -> (batch norm)

            self.conv1 = Conv1dWrapper(
                in_channels=n_filters,
                out_channels=bottleneck,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size // 2),
                groups=groups,
                bias=False,
            )
            self.block1 = nn.Sequential(
                (
                    nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum)
                    if batch_norm
                    else nn.Identity()
                ),
                copy.deepcopy(activation),
            )
            self.conv2 = Conv1dWrapper(
                in_channels=bottleneck,
                out_channels=n_filters,
                kernel_size=1,
                bias=False,
            )
            self.block2 = nn.Sequential(
                (
                    nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum)
                    if batch_norm
                    else nn.Identity()
                ),
                copy.deepcopy(activation),
            )

            nn.init.kaiming_normal_(self.conv1.weight.data, mode="fan_out")
            nn.init.kaiming_normal_(self.conv2.weight.data, mode="fan_out")

            self.block1[0].weight.data[...] = 1
            self.block2[0].weight.data[...] = 1

            self.block1[0].bias.data[...] = 0
            self.block2[0].bias.data[...] = 0
            self.block3 = None
        else:
            # Model contains
            # grouped conv (n_filters, bottleneck, 1) -> (batch norm) ->
            # (bottleneck, bottleneck, 3) -> (batch norm) ->
            # (bottleneck, n_filters, 1) -> (batch norm)

            self.conv1 = Conv1dWrapper(
                in_channels=n_filters,
                out_channels=bottleneck,
                kernel_size=1,
                bias=False,
            )

            self.block1 = nn.Sequential(
                (
                    nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum)
                    if batch_norm
                    else nn.Identity()
                ),
                copy.deepcopy(activation),
            )
            self.conv2 = Conv1dWrapper(
                in_channels=bottleneck,
                out_channels=bottleneck,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size // 2),
                groups=groups,
                bias=False,
            )

            self.block2 = nn.Sequential(
                (
                    nn.BatchNorm1d(bottleneck, momentum=batch_norm_momentum)
                    if batch_norm
                    else nn.Identity()
                ),
                copy.deepcopy(activation),
            )
            self.conv3 = (
                Conv1dWrapper(
                    in_channels=bottleneck,
                    out_channels=n_filters,
                    kernel_size=1,
                    bias=False,
                ),
            )
            self.block3 = nn.Sequential(
                (
                    nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum)
                    if batch_norm
                    else nn.Identity()
                ),
                copy.deepcopy(activation),
            )

            nn.init.kaiming_normal_(self.conv1.weight.data, mode="fan_out")
            nn.init.kaiming_normal_(self.conv2.weight.data, mode="fan_out")
            nn.init.kaiming_normal_(self.conv3.weight.data, mode="fan_out")

            self.block1[0].weight.data[...] = 1
            self.block2[0].weight.data[...] = 1
            self.block3[0].weight.data[...] = 1

            self.block1[0].bias.data[...] = 0
            self.block2[0].bias.data[...] = 0
            self.block3[0].bias.data[...] = 0

    @classmethod
    def from_inception_v1(cls, inception_v1):
        model = cls(empty=True)
        model.conv1 = inception_v1.layers[0]
        model.block1 = inception_v1.layers[1:3]
        model.conv2 = inception_v1.layers[3]
        model.block2 = inception_v1.layers[4:6]
        if len(inception_v1.layers) > 6:
            model.block3 = inception_v1.layers[6:]
        else:
            model.block3 = None
        return model

    def forward(self, X, *args, **kwargs):
        """
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)

        Returns
        -------

        """
        X = self.conv1(X, *args, **kwargs)
        X = self.block1(X)
        X = self.conv2(X, *args, **kwargs)
        X = self.block2(X)
        if self.block3 is not None:
            X = self.conv3(X, *args, **kwargs)
            X = self.block3(X)
        return X


class DilatedConvModule(nn.Module):
    """
    Good old dilated conv network

    Parameters
    ----------
    n_filters: int
        number of filters
    kernel_size: int
        kernel size
    activation: nn.Module
        activation function
    batch_norm: bool
        batch normalization
    batch_norm_momentum: float
        batch normalization momentum
    dilation: int
        dilation rate
    """

    def __init__(
        self,
        n_filters=64,
        kernel_size=3,
        activation=nn.ReLU(),
        batch_norm=False,
        batch_norm_momentum=0.1,
        dilation=1,
    ):
        super().__init__()
        self.conv = (
            Conv1dWrapper(
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size // 2),
            ),
        )
        self.block = nn.Sequential(
            (
                nn.BatchNorm1d(n_filters, momentum=batch_norm_momentum)
                if batch_norm
                else nn.Identity()
            ),
            copy.deepcopy(activation),
        )

    def forward(self, X, *args, **kwargs):
        X = self.conv(X, *args, **kwargs)
        X = self.block(X)
        return X


class DilatedCNN(nn.Module):
    """
    A stack of conv blocks (the middle part of the seq2PRINT model)

    Parameters
    ----------
    n_filters: int
        number of filters
    bottleneck: int
        number of output kernels in the first layer
    n_layers: int
        number of layers
    kernel_size: int
        kernel size
    groups: int
        number of groups in the conv layer
    activation: nn.Module
        activation function
    batch_norm: bool
        batch normalization
    batch_norm_momentum: float
        batch normalization momentum
    rezero: bool
        use rezero trick
    residual: bool
        use residual connection
    inception: bool
        use inception module or the good old dilated conv
    inception_version: int
        2 or 3, 2 stands for 2 layers, 3 stands for 3 layers, passed to ConvBlockModule
    dilation_func: function
        a function to generate dilation rates, deciding sth like whether it's 1,2,4,8,16,32 or 2,4,8,16,32,64
    """

    def __init__(
        self,
        n_filters=64,
        bottleneck=64,
        n_layers=6,
        kernel_size=3,
        groups=8,
        activation=nn.ReLU(),
        batch_norm=False,
        batch_norm_momentum=0.1,
        rezero=False,
        residual=True,
        inception=True,
        inception_version=2,
        dilation_func=None,
    ):
        super().__init__()
        if dilation_func is None:
            dilation_func = lambda x: 2 ** (x + 1)

        self.n_filters = n_filters
        self.botleneck = bottleneck
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.groups = groups
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.rezero = rezero
        self.residual = residual
        self.inception = inception
        self.inception_version = inception_version

        residual_class = Residual if residual else Pass
        if rezero:
            residual_class = ReZero
        self.layers = nn.ModuleList(
            [
                residual_class(
                    ConvBlockModule(
                        n_filters=n_filters,
                        bottleneck=bottleneck,
                        kernel_size=kernel_size,
                        dilation=dilation_func(i),
                        activation=activation,
                        batch_norm=batch_norm,
                        batch_norm_momentum=batch_norm_momentum,
                        groups=groups,
                        inception_version=inception_version,
                    )
                    if incpt
                    else DilatedConvModule(
                        n_filters=n_filters,
                        kernel_size=kernel_size,
                        activation=activation,
                        batch_norm=batch_norm,
                        batch_norm_momentum=batch_norm_momentum,
                        dilation=dilation_func(i),
                    )
                )
                for i, incpt in enumerate(inception)
            ]
        )

    def forward(self, X, *args, **kwargs):
        """
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)

        Returns
        -------

        """
        for i in range(self.n_layers):
            X = self.layers[i](X, *args, **kwargs)
        return X


class BiasAdjustedFootprintsHead(nn.Module):
    """
    This is the output head of the footprints model with a bias (such as sequencing depths) adjustment

    Parameters
    ----------
    footprints_head: nn.Module
        the footprints head
    coverages: nn.Module
        A embedding module to fetch the coverages

    """

    def __init__(self, footprints_head, bias_dim):
        super().__init__()
        self.footprints_head = footprints_head
        self.adjustment_footprint = nn.Sequential(
            nn.Linear(in_features=bias_dim, out_features=min(int(bias_dim * 16), 32), bias=True),
            # nn.Sigmoid(),  # For non-linearity
            nn.GELU(),
            nn.Linear(in_features=min(int(bias_dim * 16), 32), out_features=1, bias=True),
        )
        # Can't be all zeros, otherwise they share gradient.. then it's nonsense.
        for n, p in self.adjustment_footprint.named_parameters():
            p.data = (
                torch.zeros_like(p.data) + torch.randn_like(p.data) * 0.01
            )  # -1 * p.data if torch.sum(p.data) < 0 else p.data
        # Only last layer bias should be 1
        self.adjustment_footprint[-1].bias.data = (
            torch.tensor([1.0]) + torch.randn_like(self.adjustment_footprint[-1].bias.data) * 0.01
        )

        # Assuming linear adjustment for counts
        self.adjustment_count = nn.Linear(in_features=bias_dim, out_features=1, bias=True)
        self.adjustment_count.weight.data = (
            torch.zeros_like(self.adjustment_count.weight.data)
            + torch.randn_like(self.adjustment_count.weight.data) * 0.01
        )
        self.adjustment_count.bias.data = (
            torch.tensor([1.0]) + torch.randn_like(self.adjustment_count.bias.data) * 0.01
        )
        # self.coverages = bias_dim
        self.adjust_foot = 1
        self.adjust_count = 1

    def forward(self, X, coverages=None, *args, **kwargs):
        """

        Parameters
        ----------
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)
        cells: torch.tensor | None
            the cell indices, if None, use the global adjustment (which is no adjustment)
        args:
            pass to the footprints_head
        kwargs
            pass to the footprints_head

        Returns
        -------

        """
        if coverages is not None:
            adjustment_foot = self.adjustment_footprint(coverages)[
                :, None
            ]  # because footprint has shape of (..., bp, scales)
            adjustment_count = self.adjustment_count(coverages)[
                :, 0
            ]  # because count has shape of (bs, )
        else:
            adjustment_foot = self.adjust_foot
            adjustment_count = self.adjust_count
        X_bindingscore, X_count = self.footprints_head(X, *args, **kwargs)
        X_bindingscore = X_bindingscore * adjustment_foot
        X_count = X_count * adjustment_count

        return X_bindingscore, X_count

    def collapse_layer(self, coverage):
        """
        Collapse the adjustment to single values. This is used such that the LoRA model or a giant model for all cells would be collapsed to a single model for a specific cell or the average for a set of cells.

        Parameters
        ----------
        coverage

        Returns
        -------

        """

        # if type(cell) not in [int, list, np.ndarray, torch.Tensor]:
        #     raise ValueError("cell must be an integer")
        # if type(coverage) not in [list, np.ndarray, torch.Tensor]:
        #     coverage = [coverage]
        coverage = coverage.to(self.adjustment_footprint[0].weight.data.device)
        self.adjust_foot = self.adjustment_footprint(coverage)[:, 0].mean(axis=0)
        self.adjust_count = self.adjustment_count(coverage)[:, 0].mean(axis=0)

        # print("collapsing", self.adjust_foot, self.adjust_count)


class Footprints_head(nn.Module):
    """
    This is the output head of the footprints model


    Parameters
    ----------
    n_filters: int
        number of filters
    kernel_size: int
        kernel size
    n_scales: int
        number of footprints scales
    per_peak_feats: int
        number of features per peak (such as counts, this is for the count)
    """

    def __init__(self, n_filters, kernel_size=5, n_scales=50):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_scale = n_scales

        # For the footprint
        self.conv_layer = Conv1dWrapper(
            in_channels=n_filters,
            out_channels=n_scales,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        # For the count
        self.linear = Conv1dWrapper(
            in_channels=n_filters,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, X, *args, output_len=None, modes=None, **kwargs):
        """

        Parameters
        ----------
        X
        args:
            args pass to the conv_layer
        output_len: int | None
            the output length, if None, keep the same length as the input. This is used to trim the model
        modes: torch.Tensor | None
            the modes to be selected for calculating footprints
        kwargs
            pass to the conv_layer

        Returns
        -------

        """
        X_bindingscore = self.conv_layer(X, *args, modes=modes, **kwargs)
        # # Empircally this normalization makes the training more stable
        # X_bindingscore = X_bindingscore - torch.sum(X_bindingscore, dim=(1,2), keepdims=True) / (X_bindingscore.shape[-1] * X_bindingscore.shape[-2])
        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len)
            trim = (X_bindingscore.shape[-1] - output_len_needed_in_X) // 2

        if trim > 0:
            X_bindingscore = X_bindingscore[..., trim:-trim]

        if isinstance(self.linear, nn.Linear):
            X_count = self.linear(X.detach().mean(dim=-1) if self.training else X.mean(dim=-1))[
                ..., 0
            ]
        else:
            X_count = self.linear(
                (
                    X.detach().mean(dim=-1, keepdims=True)
                    if self.training  # detach because I constrained that the count cannot affect how the motifs are learned
                    else X.mean(dim=-1, keepdims=True)
                ),
                *args,
                **kwargs,
            )[
                ..., 0, 0
            ]  # the output shape is (bs, 1, 1), (kernel size 1, pos 1) so we need to squeeze it to (bs, )

        return X_bindingscore, X_count


class Conv1dLoRA(nn.Module):
    """
    This is the Conv1d layer with the LoRA

    Parameters
    ----------
    layer: Conv1dWrapper
        the layer to be wrapped with LoRA
    A_embedding_dim_: int
        The embedding dimension for A
    B_embedding_dim: int
        The embedding dimension for B
    r: int
        LoRA rank
    alpha: float
        LoRA hyperparameter that controls scale
    hidden_dim: int
        hidden dimension from embedding -> LoRA parameters
    n_layers: int
        number of layers in the network that goes from embedding -> LoRA parameters
    """

    def __init__(
        self,
        layer,
        A_embedding_dim=None,
        B_embedding_dim=None,
        r=8,
        alpha=None,
        hidden_dim=None,
        n_layers=0,
    ):
        super().__init__()
        assert isinstance(layer, Conv1dWrapper), "The layer must be a Conv1dWrapper layer"
        self.layer = layer
        self.pretrain_conv = layer.conv  # The actual conv layer behind Conv1dWrapper
        # Fetch the parameters from the pretrain_conv
        self.layer_dim_in = self.pretrain_conv.in_channels
        self.layer_dim_out = self.pretrain_conv.out_channels
        self.kernel_size = self.pretrain_conv.kernel_size[0]
        self.dilation = self.pretrain_conv.dilation[0]
        self.padding = self.pretrain_conv.padding[0]
        self.groups = self.pretrain_conv.groups

        self.A_embedding_dim = A_embedding_dim
        self.B_embedding_dim = B_embedding_dim

        if alpha is None:
            alpha = r

        self.scale = alpha / r
        self.r = r

        if hidden_dim is None:
            self.hidden_dim = self.A_embedding_dim
        else:
            self.hidden_dim = hidden_dim

        layers = (
            [
                # A_embedding,
                nn.Linear(in_features=self.A_embedding_dim, out_features=self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.GELU(),
            ]
            + [
                Residual(
                    nn.Sequential(
                        nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                        nn.BatchNorm1d(self.hidden_dim),
                        nn.GELU(),
                    )
                )
            ]
            * n_layers
            + [
                nn.Linear(
                    in_features=self.hidden_dim,
                    out_features=int(
                        self.layer_dim_in * r
                    ),  # lead to a weight matrix of shape (r, layer_dim_in)
                ),
            ]
        )
        self.A_embedding = nn.Sequential(*layers)

        layers = (
            [
                # B_embedding,
                nn.Linear(in_features=self.B_embedding_dim, out_features=self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.GELU(),
            ]
            + [
                Residual(
                    nn.Sequential(
                        nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                        nn.BatchNorm1d(self.hidden_dim),
                        nn.GELU(),
                    )
                )
            ]
            * n_layers
            + [
                nn.Linear(
                    in_features=self.hidden_dim,
                    out_features=int(
                        self.layer_dim_out * r * self.kernel_size / self.groups
                    ),  # lead to a weight matrix of shape (layer_dim_out, r)
                ),
            ]
        )
        self.B_embedding = nn.Sequential(*layers)

        # When combined, this will lead to a weight matrix of shape (layer_dim_out, layer_dim_in, kernel_size)

    @torch.no_grad()
    def reset_B(self):
        ## Make sure B starts as all zeros:
        # Only make the last layer to be 0
        for i in range(len(self.B_embedding)):
            if isinstance(self.B_embedding[i], nn.Linear):
                last_layer = self.B_embedding[i]
            elif isinstance(self.B_embedding[i], nn.Sequential):
                for j in range(len(self.B_embedding[i])):
                    if isinstance(self.B_embedding[i][j], nn.Linear):
                        last_layer = self.B_embedding[i][j]
        last_layer.bias.data[...] = 0
        last_layer.weight.data[...] = 0

    @torch.no_grad()
    def collapse_layer(self, A_cell, B_cell):
        """
        Collapse the LoRA layer to a constant conv1d layer with the weight matrix collapsed from the A and B embeddings at a specific cell embedding or the average of a set of cell embeddings
        Parameters
        ----------
        cell

        Returns
        -------

        """

        # if type(cell) not in [int, list, np.ndarray, torch.Tensor]:
        #     raise ValueError("cell must be an integer")
        # if type(A_cell) not in [list, np.ndarray, torch.Tensor]:
        #     A_cell = [A_cell]
        # if type(B_cell) not in [list, np.ndarray, torch.Tensor]:
        #     B_cell = [B_cell]

        A_cell = A_cell.to(self.A_embedding[0].weight.data.device)
        B_cell = B_cell.to(self.B_embedding[0].weight.data.device)

        A = self.A_embedding(A_cell).mean(dim=0)  # (shape of self.layer_dim_in * r)
        B = self.B_embedding(B_cell).mean(
            dim=0
        )  # (shape of self.layer_dim_out * r * kernel_size / groups)

        if self.kernel_size == 1 and self.groups == 1:
            # This is just to be consistent with the forward case.
            A = A.reshape((self.r, self.layer_dim_in))
            B = B.reshape((self.layer_dim_out, self.r))
            weight = torch.matmul(B, A)[..., None]
        else:
            A = A.reshape((int(self.layer_dim_in), self.r))
            B = B.reshape((self.r, int(self.layer_dim_out / self.groups) * self.kernel_size))
            weight = torch.matmul(A, B).reshape(
                (
                    self.layer_dim_out,
                    int(self.layer_dim_in / self.groups),
                    self.kernel_size,
                )
            )
        weight_scaled = weight * self.scale
        new_layer = copy.deepcopy(self.layer)
        new_layer.conv.weight.data[...] = new_layer.conv.weight.data + weight_scaled
        return new_layer

    def forward(self, X, A_cells, B_cells, modes=None, **kwargs):
        A = self.A_embedding(A_cells)  # (shape of (bs, self.layer_dim_in * r)
        B = self.B_embedding(
            B_cells
        )  # (shape of (bs, self.layer_dim_out * r * kernel_size / groups)
        if self.kernel_size == 1 and self.groups == 1:
            # When kernel_size == 1, the convolution is actually a linear layer, take a short path
            A = A.reshape((-1, self.r, self.layer_dim_in))
            B = B.reshape(
                (-1, self.layer_dim_out, self.r)
            )  # because both kernel_size and groups are 1
            # x: (batch_size, layer_dim_in, seq_len)
            lora_x = torch.bmm(A, X)  # (batch_size, r, seq_len)
            if modes is not None:
                B = B[:, modes]
            lora_x = torch.bmm(B, lora_x)  # (batch_size, layer_dim_out, seq_len
            return lora_x * self.scale + (self.layer(X, modes=modes))
        else:
            # When kernel_size > 1, the convolution can be written as groupped convolution,
            # take a long path
            bs = X.shape[0]  # batch_size
            A = A.reshape((bs, int(self.layer_dim_in), self.r))
            B = B.reshape((bs, self.r, int(self.layer_dim_out / self.groups) * self.kernel_size))
            assert modes is None, "modes is not supported for kernel_size > 1"
            weight = torch.bmm(A, B).reshape(
                (
                    bs * self.layer_dim_out,
                    int(self.layer_dim_in / self.groups),
                    self.kernel_size,
                )
            )
            # size of (batch_size * layer_dim_out, layer_dim_in / groups, kernel_size)
            # route 1
            # weight = weight.reshape((-1, int(self.layer_dim_in / self.groups), self.kernel_size))
            # size of (batch_size * layer_dim_out, layer_dim_in / groups, kernel_size)
            lora_x = F.conv1d(
                X.reshape(
                    (1, -1, X.shape[-1])
                ),  # X after reshape (1, batch_size * layer_dim_in, seq_len)
                weight=weight,
                bias=None,
                dilation=self.dilation,
                groups=bs * self.groups,
                padding=self.padding,
            )  # each batch_size is a group
            # within each group, the convolution projects from (layer_dim_in, seq_len) to (layer_dim_out, seq_len)
            # This is equivalent to a for loop over each sample in the batch
            lora_x = lora_x.view(bs, self.layer_dim_out, -1)
            X = lora_x * self.scale + self.layer(X, modes=modes)

            return X


class CumulativeCounter:
    """Cumulative counter for calculating mean and sum of values."""

    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update the cumulative counter with a new value.

        Parameters
        ----------
            value (np.ndarray or torch.Tensor): The value to be added to the counter.
        """
        try:
            self.total += float(np.nansum(value))
        except TypeError:
            # torch
            self.total += float(torch.nansum(value).detach().cpu().item())
        # both numpy and torch will work
        self.count += np.prod(value.shape)

    def mean(self) -> float:
        """
        Calculate the mean of the values in the counter.

        Returns
        -------
            float: The mean value.
        """
        if self.count == 0:
            return 0
        return self.total / self.count

    def sum(self) -> float:
        """
        Calculate the sum of the values in the counter.

        Returns
        -------
            float: The sum value.
        """
        return self.total


class CumulativePearson:
    """Cumulative pearson counter for calculating the pearson correlation coefficient."""

    def __init__(self):
        self.count = 0
        self.x_counter = CumulativeCounter()
        self.y_counter = CumulativeCounter()
        self.xy_counter = CumulativeCounter()
        self.x2_counter = CumulativeCounter()
        self.y2_counter = CumulativeCounter()

    def update(
        self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Update the cumulative pearson counter with new values.

        Parameters
        ----------
            x (np.ndarray or torch.Tensor): The x values to be added to the counter.
            y (np.ndarray or torch.Tensor): The y values to be added to the counter.
        """
        self.x_counter.update(x)
        self.y_counter.update(y)
        self.xy_counter.update(x * y)
        self.x2_counter.update(x**2)
        self.y2_counter.update(y**2)

    def corr(self) -> float:
        """
        Calculate the pearson correlation coefficient.

        Returns
        -------
            float: The pearson correlation coefficient.
        """
        nx = self.x_counter.count
        ny = self.y_counter.count
        assert nx == ny, "Length mismatch between x and y"
        count = nx

        if nx == 0:
            return 0

        sum_x = self.x_counter.sum()
        mean_x = self.x_counter.mean()
        sum_y = self.y_counter.sum()
        mean_y = self.y_counter.mean()
        sum_xy = self.xy_counter.sum()
        sum_x2 = self.x2_counter.sum()
        sum_y2 = self.y2_counter.sum()

        covariance = sum_xy - mean_x * sum_y - mean_y * sum_x + count * mean_x * mean_y
        variance_x = sum_x2 - 2 * mean_x * sum_x + count * mean_x**2
        variance_y = sum_y2 - 2 * mean_y * sum_y + count * mean_y**2

        # Pearson correlation
        correlation = covariance / (
            math.sqrt(variance_x * variance_y) + 1e-8
        )  # Adding small value for numerical stability
        return correlation
