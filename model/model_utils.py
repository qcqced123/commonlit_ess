import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def check_nan(x: torch.Tensor) -> bool:
    """ Check if there is NaN in tensor """
    checker = False
    if True in torch.isnan(x):
        checker = True
    return checker


def zero_filtering(x: torch.Tensor) -> torch.Tensor:
    """
    Add eps value for zero embedding, because competition metric is cosine similarity
    Cosine Similarity will be returned NaN, when input value has zero, like as torch.clamp()
    """
    eps = 1e-4
    x[x <= eps] = eps
    return x


def pow_nan_filtering(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    init unique eps value for each operations in torch.amp.auto_cast
    Change eps value for NaN Embedding from torch.pow, division, angular loss ...etc
    Args:
        x: tensor object, which is contained whole tensor elements
        eps: epsilon value for NaN Embedding, default is 1e-2
    """
    return torch.nan_to_num(x, nan=eps)


def matmul_nan_filtering(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    init unique eps value for each operations in torch.amp.auto_cast
    Change eps value for NaN Embedding from torch.pow, division, angular loss ...etc
    Args:
        x: tensor object, which is contained whole tensor elements
        eps: epsilon value for NaN Embedding, default is 1e-4
    """
    return torch.nan_to_num(x, nan=eps)


def float16_nan_filtering(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    init unique eps value for each operations in torch.amp.auto_cast
    Change eps value for NaN Embedding from torch.pow, division, angular loss ...etc
    Notes:
        add, matmul, conv, GRU, Linear, LSTM, RNN, mm
    Args:
        x: tensor object, which is contained whole tensor elements
        eps: epsilon value for NaN Embedding, default is 1e-15
    """
    return torch.nan_to_num(x, nan=eps)


def float32_nan_filtering(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """
    init unique eps value for each operations in torch.amp.auto_cast
    Change eps value for NaN Embedding from torch.pow, division, angular loss ...etc
    Notes:
        division, angular, cross entropy, exp, l1_loss, log_softmax, margin_ranking_loss (loss)
        smooth_l1_loss, sum, binary_cross_entorpy_with_logits, pow, log, nll_loss, layer_norm
    Args:
        x: tensor object, which is contained whole tensor elements
        eps: epsilon value for NaN Embedding, default is 1e-31
    """
    return torch.nan_to_num(x, nan=eps)


def freeze(module) -> None:
    """
    Freezes module's parameters.

    [Example]
    freezing embeddings and first 2 layers of encoder
    1) freeze(model.embeddings
    2) freeze(model.encoder.layer[:2])
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freeze_parameters(module) -> list[Tensor]:
    """
    Returns names of freezed parameters of the given module.

    [Example]
    freezed_parameters = get_freezed_parameters(model)
    """
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)

    return freezed_parameters


def init_weights(auto_cfg, module) -> None:
    """
    Initializes weights of the given module.
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=auto_cfg.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=auto_cfg.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def reinit_topk(model, num_layers):
    """
    Re-initialize the last-k transformer Encoder layers.
    Encoder Layer: Embedding, Attention Head, LayerNorm, Feed Forward
    Args:
        model: The target transformer model.
        num_layers: The number of layers to be re-initialized.
    """
    if num_layers > 0:
        model.encoder.layer[-num_layers:].apply(model._init_weights)
