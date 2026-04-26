"""ViT-B/16 multi-label classifier for the G3 Corfu Butterflies benchmark.

The backbone is torchvision's ``vit_b_16`` with the ImageNet-1k
pre-trained weights. The original 1000-way classification head is
replaced with a dropout layer (``p = 0.1``) followed by a linear
projection to ``num_labels`` logits, suitable for multi-label
classification.

Author:
    Nikolaos Korfiatis, Ionian University. Contact: nkorf@ionio.gr

License:
    MIT.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_multilabel(
    num_labels: int,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> nn.Module:
    """Build a ViT-B/16 classifier with a multi-label output head.

    Parameters
    ----------
    num_labels:
        Dimensionality of the multi-hot target vector (45 for the
        shipped G3 vocabulary).
    pretrained:
        If ``True`` (default), load the torchvision
        ``IMAGENET1K_V1`` pre-trained weights. Set to ``False`` for
        from-scratch training experiments.
    dropout:
        Dropout probability applied between the pooled ViT feature
        and the final linear projection.

    Returns
    -------
    torch.nn.Module
        A ViT-B/16 whose ``heads.head`` is a
        ``nn.Sequential(Dropout, Linear(768, num_labels))``. The
        logits are trained with a binary cross-entropy objective by
        the reference training loop.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_labels),
    )
    return model


def param_groups(
    model: nn.Module, backbone_lr: float, head_lr: float
) -> list[dict]:
    """Split parameters into a two-group optimiser configuration.

    The backbone (transformer blocks and patch embedding) uses a
    small learning rate so that pre-trained representations are not
    disrupted; the freshly-initialised head uses a larger learning
    rate so that the output mapping converges quickly.
    """
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("heads.") else backbone_params).append(p)
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
