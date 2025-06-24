import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def MLC_soft_margin(outs, labels):
    return torch.nn.functional.multilabel_soft_margin_loss(outs, labels)

def cross_entropy(outs, labels):
    return torch.nn.functional.cross_entropy(outs, labels)

def calc_downstream_loss(outs, labels, dataset, label_mixup_fn=None, smoothing=0.):
    if 'BigEarthNet' in dataset:
        return MLC_soft_margin(outs, labels)
    elif label_mixup_fn is not None:
        # smoothing is handled with mixup label transform
        return SoftTargetCrossEntropy()(outs, labels)
    elif smoothing > 0.:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)(outs, labels)
    else:
        return cross_entropy(outs, labels)