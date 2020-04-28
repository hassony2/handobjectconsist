import torch

from kornia import losses as klosses
from kornia.geometry.transform import ScalePyramid
from meshreg.optim import lossutils


class PyramidCriterion:
    def __init__(self, criterion, geom_weight=1, level_nb=1):
        self.level_nb = level_nb
        self.SP = ScalePyramid()
        if criterion == "l2":
            self.criterion = torch.nn.MSELoss(reduction="none")
        elif criterion == "l1":
            self.criterion = torch.nn.L1Loss(reduction="none")
        elif criterion == "ssim":
            self.criterion = klosses.SSIM(reduction="none")
        else:
            raise ValueError(f"{criterion} not in [l2, l1, ssim]")
        self.geom_weight = geom_weight

    def compute(self, inp, target, mask=None):
        if self.level_nb > 1:
            inps, _, _ = self.SP(inp)
            targets, _, _ = self.SP(target)
            inps = [
                inp.view(inp.shape[0], inp.shape[1] * inp.shape[2], inp.shape[3], inp.shape[4])
                for inp in inps
            ]
            targets = [
                target.view(
                    target.shape[0], target.shape[1] * target.shape[2], target.shape[3], target.shape[4]
                )
                for target in targets
            ]
            if mask is not None:
                mask = mask.float()
                masks, _, _ = self.SP(mask)
                masks = [
                    mask.view(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3], mask.shape[4])
                    for mask in masks
                ]
                diffs = [
                    self.criterion(mask * inp + (1 - mask) * target, target)
                    for mask, inp, target in zip(masks, inps, targets)
                ]
            else:
                diffs = [self.criterion(inp, tar) for inp, tar in zip(inps, targets)]
                masks = None
            losses = [diff.mean() for diff in diffs]
            if self.geom_weight != 1:
                weights = [self.geom_weight ** idx for idx in range(len(losses))]
                weights.reverse()
                losses = [loss * weight for loss, weight in zip(losses, weights)]
            losses = torch.stack(losses)
        else:
            diff = self.criterion(inp, target)
            losses = lossutils.batch_masked_mean_loss(diff, mask)
            diffs = [diff]
            masks = [mask]
            inps = [inp]
            targets = [target]
        return inps, targets, losses, diffs, masks
