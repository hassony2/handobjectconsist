def batch_masked_mean_loss(dists, mask):
    mask = mask.float()
    batch_sum = (mask * dists).sum(dim=list(range(1, dists.dim())))
    batch_valid_vals = mask.sum(dim=list(range(1, dists.dim())))
    # Don't divide by 0
    batch_valid_vals[(batch_valid_vals == 0)] = 1
    batch_losses = batch_sum / batch_valid_vals
    return batch_losses
