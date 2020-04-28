"""
get_spatial_meshgrid and warp functions are extracted from
https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py#L139
"""
import torch


def get_spatial_meshgrid(x: torch.Tensor, scale=False):
    """
    Get grid which contains spatial coordinates at each pixel location

    Args:
        x: image of shape [batch_size, channels, height, width] for which
            we want to generate the spatial grid
    """
    batch_size, _, height, width = x.size()
    # Generate mesh grid
    xx = torch.arange(0, width).view(1, -1).repeat(height, 1)
    yy = torch.arange(0, height).view(-1, 1).repeat(1, width)
    xx = xx.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if x.is_cuda:
        grid = grid.cuda()
    if scale:
        grid[:, 0] = grid[:, 0] / width
        grid[:, 1] = grid[:, 1] / height
    return grid


def warp(x, flow, thresh=0.99999, mode="bilinear"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [batch_size, channels, height, width] (im2)
    flow: [batch_size, 2, height, width] flow

    """
    batch_size, _, height, width = x.size()
    grid = get_spatial_meshgrid(x)
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(width - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(height - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, mode=mode)
    mask = x.new_ones(x.size())
    mask = torch.nn.functional.grid_sample(mask, vgrid, mode=mode)

    mask[mask < thresh] = 0
    mask[mask > 0] = 1

    return output * mask, mask


def pair_consist(
    recons_flow: torch.Tensor,
    image_ref: torch.Tensor,
    image: torch.Tensor,
    jitter_mask_ref: torch.Tensor,
    jitter_mask: torch.Tensor,
    criterion,
    use_backward: bool = False,
):
    """
    widthe use the optical flow estimated at end frame which contains sampling offsets
    from start to end frame to warp the start image to the end one

    Args:
        recons_flow: Optical flow estimated from a pair of meshes at two different
            time steps
        jitter_mask: Locations in the image that were outside of original image
            before data augmentation (which result in black paddings on the
            augmented image)
        image_ref: Image of reference (annotated) frame
        image: Image of unannotated frame
    """
    warp1, warp_mask1 = warp(image_ref.cuda(), recons_flow[1].permute(0, 3, 1, 2))
    # Warp jitter mask to know where to ignore it
    warpjitter1, _ = warp(jitter_mask_ref.cuda(), recons_flow[0].permute(0, 3, 1, 2))

    warp2, warp_mask2 = warp(image.cuda(), recons_flow[0].permute(0, 3, 1, 2))
    warpjitter2, _ = warp(jitter_mask.cuda(), recons_flow[1].permute(0, 3, 1, 2))
    # Ignore pixels that come from jittering padding
    warp_mask1 = warp_mask1 * (warpjitter2 == 1).float()
    warp_mask2 = warp_mask2 * (warpjitter1 == 1).float()
    warps = [warp1, warp2]

    # Compute masks
    masks = []
    flow_mask1 = ~(recons_flow[1] == 0)

    # Valid masks are computed by taking into account projection in pixel space
    # data augmentation and forward-backward flow consistency check
    valid_mask1 = warp_mask1[:, 0].bool() & flow_mask1[:, :, :, 0] & (jitter_mask[:, 0] == 1)
    masks.append({"warp_mask": warp_mask1, "full_mask": valid_mask1, "flow_mask": flow_mask1})
    flow_mask2 = ~(recons_flow[0] == 0)
    valid_mask2 = warp_mask2[:, 0].bool() & flow_mask2[:, :, :, 0] & (jitter_mask_ref[:, 0] == 1)
    masks.append({"warp_mask": warp_mask2, "full_mask": valid_mask2, "flow_mask": flow_mask2})
    _, _, losses_fwd, diffs_fwd, _ = criterion.compute(
        warp1, image, mask=valid_mask1.unsqueeze(1).repeat(1, 3, 1, 1)
    )
    _, _, losses_bwd, diffs_bwd, _ = criterion.compute(
        warp2, image_ref, mask=valid_mask2.unsqueeze(1).repeat(1, 3, 1, 1)
    )
    diffs_fwd = diffs_fwd[0][:, :3]
    diffs_bwd = diffs_bwd[0][:, :3]
    diffs = [diffs_fwd, diffs_bwd]
    if use_backward:
        warp_loss = losses_bwd + losses_fwd
    else:
        warp_loss = losses_fwd
    return warp_loss, masks, warps, diffs


def get_occlusion_mask(mask_flow1, mask_flow2, flow12, flow21):
    """
    Perform forward-backward conssitency check by warping a grid which
    contains the pixel locations at each grid position, and warping it
    from frame1 to frame2 and then back using the optical flow.
    If the values resulting from the two warps are not too different
    from the ones originally stored it means the pixel was consistently
    moved between in the two directions, which does not happen when a
    pixel is occluded in one of the two views.
    """
    grid1 = get_spatial_meshgrid(mask_flow1, scale=True)
    grid1 = torch.cat([grid1, mask_flow1, mask_flow1], 1)
    grid2 = get_spatial_meshgrid(mask_flow2, scale=True)
    grid2 = torch.cat([grid2, mask_flow2, mask_flow2], 1)
    warp_grid12, _ = warp(grid1.float(), flow21[:, :2], mode="nearest")
    warp_grid21, _ = warp(grid2.float(), flow12[:, :2], mode="nearest")
    warp_grid12 = warp_grid12 * mask_flow2[:, :1]
    warp_grid21 = warp_grid21 * mask_flow1[:, :1]
    warp_grid212, _ = warp(warp_grid12, flow12[:, :2], mode="nearest")
    warp_grid121, _ = warp(warp_grid21, flow21[:, :2], mode="nearest")
    warp_grid212 = warp_grid212 * mask_flow1[:, :1]
    warp_grid121 = warp_grid121 * mask_flow2[:, :1]
    warp_grid121 = warp_grid121.permute(0, 2, 3, 1)
    warp_grid212 = warp_grid212.permute(0, 2, 3, 1)
    grid2 = grid2.permute(0, 2, 3, 1)
    grid1 = grid1.permute(0, 2, 3, 1)
    occl_mask1 = occlusion_mask_from_warped_grid(grid1, warp_grid212)
    occl_mask2 = occlusion_mask_from_warped_grid(grid2, warp_grid121)
    return occl_mask1, occl_mask2


def occlusion_mask_from_warped_grid(
    grid: torch.Tensor, warped_grid: torch.Tensor, distance_thresh=0.03
) -> torch.Tensor:
    """
    Args:
        grid: Tensor of shape (batch_size, width, height, 3) where the
            last dim contains spatial dims locations (x and y coordinates
            of each location on the image grid) in first 2 channels and then
            the mask in the last channel
        warped_grid: Tensor of shape (batch_size, width, height, 3) which contains the output
            of two consecutive warps applied to the grid, using the forward and then backward
            optical flows for computing the warps


    Returns:
        valid_mask: locations which pass the forward-backward consistency check
    """
    mask = grid[:, :, :, 2] * warped_grid[:, :, :, 2]
    # If grid values have moved by more than distance_thresh, discard them from the
    # valid locations mask
    grid_displs = ((warped_grid - grid) * mask.unsqueeze(-1))[:, :, :, :2].norm(2, -1)
    motion_mask = (grid_displs < distance_thresh).float()
    valid_mask = mask * motion_mask
    return valid_mask
