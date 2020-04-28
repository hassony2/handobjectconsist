from typing import List

import torch

from libyana.camutils import project
from libyana.renderutils import textutils
from meshreg.warping import imgflowarp


def get_opticalflows(
    verts_cam: List[torch.Tensor],
    faces: torch.Tensor,
    camintrs: List[torch.Tensor],
    neurenderer,
    orig_img_size=None,
    detach_textures: bool = False,
    detach_renders: bool = False,
    ignore_face_idxs=None,
):
    """
    Compute optical flow between pairs of meshes (representing the same mesh at
    two different time steps), always comparing to the first mesh.
    (see get_opticalflow for details of optical flow computations between the pairs)

    Args:
        verts_cam : List of vertices representing the vertex positions
            in camera coordinates for the same meshes at different time steps
        faces : Faces of the meshes for which the optical flow is computed
        camintrs : Intrinsic camera parameters for each time step
        neurenderer: Neural renderer instance
    """
    all_flows = []
    # Cycle through vertex positions and camera parameters, skipping the first
    # sample
    for vert_world, camintr in zip(verts_cam[1:], camintrs[1:]):
        # Compute optical flow between current sample and first sample
        flows = get_opticalflow(
            [verts_cam[0], vert_world],
            faces,
            [camintrs[0], camintr],
            neurenderer,
            orig_img_size=orig_img_size,
            detach_textures=detach_textures,
            detach_renders=detach_renders,
            ignore_face_idxs=ignore_face_idxs,
        )
        all_flows.append(flows)
    return all_flows


def get_opticalflow(
    verts_cam: List[torch.Tensor],
    faces: torch.Tensor,
    camintrs: List[torch.Tensor],
    neurenderer,
    orig_img_size=None,
    mask_occlusions: bool = True,
    detach_textures: bool = False,
    detach_renders: bool = True,
    ignore_face_idxs=None,
):
    """
    Compute optical flow in image space given the displacement of the vertices
    in verts_cam between the first

    If detach_renders is False, gradients will be computed to update the 'shape'
    of the rendered flow image
    If detach_textures is False, gradients will be computed to update the 'colors'
    of the rendered flow image

    When detach_renders is True and detach_textures is False (the setting we use),
    the gradients only flow through the difference (as rendered by the neural renderer)
    between the positions between the pairs, (no gradients flow to update the position
    of the first mesh)

    Args:
        verts_cam: Pair of vertex positions as list of tensors of shape
            ([batch_size, point_nb, 3 (spatial coordinate)]) of len 2
        faces: Faces as tensor of vertex indexes of shape
            [batch_size, face_nb, 3 (vertex indices)]
        camintrs: Pair of intrinsic camera parameters as list of tensors
            of shape (batch_size, 3, 3) of len 2
        ignore_face_idxs: Idxs of faces for which the optical flow should
            not be computed
        detach_textures: Do not backpropagate through the optical flow offset
            *values*
        detach_renders: Do not backpropagate through the optical flow rendered
            positions

    Returns:
        pred_flow12: flow values renderered at the location of first vertices
            with flow values from 1 to 2
        pred_flow12: flow values renderered at the location of second vertices
            with flow values from 2 to 1
    """

    # Project ground truth vertices on image plane
    gt_locs2d_1 = project.batch_proj2d(verts_cam[0], camintrs[0])
    gt_locs2d_2 = project.batch_proj2d(verts_cam[1], camintrs[1])
    # Get ground truth forward optical flow
    verts_displ2d_12 = gt_locs2d_2 - gt_locs2d_1
    sample_flows = torch.cat([verts_displ2d_12, torch.ones_like(verts_displ2d_12[:, :, :1])], -1)
    all_textures = textutils.batch_vertex_textures(faces, sample_flows)
    if detach_textures:
        all_textures = all_textures.detach()

    # Only keep locations with valid flow pixel predictions
    renderout = neurenderer(verts_cam[0], faces, all_textures, K=camintrs[0], detach_renders=detach_renders)
    mask_flow1 = (renderout["alpha"].unsqueeze(1) > 0.99999).float()
    if ignore_face_idxs is not None:
        ignore_mask = (
            renderout["face_index_map"].unsqueeze(-1) - renderout["face_index_map"].new(ignore_face_idxs)
        ).abs().min(-1)[0] != 0
        ignore_mask = ignore_mask[:, list(reversed(range(ignore_mask.shape[1])))]
        ignore_mask = ignore_mask.float().unsqueeze(1)
        mask_flow1 = mask_flow1 * ignore_mask

    pred_flow12 = renderout["rgb"] * mask_flow1

    # Get ground truth backward optical flow
    verts_displ2d_21 = gt_locs2d_1 - gt_locs2d_2
    sample_flows = torch.cat([verts_displ2d_21, torch.ones_like(verts_displ2d_21[:, :, :1])], -1)
    all_textures = textutils.batch_vertex_textures(faces, sample_flows)

    # Only keep locations with valid flow pixel predictions
    renderout = neurenderer(verts_cam[1], faces, all_textures, K=camintrs[1], detach_renders=detach_renders)
    mask_flow2 = (renderout["alpha"].unsqueeze(1) > 0.99999).float()
    if ignore_face_idxs is not None:
        ignore_mask = (
            renderout["face_index_map"].unsqueeze(-1) - renderout["face_index_map"].new(ignore_face_idxs)
        ).abs().min(-1)[0] != 0
        ignore_mask = ignore_mask[:, list(reversed(range(ignore_mask.shape[1])))]
        ignore_mask = ignore_mask.float().unsqueeze(1)
        mask_flow2 = mask_flow2 * ignore_mask
    pred_flow21 = renderout["rgb"] * mask_flow2

    if mask_occlusions:
        with torch.no_grad():
            mask_flow2 = renderout["alpha"].unsqueeze(1)
            # Compute pixels which are visible in both frames by
            # performing a forward-backward consistency check
            occl_mask1, occl_mask2 = imgflowarp.get_occlusion_mask(
                mask_flow1, mask_flow2, pred_flow12, pred_flow21
            )

        mask_flow1 = mask_flow1 * occl_mask1.unsqueeze(1)
        mask_flow2 = mask_flow2 * occl_mask2.unsqueeze(1)
        pred_flow12 = pred_flow12 * mask_flow1
        pred_flow21 = pred_flow21 * mask_flow2
    pred_flow12 = pred_flow12.permute(0, 2, 3, 1)[:, :, :, :2]
    pred_flow21 = pred_flow21.permute(0, 2, 3, 1)[:, :, :, :2]
    if orig_img_size is not None:
        pred_flow12 = pred_flow12[:, : orig_img_size[1], : orig_img_size[0]]
        pred_flow21 = pred_flow21[:, : orig_img_size[1], : orig_img_size[0]]
    pred_flows = [pred_flow12, pred_flow21]
    return pred_flows
