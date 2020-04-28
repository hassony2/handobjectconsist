import torch

from libyana.renderutils import catmesh

from meshreg.datasets.queries import TransQueries, BaseQueries
from meshreg.warping import imgflowarp, opticalflow


def forward(
    samples,
    all_results,
    hand_face,
    renderer,
    image_size,
    criterion,
    gt_refs=True,
    first_only=True,
    hand_ignore_faces=None,
    use_backward=True,
):
    """
    Args:
        use_backward: Compare warp from unannotated to annotate frame with annotated image image
            (in addition to comparing the unanotted image with the warp from the annotated
            frame to the unnanotated one)
    """
    # Put inputs on GPU
    images = [sample[TransQueries.IMAGE].cuda() for sample in samples]
    jitter_masks = [sample[TransQueries.JITTERMASK].cuda() for sample in samples]
    camintrs = [sample[TransQueries.CAMINTR].cuda() for sample in samples]

    # Get visible vertices
    obj_verts = [result["recov_objverts3d"] for result in all_results]
    obj_faces = [sample[BaseQueries.OBJFACES].long().cuda() for sample in samples]
    hand_verts = [result["recov_handverts3d"] for result in all_results]
    hand_faces_b = hand_face.repeat(obj_verts[0].shape[0], 1, 1).long()
    hand_faces = [hand_faces_b for _ in range(len(samples))]
    if gt_refs:
        # Replace reference vertices by ground truth vertices
        # Comparisons are performed with the first sample in the obj_verts and
        # hand_verts lists. Hence, replacing the hand and object vertex positions
        # for the subsequent sample with ground truth locations allows to perform
        # comparisons considering that the annotated frame has ground truth
        # annotations available
        for sample_idx in range(1, len(samples)):
            obj_verts[sample_idx] = samples[sample_idx][BaseQueries.OBJVERTS3D].cuda()
            hand_verts[sample_idx] = samples[sample_idx][BaseQueries.HANDVERTS3D].cuda()
    verts_world = []
    for seq_idx in range(len(samples)):
        all_verts, all_faces, _ = catmesh.batch_cat_meshes(
            [hand_verts[seq_idx], obj_verts[seq_idx]], [hand_faces[seq_idx], obj_faces[seq_idx]]
        )
        if first_only and seq_idx > 0:
            all_verts = all_verts.detach()
        verts_world.append(all_verts)

    # Estimates pixel motions in image space from the estimated vertex motions of
    # the mesh models
    recons_flows = opticalflow.get_opticalflows(
        verts_world,
        all_faces,
        camintrs,
        renderer,
        image_size,
        detach_textures=False,
        detach_renders=True,
        ignore_face_idxs=hand_ignore_faces,
    )
    all_masks = []
    all_warps = []
    all_diffs = []
    full_losses = []
    for recons_flow, image, jitter_mask in zip(recons_flows, images[1:], jitter_masks[1:]):
        warp_loss, masks, warps, diffs = imgflowarp.pair_consist(
            recons_flow,
            image_ref=images[0],
            image=image,
            jitter_mask_ref=jitter_masks[0],
            jitter_mask=jitter_mask,
            criterion=criterion,
            use_backward=use_backward,
        )
        all_masks.append(masks)
        full_losses.append(warp_loss)
        all_warps.append(warps)
        all_diffs.append(diffs)
    stack_losses = torch.stack(full_losses)
    full_loss = stack_losses.mean()
    pair_results = {
        "masks": all_masks,
        "warps": all_warps,
        "recons_flows": recons_flows,
        "diffs": all_diffs,
        "diff_losses": stack_losses,
    }
    return full_loss, pair_results
