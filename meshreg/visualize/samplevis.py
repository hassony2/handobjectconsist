import torch

from libyana.visutils.viz2d import visualize_joints_2d

from meshreg.datasets.queries import BaseQueries, TransQueries
from meshreg.visualize import consistdisplay


def get_check_none(data, key, cpu=True):
    if key in data and data[key] is not None:
        if cpu:
            return data[key].cpu().detach()
        else:
            return data[key].detach().cuda()
    else:
        return None


def sample_vis(sample, results, save_img_path, fig=None, max_rows=5, display_centered=False):
    fig.clf()
    images = sample[TransQueries.IMAGE].permute(0, 2, 3, 1).cpu() + 0.5
    batch_size = images.shape[0]
    # pred_handverts2d = get_check_none(results, "verts2d")
    gt_objverts2d = get_check_none(sample, TransQueries.OBJVERTS2D)
    pred_objverts2d = get_check_none(results, "obj_verts2d")
    gt_objcorners2d = get_check_none(sample, TransQueries.OBJCORNERS2D)
    pred_objcorners2d = get_check_none(results, "obj_corners2d")
    gt_objcorners3dw = get_check_none(sample, BaseQueries.OBJCORNERS3D)
    pred_objcorners3d = get_check_none(results, "obj_corners3d")
    gt_objverts3d = get_check_none(sample, TransQueries.OBJVERTS3D)
    gt_canobjverts3d = get_check_none(sample, TransQueries.OBJCANROTVERTS)
    gt_canobjcorners3d = get_check_none(sample, TransQueries.OBJCANROTCORNERS)
    pred_objverts3d = get_check_none(results, "obj_verts3d")
    gt_handjoints2d = get_check_none(sample, TransQueries.JOINTS2D)
    pred_handjoints2d = get_check_none(results, "joints2d")
    gt_handjoints3d = get_check_none(sample, TransQueries.JOINTS3D)
    pred_handjoints3d = get_check_none(results, "joints3d")
    gt_handverts3d = get_check_none(sample, TransQueries.HANDVERTS3D)
    gt_objverts3dw = get_check_none(sample, BaseQueries.OBJVERTS3D)
    pred_handjoints3dw = get_check_none(results, "recov_joints3d")
    gt_handjoints3dw = get_check_none(sample, BaseQueries.JOINTS3D)
    pred_objverts3dw = get_check_none(results, "recov_objverts3d")
    pred_objcorners3dw = get_check_none(results, "recov_objcorners3d")
    pred_handverts3d = get_check_none(results, "verts3d")
    row_nb = min(max_rows, batch_size)
    if display_centered:
        col_nb = 7
    else:
        col_nb = 4
    axes = fig.subplots(row_nb, col_nb)
    for row_idx in range(row_nb):
        # Column 0
        axes[row_idx, 0].imshow(images[row_idx])
        axes[row_idx, 0].axis("off")
        if pred_handjoints2d is not None:
            visualize_joints_2d(axes[row_idx, 0], pred_handjoints2d[row_idx], alpha=1, joint_idxs=False)
        if gt_handjoints2d is not None:
            visualize_joints_2d(axes[row_idx, 0], gt_handjoints2d[row_idx], alpha=0.5, joint_idxs=False)

        # Column 1
        axes[row_idx, 1].imshow(images[row_idx])
        axes[row_idx, 1].axis("off")
        if pred_objverts2d is not None:
            axes[row_idx, 1].scatter(
                pred_objverts2d[row_idx, :, 0], pred_objverts2d[row_idx, :, 1], c="r", s=1, alpha=0.2
            )
        if gt_objverts2d is not None:
            axes[row_idx, 1].scatter(
                gt_objverts2d[row_idx, :, 0], gt_objverts2d[row_idx, :, 1], c="b", s=1, alpha=0.02
            )
        if pred_objcorners2d is not None:
            visualize_joints_2d(
                axes[row_idx, 1],
                pred_objcorners2d[row_idx],
                alpha=1,
                joint_idxs=False,
                links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
            )
        if gt_objcorners2d is not None:
            visualize_joints_2d(
                axes[row_idx, 1],
                gt_objcorners2d[row_idx],
                alpha=0.5,
                joint_idxs=False,
                links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
            )
        if gt_objverts2d is not None and pred_objverts2d is not None:
            idxs = list(range(6))
            arrow_nb = len(idxs)
            arrows = torch.cat([gt_objverts2d[:, idxs].float(), pred_objverts2d[:, idxs].float()], 1)
            links = [[i, i + arrow_nb] for i in range(arrow_nb)]
            visualize_joints_2d(
                axes[row_idx, 1],
                arrows[row_idx],
                alpha=0.5,
                joint_idxs=False,
                links=links,
                color=["k"] * arrow_nb,
            )
        # Column 2
        col_idx = 2
        if gt_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                gt_objverts3dw[row_idx, :, 0], gt_objverts3dw[row_idx, :, 1], c="b", s=1, alpha=0.02
            )
        if pred_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                pred_objverts3dw[row_idx, :, 0], pred_objverts3dw[row_idx, :, 1], c="r", s=1, alpha=0.02
            )
        if pred_handjoints3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx], pred_handjoints3dw[row_idx][:], alpha=1, joint_idxs=False
            )
        if gt_handjoints3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx], gt_handjoints3dw[row_idx][:], alpha=0.5, joint_idxs=False
            )
        axes[row_idx, col_idx].invert_yaxis()

        if pred_objcorners3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx],
                pred_objcorners3dw[row_idx],
                alpha=1,
                joint_idxs=False,
                links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
            )
        if gt_objcorners3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx],
                gt_objcorners3dw[row_idx],
                alpha=0.5,
                joint_idxs=False,
                links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
            )
        if pred_objverts3dw is not None and gt_objverts3dw is not None:
            arrow_nb = 6
            arrows = torch.cat([gt_objverts3dw[:, :arrow_nb], pred_objverts3dw[:, :arrow_nb]], 1)
            links = [[i, i + arrow_nb] for i in range(arrow_nb)]
            visualize_joints_2d(
                axes[row_idx, col_idx],
                arrows[row_idx],
                alpha=0.5,
                joint_idxs=False,
                links=links,
                color=["k"] * arrow_nb,
            )
        # Column 3
        col_idx = 3
        if gt_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                gt_objverts3dw[row_idx, :, 1], gt_objverts3dw[row_idx, :, 2], c="b", s=1, alpha=0.02
            )
        if pred_objverts3dw is not None:
            axes[row_idx, col_idx].scatter(
                pred_objverts3dw[row_idx, :, 1], pred_objverts3dw[row_idx, :, 2], c="r", s=1, alpha=0.02
            )
        if pred_handjoints3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx], pred_handjoints3dw[row_idx][:, 1:], alpha=1, joint_idxs=False
            )
        if gt_handjoints3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx], gt_handjoints3dw[row_idx][:, 1:], alpha=0.5, joint_idxs=False
            )
        if pred_objcorners3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx],
                pred_objcorners3dw[row_idx, :, 1:],
                alpha=1,
                joint_idxs=False,
                links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
            )
        if gt_objcorners3dw is not None:
            visualize_joints_2d(
                axes[row_idx, col_idx],
                gt_objcorners3dw[row_idx, :, 1:],
                alpha=0.5,
                joint_idxs=False,
                links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
            )
        if pred_objverts3dw is not None and gt_objverts3dw is not None:
            arrow_nb = 6
            arrows = torch.cat([gt_objverts3dw[:, :arrow_nb, 1:], pred_objverts3dw[:, :arrow_nb, 1:]], 1)
            links = [[i, i + arrow_nb] for i in range(arrow_nb)]
            visualize_joints_2d(
                axes[row_idx, col_idx],
                arrows[row_idx],
                alpha=0.5,
                joint_idxs=False,
                links=links,
                color=["k"] * arrow_nb,
            )

        if display_centered:
            # Column 4
            col_idx = 4
            if gt_canobjverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_canobjverts3d[row_idx, :, 0], gt_canobjverts3d[row_idx, :, 1], c="b", s=1, alpha=0.02
                )
            if pred_objverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    pred_objverts3d[row_idx, :, 0], pred_objverts3d[row_idx, :, 1], c="r", s=1, alpha=0.02
                )
            if pred_objcorners3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx],
                    pred_objcorners3d[row_idx],
                    alpha=1,
                    joint_idxs=False,
                    links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
                )
            if gt_canobjcorners3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx],
                    gt_canobjcorners3d[row_idx],
                    alpha=0.5,
                    joint_idxs=False,
                    links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]],
                )
            if pred_objcorners3d is not None and gt_canobjcorners3d is not None:
                arrow_nb = 6
                arrows = torch.cat([gt_canobjcorners3d[:, :arrow_nb], pred_objcorners3d[:, :arrow_nb]], 1)
                links = [[i, i + arrow_nb] for i in range(arrow_nb)]
                visualize_joints_2d(
                    axes[row_idx, col_idx],
                    arrows[row_idx],
                    alpha=0.5,
                    joint_idxs=False,
                    links=links,
                    color=["k"] * arrow_nb,
                )
            axes[row_idx, col_idx].set_aspect("equal")
            axes[row_idx, col_idx].invert_yaxis()

            # Column 5
            col_idx = 5
            if gt_objverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_objverts3d[row_idx, :, 0], gt_objverts3d[row_idx, :, 1], c="b", s=1, alpha=0.02
                )
            # if pred_objverts3d is not None:
            #     axes[row_idx, 2].scatter(
            #         pred_objverts3d[row_idx, :, 0], pred_objverts3d[row_idx, :, 1], c="r", s=1, alpha=0.02
            #     )
            if gt_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_handverts3d[row_idx, :, 0], gt_handverts3d[row_idx, :, 1], c="g", s=1, alpha=0.2
                )
            if pred_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    pred_handverts3d[row_idx, :, 0], pred_handverts3d[row_idx, :, 1], c="c", s=1, alpha=0.2
                )
            if pred_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], pred_handjoints3d[row_idx], alpha=1, joint_idxs=False
                )
            if gt_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], gt_handjoints3d[row_idx], alpha=0.5, joint_idxs=False
                )
            axes[row_idx, col_idx].invert_yaxis()

            # Column 6
            col_idx = 6
            if gt_objverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_objverts3d[row_idx, :, 1], gt_objverts3d[row_idx, :, 2], c="b", s=1, alpha=0.02
                )
            # if pred_objverts3d is not None:
            #     axes[row_idx, 3].scatter(
            #         pred_objverts3d[row_idx, :, 1], pred_objverts3d[row_idx, :, 2], c="r", s=1, alpha=0.02
            #     )
            if gt_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    gt_handverts3d[row_idx, :, 1], gt_handverts3d[row_idx, :, 2], c="g", s=1, alpha=0.2
                )
            if pred_handverts3d is not None:
                axes[row_idx, col_idx].scatter(
                    pred_handverts3d[row_idx, :, 1], pred_handverts3d[row_idx, :, 2], c="c", s=1, alpha=0.2
                )
            if pred_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], pred_handjoints3d[row_idx][:, 1:], alpha=1, joint_idxs=False
                )
            if gt_handjoints3d is not None:
                visualize_joints_2d(
                    axes[row_idx, col_idx], gt_handjoints3d[row_idx][:, 1:], alpha=0.5, joint_idxs=False
                )

    consistdisplay.squashfig(fig)
    fig.savefig(save_img_path, dpi=300)
