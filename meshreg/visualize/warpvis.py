from matplotlib import pyplot as plt
import numpy as np
import torch

from meshreg.visualize import consistdisplay
from meshreg.datasets.queries import TransQueries
from meshreg.neurender import fastrender
from meshreg.visualize import samplevis, visutils


def sample_vis(batch, results, pair_results, save_img_prefix, fig=None, max_rows=4):
    fig.clf()
    samples = batch["data"]
    sample_nb = len(samples)
    if sample_nb == 1:
        save_img_path = f"{save_img_prefix}_single.png"
        samplevis.sample_vis(samples[0], results[0], save_img_path, fig=fig, max_rows=max_rows)
    if pair_results is not None:
        save_img_path = f"{save_img_prefix}_pair.png"
        images = [sample[TransQueries.IMAGE].cpu().permute(0, 2, 3, 1) + 0.5 for sample in samples]
        sample_nb = len(pair_results)
        if sample_nb >= 4:
            max_rows = int(np.ceil(max_rows / 4))
        elif sample_nb >= 2:
            max_rows = int(np.ceil(max_rows / 2))
        recons_flows1 = [pair_result[0].cpu().detach() for pair_result in pair_results["recons_flows"]]
        recons_flows2 = [pair_result[1].cpu().detach() for pair_result in pair_results["recons_flows"]]
        all_renders = []
        all_rot_renders = []
        for sample, result in zip(samples, results):
            render_results = fastrender.hand_obj_render(sample, result, modes=("all"), rotate=True)
            renders = render_results["all"].cpu()
            all_renders.append(renders)
            rot_renders = render_results["all_rotated"].cpu()
            all_rot_renders.append(rot_renders)
        warps1 = [
            pair_result[0].cpu().detach().permute(0, 2, 3, 1) + 0.5 for pair_result in pair_results["warps"]
        ]
        warps2 = [
            pair_result[1].cpu().detach().permute(0, 2, 3, 1) + 0.5 for pair_result in pair_results["warps"]
        ]
        diffs1 = [
            pair_result[0].cpu().detach().permute(0, 2, 3, 1) + 0.5 for pair_result in pair_results["diffs"]
        ]
        diffs2 = [
            pair_result[1].cpu().detach().permute(0, 2, 3, 1) + 0.5 for pair_result in pair_results["diffs"]
        ]
        masks1 = [pair_result[0]["full_mask"].cpu().detach() for pair_result in pair_results["masks"]]
        masks2 = [pair_result[1]["full_mask"].cpu().detach() for pair_result in pair_results["masks"]]
        # Prepare figure
        batch_size = images[0].shape[0]
        col_nb = 6
        row_nb = min(batch_size, max_rows)
        if fig is None:
            fig, axes = plt.subplots(row_nb, col_nb)
        else:
            axes = fig.subplots(row_nb, col_nb)
        for row_idx in range(row_nb):
            if row_nb == 1:
                row_ax = axes
            else:
                row_ax = axes[row_idx]
            for col_idx in range(col_nb):
                row_ax[col_idx].axis("off")
            # Col 0
            col_idx = 0
            img_s = images[0]
            all_imgs = visutils.colvis_pairef(images[1:], img_s, row_idx)
            row_ax[col_idx].imshow(all_imgs)

            # Col 1
            col_idx = 1
            show_rec_flows = visutils.colvis_flowpairs(recons_flows1, recons_flows2, row_idx)
            row_ax[col_idx].imshow(show_rec_flows)

            show_masks = visutils.colvis_pairs(masks2, masks1, row_idx)
            # Col 2
            col_idx = 2
            show_warps = visutils.colvis_pairs(warps2, warps1, row_idx)
            show_warps = torch.cat([show_warps, show_masks.unsqueeze(-1).float()], -1)
            row_ax[col_idx].imshow(all_imgs)
            row_ax[col_idx].imshow(torch.zeros_like(show_warps[:, :, :3]), alpha=0.5)
            row_ax[col_idx].imshow(show_warps)

            # Col 3
            col_idx = 3
            show_diffs = visutils.colvis_pairs(diffs2, diffs1, row_idx)
            combined = torch.cat([show_diffs, 0.2 + (0.8) * show_masks.float().unsqueeze(-1)], -1)
            row_ax[col_idx].imshow(combined)

            # Col 4
            col_idx = 4
            show_renders = visutils.colvis_pairef(all_renders[1:], all_renders[0], row_idx)
            row_ax[col_idx].imshow(all_imgs)
            row_ax[col_idx].imshow(show_renders)

            # Col 5
            col_idx = 5
            show_rot_renders = visutils.colvis_pairef(all_rot_renders[1:], all_rot_renders[0], row_idx)
            row_ax[col_idx].imshow(show_rot_renders)

        consistdisplay.squashfig(fig)
        fig.savefig(save_img_path, dpi=300)
