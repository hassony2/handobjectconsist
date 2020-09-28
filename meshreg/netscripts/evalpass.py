import os

import numpy as np
from tqdm import tqdm
import torch

from libyana.evalutils.avgmeter import AverageMeters
from libyana.evalutils.zimeval import EvalUtil

from meshreg.visualize import samplevis
from meshreg.netscripts import evaluate
from meshreg.datasets.queries import BaseQueries
from meshreg.datasets import ho3dv2utils


def get_order_idxs():
    reorder_idxs = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    unorder_idxs = np.argsort(reorder_idxs)
    return reorder_idxs, unorder_idxs


def epoch_pass(
    loader,
    model,
    optimizer=None,
    scheduler=None,
    epoch=0,
    img_folder=None,
    fig=None,
    display_freq=10,
    epoch_display_freq=1,
    lr_decay_gamma=0,
    freeze_batchnorm=True,
    dump_results_path=None,
    render_folder=None,
    render_freq=10,
    true_root=False,
):
    prefix = "val"
    reorder_idxs, unorder_idxs = get_order_idxs()
    evaluators = {
        # "joints2d_trans": EvalUtil(),
        "joints2d_base": EvalUtil(),
        "corners2d_base": EvalUtil(),
        "verts2d_base": EvalUtil(),
        "joints3d_cent": EvalUtil(),
        "joints3d": EvalUtil(),
    }
    model.eval()
    model.cuda()
    avg_meters = AverageMeters()
    all_joints = []
    all_verts = []
    for batch_idx, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            loss, results, losses = model(batch)
            # Collect hand joints
            if true_root:
                results["recov_joints3d"][:, 0] = batch[BaseQueries.JOINTS3D][:, 0]
            recov_joints = results["recov_joints3d"].cpu().detach()[:, unorder_idxs]
            recov_joints[:, :, 0] = -recov_joints[:, :, 0]
            new_joints = [-val.numpy()[0] for val in recov_joints.split(1)]
            all_joints.extend(new_joints)

            # Collect hand vertices
            recov_verts = results["recov_handverts3d"].cpu().detach()
            recov_verts[:, :, 0] = -recov_verts[:, :, 0]
            new_verts = [-val.numpy()[0] for val in recov_verts.split(1)]
            all_verts.extend(new_verts)

        evaluate.feed_avg_meters(avg_meters, batch, results)
        if batch_idx % display_freq == 0 and epoch % epoch_display_freq == 0:
            img_filepath = f"{prefix}_epoch{epoch:04d}_batch{batch_idx:06d}.png"
            save_img_path = os.path.join(img_folder, img_filepath)
            samplevis.sample_vis(batch, results, fig=fig, save_img_path=save_img_path)
        evaluate.feed_evaluators(evaluators, batch, results)
    ho3dv2utils.dump(dump_results_path, all_joints, all_verts, codalab=True)
