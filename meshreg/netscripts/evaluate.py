import warnings

import torch

from meshreg.datasets.queries import TransQueries, BaseQueries


def recover_back(joints_trans, affinetrans):
    """
    Given 2d point coordinates and an affine transform, recovers original pixel points
    (locations before translation, rotation, crop, scaling... are applied during data
    augmentation)
    """
    batch_size = joints_trans.shape[0]
    point_nb = joints_trans.shape[1]
    hom2d = torch.cat([joints_trans, joints_trans.new_ones(batch_size, point_nb, 1)], -1)
    rec2d = torch.inverse(affinetrans).bmm(hom2d.transpose(1, 2).float()).transpose(1, 2)[:, :, :2]
    return rec2d


def recover_3d_proj(joints3d, joints2d, camintr, est_scale, est_trans, center_idx=9):
    # Estimate scale and trans between 3D and 2D
    trans3d = joints3d[:, center_idx : center_idx + 1]
    joints3d_c = joints3d - trans3d
    focal = camintr[:, :1, :1]
    est_Z0 = focal / est_scale
    est_XY0 = (est_trans[:, 0] - camintr[:, :2, 2]) * est_Z0[:, 0] / focal[:, 0]
    est_c3d = torch.cat([est_XY0, est_Z0[:, 0]], -1).unsqueeze(1)
    recons3d = est_c3d + joints3d_c
    return recons3d, est_c3d


def feed_avg_meters(avg_meters, sample, results):
    if "obj_verts2d" in results and results["obj_verts2d"] is not None and BaseQueries.OBJVERTS2D in sample:
        obj_verts2d_gt = sample[TransQueries.OBJVERTS2D]
        affinetrans = sample[TransQueries.AFFINETRANS]
        or_verts2d = sample[BaseQueries.OBJVERTS2D]
        rec_pred = recover_back(results["obj_verts2d"].detach().cpu(), affinetrans)
        rec_gt = recover_back(obj_verts2d_gt, affinetrans)
        # Sanity check, this should be ~1pixel
        gt_err = (rec_gt - or_verts2d).norm(2, -1).mean()
        if gt_err > 1:
            warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
        verts2d_dists = (rec_pred - or_verts2d.cpu()).norm(2, -1)
        avg_meters.add_loss_value("objverts2d_mepe", verts2d_dists.mean().item())
    if "obj_verts3d" in results and results["obj_verts3d"] is not None and BaseQueries.OBJVERTS3D in sample:
        height, width = tuple(sample[BaseQueries.IMAGE].shape[2:])
        or_verts3d = sample[BaseQueries.OBJVERTS3D].cpu()

        verts3d_diststrans = (results["recov_objverts3d"].cpu() - or_verts3d).norm(2, -1)
        avg_meters.add_loss_value("objverts3d_mepe_trans", verts3d_diststrans.mean().item())


def feed_evaluators(evaluators, sample, results, idxs=None, center_idx=9):
    if idxs is None:
        idxs = list(range(21))
    if "joints2d" in results and BaseQueries.JOINTS2D in sample:
        gt_joints2d = sample[TransQueries.JOINTS2D]
        affinetrans = sample[TransQueries.AFFINETRANS]
        or_joints2d = sample[BaseQueries.JOINTS2D]
        rec_pred = recover_back(results["joints2d"].detach().cpu(), affinetrans)
        rec_gt = recover_back(gt_joints2d, affinetrans)
        # Sanity check, this should be ~1pixel
        gt_err = (rec_gt - or_joints2d).norm(2, -1).mean()
        if gt_err > 1:
            warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
        for gt_joints, pred_joints in zip(rec_pred.numpy(), or_joints2d.cpu().numpy()):
            evaluators["joints2d_base"].feed(gt_joints, pred_joints)
    # Object 2d metric
    if (
        "obj_corners2d" in results
        and results["obj_corners2d"] is not None
        and BaseQueries.OBJCORNERS2D in sample
    ):
        obj_corners2d_gt = sample[TransQueries.OBJCORNERS2D]
        affinetrans = sample[TransQueries.AFFINETRANS]
        or_corners2d = sample[BaseQueries.OBJCORNERS2D]
        rec_pred = recover_back(results["obj_corners2d"].detach().cpu(), affinetrans)
        rec_gt = recover_back(obj_corners2d_gt, affinetrans)
        # Sanity check, this should be ~1pixel
        gt_err = (rec_gt - or_corners2d).norm(2, -1).mean()
        if gt_err > 1:
            warnings.warn(f"Back to orig error on gt {gt_err} > 1 pixel")
        for gt_corners, pred_corners in zip(rec_pred.numpy(), or_corners2d.cpu().numpy()):
            evaluators["corners2d_base"].feed(gt_corners, pred_corners)
    # Centered 3D hand metric
    if "joints3d" in results and BaseQueries.JOINTS3D in sample:
        gt_joints3d = sample[TransQueries.JOINTS3D]
        pred_joints3d = results["joints3d"].cpu().detach()
        if center_idx is not None:
            gt_joints3d_cent = gt_joints3d - gt_joints3d[:, center_idx : center_idx + 1]
            pred_joints3d_cent = pred_joints3d - pred_joints3d[:, center_idx : center_idx + 1]
            for gt_joints, pred_joints in zip(gt_joints3d_cent.numpy(), pred_joints3d_cent.numpy()):
                evaluators["joints3d_cent"].feed(gt_joints, pred_joints)
        if "recov_joints3d" in results:
            joints3d_pred = results["recov_joints3d"].detach().cpu()
            for gt_joints, pred_joints in zip(sample[BaseQueries.JOINTS3D].cpu(), joints3d_pred):
                evaluators["joints3d"].feed(gt_joints, pred_joints)


def parse_evaluators(evaluators, config=None):
    """
    Parse evaluators for which PCK curves and other statistics
    must be computed
    """
    if config is None:
        config = {
            # "joints2d_trans": [0, 50, 20],
            "joints2d_base": [0, 100, 100],
            "corners2d_base": [0, 100, 100],
            "verts2d_base": [0, 100, 100],
            "joints3d_cent": [0, 0.2, 20],
            "joints3d": [0, 0.5, 20],
        }
    eval_results = {}
    for evaluator_name, evaluator in evaluators.items():
        start, end, steps = [config[evaluator_name][idx] for idx in range(3)]
        (epe_mean, epe_mean_joints, epe_median, auc, pck_curve, thresholds) = evaluator.get_measures(
            start, end, steps
        )
        eval_results[evaluator_name] = {
            "epe_mean": epe_mean,
            "epe_mean_joints": epe_mean_joints,
            "epe_median": epe_median,
            "auc": auc,
            "thresholds": thresholds,
            "pck_curve": pck_curve,
        }
    return eval_results
