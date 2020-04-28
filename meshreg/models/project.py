import torch
from meshreg.datasets.queries import TransQueries


def recover_3d_proj(objpoints3d, camintr, est_scale, est_trans, off_z=0.4, input_res=(128, 128)):
    """
    Given estimated centered points, camera intrinsics and predicted scale and translation
    in pixel world, compute the point coordinates in camera coordinate system
    """
    # Estimate scale and trans between 3D and 2D
    focal = camintr[:, :1, :1]
    batch_size = objpoints3d.shape[0]
    focal = focal.view(batch_size, 1)
    est_scale = est_scale.view(batch_size, 1)
    est_trans = est_trans.view(batch_size, 2)
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2]
    img_centers = (cam_centers.new(input_res) / 2).view(1, 2).repeat(batch_size, 1)
    est_XY0 = (est_trans + img_centers - cam_centers) * est_Z0 / focal
    est_c3d = torch.cat([est_XY0, est_Z0], -1).unsqueeze(1)
    recons3d = est_c3d + objpoints3d
    return recons3d, est_c3d
