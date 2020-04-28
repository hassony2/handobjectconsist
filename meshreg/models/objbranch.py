from torch import nn

from manopth import rodrigues_layer

from meshreg.datasets.queries import BaseQueries, TransQueries
from meshreg.models import project
from libyana.camutils import project as camproject


class ObjBranch(nn.Module):
    def __init__(self, trans_factor=1, scale_factor=1):
        """
        Args:
            trans_factor: Scaling parameter to insure translation and scale
                are updated similarly during training (if one is updated 
                much more than the other, training is slowed down, because
                for instance only the variation of translation or scale
                significantly influences the final loss variation)
            scale_factor: Scaling parameter to insure translation and scale
                are updated similarly during training
        """
        super(ObjBranch, self).__init__()
        self.trans_factor = trans_factor
        self.scale_factor = scale_factor
        self.inp_res = [256, 256]

    def forward(self, sample, scaletrans=None, scale=None, trans=None, rotaxisang=None):
        """
        Args:
            scaletrans: torch.Tensor of shape [batch_size, channels] with channels == 6
                with in first position the predicted scale values and in 2,3 the 
                predicted translation values, and global rotation encoded as axis-angles
                in channel positions 4,5,6
        """
        if scaletrans is None:
            batch_size = scale.shape[0]
        else:
            batch_size = scaletrans.shape[0]
        if scale is None:
            scale = scaletrans[:, :1]
        if trans is None:
            trans = scaletrans[:, 1:3]
        if rotaxisang is None:
            rotaxisang = scaletrans[:, 3:]
        # Get rotation matrixes from axis-angles
        rotmat = rodrigues_layer.batch_rodrigues(rotaxisang).view(rotaxisang.shape[0], 3, 3)
        canobjverts = sample[BaseQueries.OBJCANVERTS].cuda()
        rotobjverts = rotmat.bmm(canobjverts.float().transpose(1, 2)).transpose(1, 2)

        final_trans = trans.unsqueeze(1) * self.trans_factor
        final_scale = scale.view(batch_size, 1, 1) * self.scale_factor
        height, width = tuple(sample[TransQueries.IMAGE].shape[2:])
        camintr = sample[TransQueries.CAMINTR].cuda()
        objverts3d, center3d = project.recover_3d_proj(
            rotobjverts, camintr, final_scale, final_trans, input_res=(width, height)
        )
        # Recover 2D positions given camera intrinsic parameters and object vertex
        # coordinates in camera coordinate reference
        pred_objverts2d = camproject.batch_proj2d(objverts3d, camintr)
        if BaseQueries.OBJCORNERS3D in sample:
            canobjcorners = sample[BaseQueries.OBJCANCORNERS].cuda()
            rotobjcorners = rotmat.bmm(canobjcorners.float().transpose(1, 2)).transpose(1, 2)
            recov_objcorners3d = rotobjcorners + center3d
            pred_objcorners2d = camproject.batch_proj2d(rotobjcorners + center3d, camintr)
        else:
            pred_objcorners2d = None
            recov_objcorners3d = None
            rotobjcorners = None
        return {
            "obj_verts2d": pred_objverts2d,
            "obj_verts3d": rotobjverts,
            "recov_objverts3d": objverts3d,
            "recov_objcorners3d": recov_objcorners3d,
            "obj_scale": final_scale,
            "obj_prescale": scale,
            "obj_prerot": rotaxisang,
            "obj_trans": final_trans,
            "obj_pretrans": trans,
            "obj_corners2d": pred_objcorners2d,
            "obj_corners3d": rotobjcorners,
        }
