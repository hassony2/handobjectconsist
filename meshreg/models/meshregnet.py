import pickle
import torch
from torch import nn
import torch.nn.functional as torch_f

from libyana.camutils import project as camproject
from libyana.modelutils.freeze import rec_freeze

from meshreg.models import resnet
from meshreg.models.absolutebranch import AbsoluteBranch
from meshreg.models.manobranch import ManoBranch, ManoLoss
from meshreg.models.objbranch import ObjBranch
from meshreg.models import project
from meshreg.datasets.queries import TransQueries, BaseQueries, one_query_in


def normalize_pixel_out(data, inp_res=256):
    centered = data - inp_res
    scaled = centered / inp_res / 2
    return scaled.float()


class ManoAdaptor(torch.nn.Module):
    def __init__(self, mano_layer, load_path=None):
        super().__init__()
        self.adaptor = torch.nn.Linear(778, 21, bias=False)
        if load_path is not None:
            with open(load_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
                weights = exp_data["adaptor"]
            regressor = torch.Tensor(weights)
            self.register_buffer("J_regressor", regressor)
        else:
            regressor = mano_layer._buffers["th_J_regressor"]
            tip_reg = regressor.new_zeros(5, regressor.shape[1])
            tip_reg[0, 745] = 1
            tip_reg[1, 317] = 1
            tip_reg[2, 444] = 1
            tip_reg[3, 556] = 1
            tip_reg[4, 673] = 1
            reordered_reg = torch.cat([regressor, tip_reg])[
                [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            ]
            self.register_buffer("J_regressor", reordered_reg)
        self.adaptor.weight.data = self.J_regressor

    def forward(self, inp):
        fix_idxs = [0, 4, 8, 12, 16, 20]
        for idx in fix_idxs:
            self.adaptor.weight.data[idx] = self.J_regressor[idx]
        return self.adaptor(inp.transpose(2, 1)), self.adaptor.weight - self.J_regressor


class MeshRegNet(nn.Module):
    def __init__(
        self,
        fc_dropout=0,
        resnet_version=18,
        criterion2d="l2",
        mano_neurons=[512, 512],
        mano_comps=15,
        mano_use_shape=False,
        mano_lambda_pose_reg=0,
        mano_use_pca=True,
        mano_center_idx=9,
        mano_root="assets/mano",
        mano_lambda_joints3d=None,
        mano_lambda_recov_joints3d=None,
        mano_lambda_recov_verts3d=None,
        mano_lambda_joints2d=None,
        mano_lambda_verts3d=None,
        mano_lambda_verts2d=None,
        mano_lambda_shape=None,
        mano_pose_coeff: int = 1,
        mano_fhb_hand: bool = False,
        obj_lambda_verts3d=None,
        obj_lambda_verts2d=None,
        obj_lambda_recov_verts3d=None,
        obj_trans_factor=1,
        obj_scale_factor=1,
        inp_res=256,
    ):
        """
        Args:
            mano_fhb_hand: Use pre-computed mapping from MANO joints to First Person
            Hand Action Benchmark hand skeleton
            mano_root (path): dir containing mano pickle files
            mano_neurons: number of neurons in each layer of base mano decoder
            mano_use_pca: predict pca parameters directly instead of rotation
                angles
            mano_comps (int): number of principal components to use if
                mano_use_pca
            mano_lambda_pca: weight to supervise hand pose in PCA space
            mano_lambda_pose_reg: weight to supervise hand pose in axis-angle
                space
            mano_lambda_verts: weight to supervise vertex distances
            mano_lambda_joints3d: weight to supervise distances
            adapt_atlas_decoder: add layer between encoder and decoder, usefull
                when finetuning from separately pretrained encoder and decoder
        """
        super().__init__()
        self.inp_res = inp_res
        if int(resnet_version) == 18:
            img_feature_size = 512
            base_net = resnet.resnet18(pretrained=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            base_net = resnet.resnet50(pretrained=True)
        else:
            raise NotImplementedError("Resnet {} not supported".format(resnet_version))
        self.criterion2d = criterion2d
        mano_base_neurons = [img_feature_size] + mano_neurons
        self.mano_fhb_hand = mano_fhb_hand
        self.base_net = base_net
        # Predict translation and scaling for hand
        self.scaletrans_branch = AbsoluteBranch(
            base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=3
        )
        # Predict translation, scaling and rotation for object
        self.scaletrans_branch_obj = AbsoluteBranch(
            base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=6
        )

        # Initialize object branch
        self.obj_branch = ObjBranch(trans_factor=obj_trans_factor, scale_factor=obj_scale_factor)
        self.obj_scale_factor = obj_scale_factor
        self.obj_trans_factor = obj_trans_factor

        # Initialize mano branch
        self.mano_branch = ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            dropout=fc_dropout,
            mano_pose_coeff=mano_pose_coeff,
            mano_root=mano_root,
            center_idx=mano_center_idx,
            use_shape=mano_use_shape,
            use_pca=mano_use_pca,
        )
        self.mano_center_idx = mano_center_idx
        if self.mano_fhb_hand:
            load_fhb_path = f"assets/mano/fhb_skel_centeridx{mano_center_idx}.pkl"
            with open(load_fhb_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
            self.register_buffer("fhb_shape", torch.Tensor(exp_data["shape"]))
            self.adaptor = ManoAdaptor(self.mano_branch.mano_layer_right, load_fhb_path)
            rec_freeze(self.adaptor)
        else:
            self.adaptor = None
        if (
            mano_lambda_verts2d
            or mano_lambda_verts3d
            or mano_lambda_joints3d
            or mano_lambda_joints2d
            or mano_lambda_recov_joints3d
            or mano_lambda_recov_verts3d
        ):
            self.mano_lambdas = True
        else:
            self.mano_lambdas = False
        if obj_lambda_verts2d or obj_lambda_verts3d or obj_lambda_recov_verts3d:
            self.obj_lambdas = True
        else:
            self.obj_lambdas = False
        self.mano_loss = ManoLoss(
            lambda_verts3d=mano_lambda_verts3d,
            lambda_joints3d=mano_lambda_joints3d,
            lambda_shape=mano_lambda_shape,
            lambda_pose_reg=mano_lambda_pose_reg,
        )
        self.mano_lambda_joints2d = mano_lambda_joints2d
        self.mano_lambda_recov_joints3d = mano_lambda_recov_joints3d
        self.mano_lambda_recov_verts3d = mano_lambda_recov_verts3d
        self.mano_lambda_verts2d = mano_lambda_verts2d
        self.obj_lambda_verts2d = obj_lambda_verts2d
        self.obj_lambda_verts3d = obj_lambda_verts3d
        self.obj_lambda_recov_verts3d = obj_lambda_recov_verts3d

    def recover_mano(
        self,
        sample,
        features=None,
        pose=None,
        shape=None,
        no_loss=False,
        total_loss=None,
        scale=None,
        trans=None,
    ):
        # Get hand projection, centered
        mano_results = self.mano_branch(features, sides=sample[BaseQueries.SIDE], pose=pose, shape=shape)
        if self.adaptor:
            adapt_joints, _ = self.adaptor(mano_results["verts3d"])
            adapt_joints = adapt_joints.transpose(1, 2)
            mano_results["joints3d"] = adapt_joints - adapt_joints[:, self.mano_center_idx].unsqueeze(1)
            mano_results["verts3d"] = mano_results["verts3d"] - adapt_joints[
                :, self.mano_center_idx
            ].unsqueeze(1)
        if not no_loss:
            mano_total_loss, mano_losses = self.mano_loss.compute_loss(mano_results, sample)
            if total_loss is None:
                total_loss = mano_total_loss
            else:
                total_loss += mano_total_loss
            mano_losses["mano_total_loss"] = mano_total_loss.clone()

        # Recover hand position in camera coordinates
        if (
            self.mano_lambda_joints2d
            or self.mano_lambda_verts2d
            or self.mano_lambda_recov_joints3d
            or self.mano_lambda_recov_verts3d
        ):
            if scale is None and trans is None:
                scaletrans = self.scaletrans_branch(features)
                if trans is None:
                    trans = scaletrans[:, 1:]
                if scale is None:
                    scale = scaletrans[:, :1]
            final_trans = trans.unsqueeze(1) * self.obj_trans_factor
            final_scale = scale.view(scale.shape[0], 1, 1) * self.obj_scale_factor
            height, width = tuple(sample[TransQueries.IMAGE].shape[2:])
            camintr = sample[TransQueries.CAMINTR].cuda()
            recov_joints3d, center3d = project.recover_3d_proj(
                mano_results["joints3d"], camintr, final_scale, final_trans, input_res=(width, height)
            )
            recov_hand_verts3d = mano_results["verts3d"] + center3d
            proj_joints2d = camproject.batch_proj2d(recov_joints3d, camintr)
            proj_verts2d = camproject.batch_proj2d(mano_results["verts3d"] + center3d, camintr)

            mano_results["joints2d"] = proj_joints2d
            mano_results["recov_joints3d"] = recov_joints3d
            mano_results["recov_handverts3d"] = recov_hand_verts3d
            mano_results["verts2d"] = proj_verts2d
            mano_results["hand_pretrans"] = trans
            mano_results["hand_prescale"] = scale
            mano_results["hand_trans"] = final_trans
            mano_results["hand_scale"] = final_scale
            if not no_loss:
                # Compute hand losses in pixel space and camera coordinates
                if self.mano_lambda_joints2d is not None and TransQueries.JOINTS2D in sample:
                    gt_joints2d = sample[TransQueries.JOINTS2D].cuda().float()
                    if self.criterion2d == "l2":
                        # Normalize predictions in pixel space so that results are roughly centered
                        # and have magnitude ~1
                        norm_joints2d_pred = normalize_pixel_out(proj_joints2d)
                        norm_joints2d_gt = normalize_pixel_out(gt_joints2d)
                        joints2d_loss = torch_f.mse_loss(norm_joints2d_pred, norm_joints2d_gt)
                    elif self.criterion2d == "l1":
                        joints2d_loss = torch_f.l1_loss(proj_joints2d, gt_joints2d)
                    elif self.criterion2d == "smoothl1":
                        joints2d_loss = torch_f.smooth_l1_loss(proj_joints2d, gt_joints2d)
                    total_loss += self.mano_lambda_joints2d * joints2d_loss
                    mano_losses["joints2d"] = joints2d_loss
                if self.mano_lambda_verts2d is not None and TransQueries.HANDVERTS2D in sample:
                    gt_verts2d = sample[TransQueries.HANDVERTS2D].cuda().float()
                    verts2d_loss = torch_f.mse_loss(
                        normalize_pixel_out(proj_verts2d, self.inp_res),
                        normalize_pixel_out(gt_verts2d, self.inp_res),
                    )
                    total_loss += self.mano_lambda_verts2d * verts2d_loss
                    mano_losses["verts2d"] = verts2d_loss
                if self.mano_lambda_recov_joints3d is not None and BaseQueries.JOINTS3D in sample:
                    joints3d_gt = sample[BaseQueries.JOINTS3D].cuda()
                    recov_loss = torch_f.mse_loss(recov_joints3d, joints3d_gt)
                    total_loss += self.mano_lambda_recov_joints3d * recov_loss
                    mano_losses["recov_joint3d"] = recov_loss
                if self.mano_lambda_recov_verts3d is not None and BaseQueries.HANDVERTS3D in sample:
                    hand_verts3d_gt = sample[BaseQueries.HANDVERTS3D].cuda()
                    recov_loss = torch_f.mse_loss(recov_hand_verts3d, hand_verts3d_gt)
                    total_loss += self.mano_lambda_recov_verts3d * recov_loss
        return mano_results, total_loss, mano_losses

    def recover_object(self, sample, features=None, total_loss=None, scale=None, trans=None, rotaxisang=None):
        """
        Compute object vertex and corner positions in camera coordinates by predicting object translation
        and scaling, and recovering 3D positions given known object model
        """
        if features is None:
            scaletrans_obj = None
        else:
            scaletrans_obj = self.scaletrans_branch_obj(features)
        obj_results = self.obj_branch(sample, scaletrans_obj, scale=scale, trans=trans, rotaxisang=rotaxisang)
        obj_losses = {}
        if self.criterion2d == "l2" and TransQueries.OBJVERTS2D in sample:
            obj2d_loss = torch_f.mse_loss(
                normalize_pixel_out(obj_results["obj_verts2d"], self.inp_res),
                normalize_pixel_out(sample[TransQueries.OBJVERTS2D].cuda(), self.inp_res),
            )
            obj_losses["objverts2d"] = obj2d_loss
            total_loss += self.obj_lambda_verts2d * obj2d_loss
        elif self.criterion2d == "l1" and TransQueries.OBJVERTS2D in sample:
            obj2d_loss = torch_f.l1_loss(
                normalize_pixel_out(obj_results["obj_verts2d"], self.inp_res),
                normalize_pixel_out(sample[TransQueries.OBJVERTS2D].cuda(), self.inp_res),
            )
            obj_losses["objverts2d"] = obj2d_loss
            total_loss += self.obj_lambda_verts2d * obj2d_loss
        elif self.criterion2d == "smoothl1" and TransQueries.OBJVERTS2D in sample:
            obj2d_loss = torch_f.smooth_l1_loss(
                normalize_pixel_out(obj_results["obj_verts2d"], self.inp_res),
                normalize_pixel_out(sample[TransQueries.OBJVERTS2D].cuda(), self.inp_res),
            )
            obj_losses["objverts2d"] = obj2d_loss
            total_loss += self.obj_lambda_verts2d * obj2d_loss
        if TransQueries.OBJCANROTVERTS in sample:
            obj3d_loss = torch_f.smooth_l1_loss(
                obj_results["obj_verts3d"], sample[TransQueries.OBJCANROTVERTS].float().cuda()
            )
            obj_losses["objverts3d"] = obj3d_loss
            total_loss += self.obj_lambda_verts3d * obj3d_loss

        if self.obj_lambda_recov_verts3d is not None and BaseQueries.OBJVERTS3D in sample:
            objverts3d_gt = sample[BaseQueries.OBJVERTS3D].cuda()
            recov_verts3d = obj_results["recov_objverts3d"]

            obj_recov_loss = torch_f.mse_loss(recov_verts3d, objverts3d_gt)
            if total_loss is None:
                total_loss = self.obj_lambda_recov_verts3d * obj_recov_loss
            else:
                total_loss += self.obj_lambda_recov_verts3d * obj_recov_loss
            obj_losses["recov_objverts3d"] = obj_recov_loss
        return obj_results, total_loss, obj_losses

    def forward(self, sample, no_loss=False, step=0, preparams=None):
        total_loss = torch.Tensor([0]).cuda()
        results = {}
        losses = {}
        image = sample[TransQueries.IMAGE].cuda()
        features, _ = self.base_net(image)
        has_mano_super = one_query_in(
            sample.keys(),
            [
                TransQueries.JOINTS3D,
                TransQueries.JOINTS2D,
                TransQueries.HANDVERTS2D,
                TransQueries.HANDVERTS3D,
            ],
        )
        if has_mano_super and self.mano_lambdas:
            if preparams is not None:
                hand_scale = preparams["hand_prescale"]
                hand_pose = preparams["pose"]
                hand_shape = preparams["shape"]
                hand_trans = preparams["hand_pretrans"]
            else:
                hand_scale = None
                hand_pose = None
                hand_shape = None
                hand_trans = None
            mano_results, total_loss, mano_losses = self.recover_mano(
                sample,
                features=features,
                no_loss=no_loss,
                total_loss=total_loss,
                trans=hand_trans,
                scale=hand_scale,
                pose=hand_pose,
                shape=hand_shape,
            )
            losses.update(mano_losses)
            results.update(mano_results)

        has_obj_super = one_query_in(sample.keys(), [TransQueries.OBJVERTS2D, TransQueries.OBJVERTS3D])
        if has_obj_super and self.obj_lambdas:
            if preparams is not None:
                obj_scale = preparams["obj_prescale"]
                obj_rot = preparams["obj_prerot"]
                obj_trans = preparams["obj_pretrans"]
            else:
                obj_scale = None
                obj_rot = None
                obj_trans = None
            obj_results, total_loss, obj_losses = self.recover_object(
                sample, features, total_loss=total_loss, scale=obj_scale, trans=obj_trans, rotaxisang=obj_rot
            )
            losses.update(obj_losses)
            results.update(obj_results)

        if total_loss is not None:
            losses["total_loss"] = total_loss
        else:
            losses["total_loss"] = None
        return total_loss, results, losses
