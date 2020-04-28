import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f

from manopth.manolayer import ManoLayer

from meshreg.datasets.queries import TransQueries


class ManoBranch(nn.Module):
    def __init__(
        self,
        ncomps: int = 6,
        base_neurons=[512, 512],
        center_idx: int = 9,
        use_shape=False,
        use_pca=True,
        mano_root="misc/mano",
        mano_pose_coeff=1,
        dropout=0,
    ):
        """
        Args:
            mano_root (path): dir containing mano pickle files
            center_idx: Joint idx on which to hand is centered (given joint has position
                [0, 0, 0]
            ncomps: Number of pose principal components that are predicted
        """
        super(ManoBranch, self).__init__()

        self.use_shape = use_shape
        self.use_pca = use_pca
        self.mano_pose_coeff = mano_pose_coeff

        if self.use_pca:
            # Final number of coefficients to predict for pose
            # is sum of PCA components and 3 global axis-angle params
            # for the global rotation
            mano_pose_size = ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 components per joint
            # rotation
            mano_pose_size = 16 * 9
        # Initial base layers of MANO decoder
        base_layers = []
        for inp_neurons, out_neurons in zip(base_neurons[:-1], base_neurons[1:]):
            if dropout:
                base_layers.append(nn.Dropout(p=dropout))
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        # Pose layers to predict pose parameters
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)
        if not self.use_pca:
            # Initialize all nondiagonal items on rotation matrix weights to 0
            self.pose_reg.bias.data.fill_(0)
            weight_mask = self.pose_reg.weight.data.new(np.identity(3)).view(9).repeat(16)
            self.pose_reg.weight.data = torch.abs(
                weight_mask.unsqueeze(1).repeat(1, 256).float() * self.pose_reg.weight.data
            )

        # Shape layers to predict MANO shape parameters
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(nn.Linear(base_neurons[-1], 10))

        # Mano layer which outputs the hand mesh given the hand pose and shape
        # paramters
        self.mano_layer_right = ManoLayer(
            ncomps=ncomps,
            center_idx=center_idx,
            side="right",
            mano_root=mano_root,
            use_pca=use_pca,
            flat_hand_mean=False,
        )
        self.mano_layer_left = ManoLayer(
            ncomps=ncomps,
            center_idx=center_idx,
            side="left",
            mano_root=mano_root,
            use_pca=use_pca,
            flat_hand_mean=False,
        )
        self.faces = self.mano_layer_right.th_faces

    def forward(self, inp, sides, shape=None, pose=None):
        base_features = self.base_layer(inp)
        if pose is None:
            pose = self.pose_reg(base_features)
        if self.mano_pose_coeff != 1:
            pose = torch.cat([pose[:, :3], self.mano_pose_coeff * pose[:, 3:]], 1)
        if not self.use_pca:
            # Reshape to rotation matrixes
            mano_pose = pose.reshape(pose.shape[0], 16, 3, 3)
        else:
            mano_pose = pose

        # Prepare for splitting batch in right hands and left hands
        is_rights = (inp.new_tensor([side == "right" for side in sides])).bool()
        is_lefts = ~is_rights
        is_rights = is_rights[: pose.shape[0]]
        is_lefts = is_lefts[: pose.shape[0]]

        # Get shape
        if shape is not None:
            shape_right = shape[is_rights]
            shape_left = shape[is_lefts]
        elif self.use_shape:
            shape = self.shape_reg(base_features)
            shape_right = shape[is_rights]
            shape_left = shape[is_lefts]
        else:
            shape = None
            shape_right = None
            shape_left = None

        # Get trans
        trans_right = torch.Tensor([0])
        trans_left = torch.Tensor([0])

        # Get pose
        pose_right = mano_pose[is_rights]
        pose_left = mano_pose[is_lefts]

        # Get MANO vertices and joints for left and right hands given
        # predicted mano parameters
        if pose_right.shape[0] > 0:
            verts_right, joints_right = self.mano_layer_right(
                pose_right, th_betas=shape_right, th_trans=trans_right
            )
        if pose_left.shape[0] > 0:
            verts_left, joints_left = self.mano_layer_left(
                pose_left, th_betas=shape_left, th_trans=trans_left
            )
        # Reassemble right and left hands
        verts = inp.new_empty((inp.shape[0], 778, 3))
        joints = inp.new_empty((inp.shape[0], 21, 3))
        if pose_right.shape[0] > 0:
            verts[is_rights] = verts_right
            joints[is_rights] = joints_right
        if pose_left.shape[0] > 0:
            verts[is_lefts] = verts_left
            joints[is_lefts] = joints_left
        if shape is not None:
            shape = inp.new_empty((inp.shape[0], 10))
            if pose_right.shape[0] > 0:
                shape[is_rights] = shape_right
            if pose_left.shape[0] > 0:
                shape[is_lefts] = shape_left

        # Gather results in metric space (vs MANO millimeter outputs)
        results = {"verts3d": verts / 1000, "joints3d": joints / 1000, "shape": shape, "pose": pose}
        return results


class ManoLoss:
    def __init__(
        self,
        lambda_verts3d: float = None,
        lambda_joints3d: float = None,
        lambda_shape: float = None,
        lambda_pose_reg: float = None,
        center_idx: int = 9,
    ):
        """
        Computed terms of MANO weighted loss, which encompasses vertex/joint
        supervision and pose/shape regularization
        """
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_shape = lambda_shape
        self.lambda_pose_reg = lambda_pose_reg
        self.center_idx = center_idx

    def compute_loss(self, preds, target):
        final_loss = torch.Tensor([0]).cuda()
        reg_loss = torch.Tensor([0]).cuda()
        mano_losses = {}

        # If needed, compute and add vertex loss
        if TransQueries.HANDVERTS3D in target and self.lambda_verts3d:
            verts3d_loss = torch_f.mse_loss(preds["verts3d"], target[TransQueries.HANDVERTS3D].cuda())
            final_loss += self.lambda_verts3d * verts3d_loss
            verts3d_loss = verts3d_loss
        else:
            verts3d_loss = None
        mano_losses["mano_verts3d"] = verts3d_loss

        # Compute joints loss in all cases
        if TransQueries.JOINTS3D in target:
            pred_joints = preds["joints3d"]
            target_joints = target[TransQueries.JOINTS3D]

            # Add to final_loss for backpropagation if needed
            if self.lambda_joints3d:
                joints3d_loss = torch_f.mse_loss(pred_joints, target_joints.cuda())
                final_loss += self.lambda_joints3d * joints3d_loss
                mano_losses["mano_joints3d"] = joints3d_loss

        # Compute hand shape regularization loss
        if self.lambda_shape:
            shape_loss = torch_f.mse_loss(preds["shape"], torch.zeros_like(preds["shape"]))
            final_loss += self.lambda_shape * shape_loss
            reg_loss += self.lambda_shape * shape_loss
            shape_loss = shape_loss
        else:
            shape_loss = None
        mano_losses["mano_shape"] = shape_loss

        # Compute hand pose regularization loss
        if self.lambda_pose_reg:
            pose_reg_loss = torch_f.mse_loss(preds["pose"][:, 3:], torch.zeros_like(preds["pose"][:, 3:]))
            final_loss += self.lambda_pose_reg * pose_reg_loss
            reg_loss += self.lambda_pose_reg * pose_reg_loss
            mano_losses["pose_reg"] = pose_reg_loss

        mano_losses["mano_total_loss"] = final_loss
        mano_losses["mano_reg_loss"] = reg_loss
        return final_loss, mano_losses
