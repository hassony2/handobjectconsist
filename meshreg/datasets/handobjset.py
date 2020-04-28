import random
import traceback

import numpy as np
from PIL import Image, ImageFilter
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms

from libyana.transformutils import colortrans, handutils
from meshreg.datasets.queries import BaseQueries, TransQueries, one_query_in
from meshreg.datasets import datutils


class HandObjSet(Dataset):
    """Hand-Object dataset
    """

    def __init__(
        self,
        pose_dataset,
        center_idx=9,
        inp_res=(256, 256),
        max_rot=np.pi,
        normalize_img=False,
        split="train",
        scale_jittering=0.3,
        center_jittering=0.2,
        train=True,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        spacing=2,
        queries=[
            BaseQueries.IMAGE,
            TransQueries.JOINTS2D,
            TransQueries.HANDVERTS3D,
            TransQueries.OBJVERTS2D,
            TransQueries.OBJCORNERS2D,
            TransQueries.HANDVERTS2D,
            TransQueries.OBJVERTS3D,
            TransQueries.OBJCORNERS3D,
            BaseQueries.OBJCANVERTS,
            BaseQueries.OBJCANCORNERS,
            TransQueries.JOINTS3D,
        ],
        sides="both",
        block_rot=False,
        sample_nb=None,
        has_dist2strong=False,
    ):
        """
        Args:
            sample_nb: Number of samples to return: first sample is
            spacing: if 0, sample closest ground truth frame
            center_idx: idx of joint on which to center 3d pose
                not present
            sides: if both, don't flip hands, if 'right' flip all left hands to
                right hands, if 'left', do the opposite
        """
        # Dataset attributes
        self.pose_dataset = pose_dataset
        self.inp_res = tuple(inp_res)
        self.normalize_img = normalize_img
        self.center_idx = center_idx
        self.sides = sides

        # Sequence attributes
        self.sample_nb = sample_nb
        self.spacing = spacing

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        self.max_rot = max_rot
        self.block_rot = block_rot

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering

        self.queries = queries
        self.has_dist2strong = has_dist2strong

    def __len__(self):
        return len(self.pose_dataset)

    def get_sample(self, idx, query=None, color_augm=None, space_augm=None):
        if query is None:
            query = self.queries
        sample = {}

        if BaseQueries.IMAGE in query or TransQueries.IMAGE in query:
            center, scale = self.pose_dataset.get_center_scale(idx)
            needs_center_scale = True
        else:
            needs_center_scale = False

        if BaseQueries.JOINTVIS in query:
            jointvis = self.pose_dataset.get_jointvis(idx)
            sample[BaseQueries.JOINTVIS] = jointvis

        # Get sides
        if BaseQueries.SIDE in query:
            hand_side = self.pose_dataset.get_sides(idx)
            hand_side, flip = datutils.flip_hand_side(self.sides, hand_side)
            sample[BaseQueries.SIDE] = hand_side
        else:
            flip = False

        # Get original image
        if BaseQueries.IMAGE in query or TransQueries.IMAGE in query:
            img = self.pose_dataset.get_image(idx)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.IMAGE in query:
                sample[BaseQueries.IMAGE] = np.array(img)

        # Flip and image 2d if needed
        if flip:
            center[0] = img.size[0] - center[0]
        # Data augmentation
        if space_augm is not None:
            center = space_augm["center"]
            scale = space_augm["scale"]
            rot = space_augm["rot"]
        elif self.train and needs_center_scale:
            # Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_jit = Uniform(low=-1, high=1).sample((2,)).numpy()
            center_offsets = self.center_jittering * scale * center_jit
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jit = Normal(0, 1).sample().item() + 1
            scale_jittering = self.scale_jittering * scale_jit
            scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
            scale = scale * scale_jittering

            rot = Uniform(low=-self.max_rot, high=self.max_rot).sample().item()
        else:
            rot = 0
        if self.block_rot:
            rot = 0
        space_augm = {"rot": rot, "scale": scale, "center": center}
        sample["space_augm"] = space_augm
        rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]]).astype(
            np.float32
        )

        # Get 2D hand joints
        if (TransQueries.JOINTS2D in query) or (TransQueries.IMAGE in query):
            affinetrans, post_rot_trans = handutils.get_affine_transform(center, scale, self.inp_res, rot=rot)
            if TransQueries.AFFINETRANS in query:
                sample[TransQueries.AFFINETRANS] = affinetrans
        if BaseQueries.JOINTS2D in query or TransQueries.JOINTS2D in query:
            joints2d = self.pose_dataset.get_joints2d(idx)
            if flip:
                joints2d = joints2d.copy()
                joints2d[:, 0] = img.size[0] - joints2d[:, 0]
            if BaseQueries.JOINTS2D in query:
                sample[BaseQueries.JOINTS2D] = joints2d.astype(np.float32)
        if TransQueries.JOINTS2D in query:
            rows = handutils.transform_coords(joints2d, affinetrans)
            sample[TransQueries.JOINTS2D] = np.array(rows).astype(np.float32)

        if BaseQueries.CAMINTR in query or TransQueries.CAMINTR in query:
            camintr = self.pose_dataset.get_camintr(idx)
            if BaseQueries.CAMINTR in query:
                sample[BaseQueries.CAMINTR] = camintr.astype(np.float32)
            if TransQueries.CAMINTR in query:
                # Rotation is applied as extr transform
                new_camintr = post_rot_trans.dot(camintr)
                sample[TransQueries.CAMINTR] = new_camintr.astype(np.float32)

        # Get 2D object points
        if BaseQueries.OBJVERTS2D in query or (TransQueries.OBJVERTS2D in query):
            objverts2d = self.pose_dataset.get_objverts2d(idx)
            if flip:
                objverts2d = objverts2d.copy()
                objverts2d[:, 0] = img.size[0] - objverts2d[:, 0]
            if BaseQueries.OBJVERTS2D in query:
                sample[BaseQueries.OBJVERTS2D] = objverts2d.astype(np.float32)
            if TransQueries.OBJVERTS2D in query:
                transobjverts2d = handutils.transform_coords(objverts2d, affinetrans)
                sample[TransQueries.OBJVERTS2D] = np.array(transobjverts2d).astype(np.float32)
            if BaseQueries.OBJVIS2D in query:
                objvis2d = self.pose_dataset.get_objvis2d(idx)
                sample[BaseQueries.OBJVIS2D] = objvis2d

        # Get 2D object points
        if BaseQueries.OBJCORNERS2D in query or (TransQueries.OBJCORNERS2D in query):
            objcorners2d = self.pose_dataset.get_objcorners2d(idx)
            if flip:
                objcorners2d = objcorners2d.copy()
                objcorners2d[:, 0] = img.size[0] - objcorners2d[:, 0]
            if BaseQueries.OBJCORNERS2D in query:
                sample[BaseQueries.OBJCORNERS2D] = np.array(objcorners2d)
            if TransQueries.OBJCORNERS2D in query:
                transobjcorners2d = handutils.transform_coords(objcorners2d, affinetrans)
                sample[TransQueries.OBJCORNERS2D] = np.array(transobjcorners2d)

        # Get 2D hand points
        if BaseQueries.HANDVERTS2D in query or (TransQueries.HANDVERTS2D in query):
            handverts2d = self.pose_dataset.get_hand_verts2d(idx)
            if flip:
                handverts2d = handverts2d.copy()
                handverts2d[:, 0] = img.size[0] - handverts2d[:, 0]
            if BaseQueries.HANDVERTS2D in query:
                sample[BaseQueries.HANDVERTS2D] = handverts2d
            if TransQueries.HANDVERTS2D in query:
                transhandverts2d = handutils.transform_coords(handverts2d, affinetrans)
                sample[TransQueries.HANDVERTS2D] = np.array(transhandverts2d)
            if BaseQueries.HANDVIS2D in query:
                handvis2d = self.pose_dataset.get_handvis2d(idx)
                sample[BaseQueries.HANDVIS2D] = handvis2d

        # Get 3D hand joints
        if (
            (BaseQueries.JOINTS3D in query)
            or (TransQueries.JOINTS3D in query)
            or (TransQueries.HANDVERTS3D in query)
            or (TransQueries.OBJVERTS3D in query)
        ):
            # Center on root joint
            center3d_queries = [TransQueries.JOINTS3D, BaseQueries.JOINTS3D, TransQueries.HANDVERTS3D]
            if one_query_in([TransQueries.OBJVERTS3D] + center3d_queries, query):
                joints3d = self.pose_dataset.get_joints3d(idx)
                if flip:
                    joints3d[:, 0] = -joints3d[:, 0]

                if BaseQueries.JOINTS3D in query:
                    sample[BaseQueries.JOINTS3D] = joints3d.astype(np.float32)
                if self.train:
                    joints3d = rot_mat.dot(joints3d.transpose(1, 0)).transpose()
                # Compute 3D center
                if self.center_idx is not None:
                    if self.center_idx == -1:
                        center3d = (joints3d[9] + joints3d[0]) / 2
                    else:
                        center3d = joints3d[self.center_idx]
                if TransQueries.JOINTS3D in query and (self.center_idx is not None):
                    joints3d = joints3d - center3d
                if TransQueries.JOINTS3D in query:
                    sample[TransQueries.JOINTS3D] = joints3d.astype(np.float32)

        # Get 3D hand vertices
        if TransQueries.HANDVERTS3D in query or BaseQueries.HANDVERTS3D in query:
            hand_verts3d = self.pose_dataset.get_hand_verts3d(idx)
            if flip:
                hand_verts3d[:, 0] = -hand_verts3d[:, 0]
            if BaseQueries.OBJVERTS3D in query:
                sample[BaseQueries.HANDVERTS3D] = hand_verts3d.astype(np.float32)
            if TransQueries.HANDVERTS3D in query:
                hand_verts3d = rot_mat.dot(hand_verts3d.transpose(1, 0)).transpose()
                if self.center_idx is not None:
                    hand_verts3d = hand_verts3d - center3d
                sample[TransQueries.HANDVERTS3D] = hand_verts3d.astype(np.float32)

        # Get 3D obj vertices
        if TransQueries.OBJVERTS3D in query or BaseQueries.OBJVERTS3D in query:
            obj_verts3d = self.pose_dataset.get_obj_verts_trans(idx)
            if flip:
                obj_verts3d[:, 0] = -obj_verts3d[:, 0]
            if BaseQueries.OBJVERTS3D in query:
                sample[BaseQueries.OBJVERTS3D] = obj_verts3d
            if TransQueries.OBJVERTS3D in query:
                origin_trans_mesh = rot_mat.dot(obj_verts3d.transpose(1, 0)).transpose()
                if self.center_idx is not None:
                    origin_trans_mesh = origin_trans_mesh - center3d
                sample[TransQueries.OBJVERTS3D] = origin_trans_mesh.astype(np.float32)

        # Get 3D obj vertices
        if TransQueries.OBJCANROTVERTS in query or BaseQueries.OBJCANROTVERTS in query:
            obj_canverts3d = self.pose_dataset.get_obj_verts_can_rot(idx)
            if flip:
                obj_canverts3d[:, 0] = -obj_canverts3d[:, 0]
            if BaseQueries.OBJCANROTVERTS in query:
                sample[BaseQueries.OBJCANROTVERTS] = obj_canverts3d
            if TransQueries.OBJCANROTVERTS in query:
                can_rot_mesh = rot_mat.dot(obj_canverts3d.transpose(1, 0)).transpose()
                sample[TransQueries.OBJCANROTVERTS] = can_rot_mesh

        # Get 3D obj vertices
        if TransQueries.OBJCANROTCORNERS in query or BaseQueries.OBJCANROTCORNERS in query:
            obj_cancorners3d = self.pose_dataset.get_obj_corners_can_rot(idx)
            if flip:
                obj_cancorners3d[:, 0] = -obj_cancorners3d[:, 0]
            if BaseQueries.OBJCANROTCORNERS in query:
                sample[BaseQueries.OBJCANROTCORNERS] = obj_cancorners3d
            if TransQueries.OBJCANROTCORNERS in query:
                can_rot_corners = rot_mat.dot(obj_cancorners3d.transpose(1, 0)).transpose()
                sample[TransQueries.OBJCANROTCORNERS] = can_rot_corners

        if BaseQueries.OBJFACES in query:
            obj_faces = self.pose_dataset.get_obj_faces(idx)
            sample[BaseQueries.OBJFACES] = obj_faces
        if BaseQueries.OBJCANVERTS in query:
            obj_canverts, obj_cantrans, obj_canscale = self.pose_dataset.get_obj_verts_can(idx)
            if flip:
                obj_canverts[:, 0] = -obj_canverts[:, 0]
            sample[BaseQueries.OBJCANVERTS] = obj_canverts
            sample[BaseQueries.OBJCANSCALE] = obj_canscale
            sample[BaseQueries.OBJCANTRANS] = obj_cantrans

        # Get 3D obj corners
        if BaseQueries.OBJCORNERS3D in query or TransQueries.OBJCORNERS3D in query:
            obj_corners3d = self.pose_dataset.get_obj_corners3d(idx)
            if flip:
                obj_corners3d[:, 0] = -obj_corners3d[:, 0]
            if BaseQueries.OBJCORNERS3D in query:
                sample[BaseQueries.OBJCORNERS3D] = obj_corners3d
            if TransQueries.OBJCORNERS3D in query:
                origin_trans_corners = rot_mat.dot(obj_corners3d.transpose(1, 0)).transpose()
                if self.center_idx is not None:
                    origin_trans_corners = origin_trans_corners - center3d
                sample[TransQueries.OBJCORNERS3D] = origin_trans_corners
        if BaseQueries.OBJCANCORNERS in query:
            if flip:
                obj_canverts[:, 0] = -obj_canverts[:, 0]
            obj_cancorners = self.pose_dataset.get_obj_corners_can(idx)
            sample[BaseQueries.OBJCANCORNERS] = obj_cancorners

        if TransQueries.CENTER3D in query:
            sample[TransQueries.CENTER3D] = center3d

        # Get rgb image
        if TransQueries.IMAGE in query:
            # Data augmentation
            if self.train:
                blur_radius = Uniform(low=0, high=1).sample().item() * self.blur_radius
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                if color_augm is None:
                    bright, contrast, sat, hue = colortrans.get_color_params(
                        brightness=self.brightness,
                        saturation=self.saturation,
                        hue=self.hue,
                        contrast=self.contrast,
                    )
                else:
                    sat = color_augm["sat"]
                    contrast = color_augm["contrast"]
                    hue = color_augm["hue"]
                    bright = color_augm["bright"]
                img = colortrans.apply_jitter(
                    img, brightness=bright, saturation=sat, hue=hue, contrast=contrast
                )
                sample["color_augm"] = {"sat": sat, "bright": bright, "contrast": contrast, "hue": hue}
            else:
                sample["color_augm"] = None
            # Create buffer white image if needed
            if TransQueries.JITTERMASK in query:
                whiteimg = Image.new("RGB", img.size, (255, 255, 255))
            # Transform and crop
            img = handutils.transform_img(img, affinetrans, self.inp_res)
            img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))

            # Tensorize and normalize_img
            img = func_transforms.to_tensor(img).float()
            if self.normalize_img:
                img = func_transforms.normalize(img, self.mean, self.std)
            else:
                img = func_transforms.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
            if TransQueries.IMAGE in query:
                sample[TransQueries.IMAGE] = img
            if TransQueries.JITTERMASK in query:
                jittermask = handutils.transform_img(whiteimg, affinetrans, self.inp_res)
                jittermask = jittermask.crop((0, 0, self.inp_res[0], self.inp_res[1]))
                jittermask = func_transforms.to_tensor(jittermask).float()
                sample[TransQueries.JITTERMASK] = jittermask
        if self.pose_dataset.has_dist2strong and self.has_dist2strong:
            dist2strong = self.pose_dataset.get_dist2strong(idx)
            sample["dist2strong"] = dist2strong

        return sample

    def get_safesample(self, idx, color_augm=None, space_augm=None):
        try:
            sample = self.get_sample(idx, self.queries, color_augm=color_augm, space_augm=space_augm)
        except Exception:
            traceback.print_exc()
            rand_idx = random.randint(max(0, idx - 10), min(len(self), idx + 10))
            print(f"Encountered error processing sample {idx}, trying {rand_idx} instead")
            sample = self.get_sample(rand_idx, self.queries)
        return sample

    def __getitem__(self, idx):
        sample = self.get_safesample(idx)
        sample["dist2query"] = 0
        space_augm = sample.pop("space_augm")
        color_augm = sample.pop("color_augm")
        if self.sample_nb is not None:
            samples = [sample]
            dist = 0
            # Keep track if last frame was sampled forward or backward
            sign = True
            # If several samples are needed, either get closest frame
            # (if spacing == 0) or alternate between future and past frames
            # (if spacing > 0)
            for sample_idx in range(self.sample_nb - 1):
                if sign:
                    dist += self.spacing
                    # cur_dist > 0 signals to get future closest annotated frame
                    cur_dist = dist
                    sign = False
                else:
                    # cur_dist < 0 signals to get past closest annotated frame
                    cur_dist = -dist
                    sign = True
                    dist += self.spacing
                next_idx, dist2query = self.pose_dataset.get_dist_idx(idx, dist=cur_dist)
                # Get past and future samples with Same data augmentation (so that
                # photometric consistency holds)
                sample_next = self.get_safesample(next_idx, color_augm=color_augm, space_augm=space_augm)
                sample_next["dist2query"] = dist2query
                sample_next.pop("space_augm")
                sample_next.pop("color_augm")
                samples.append(sample_next)
            return samples
        else:
            return sample
