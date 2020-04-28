import os
import pickle
import warnings

import cv2
import numpy as np
from PIL import Image
import torch

from manopth import manolayer

from meshreg.datasets import ho3dfullutils, ho3dv2utils
from meshreg.datasets.queries import BaseQueries, get_trans_queries
from libyana.meshutils import meshnorm


class HO3DV2:
    def __init__(
        self,
        split,
        split_mode="objects",
        root="data",
        fraction=1,
        joint_nb=21,
        mini_factor=0,
        full_image=True,
        mode="full",
        full_sequences=False,
        use_cache=True,
        like_v1=True,
    ):  # Put back one to compare with arxiv citation !!
        super().__init__()
        self.split_mode = split_mode
        if split_mode == "paper" and mode == "weak":
            raise ValueError(f"split mode {split_mode} incompatible will mode {mode} != full")

        self.name = "ho3dv2"
        self.full_sequences = full_sequences
        self.fraction = fraction
        self.has_dist2strong = True
        self.image_size = [640, 480]
        cache_folder = os.path.join("data", "cache")
        os.makedirs(cache_folder, exist_ok=True)
        if like_v1:
            prefix = "likev1"
        else:
            prefix = ""
        cache_path = os.path.join(
            cache_folder, f"{self.name}_{prefix}_{split_mode}_{split}_{mode}_{fraction}.pkl"
        )

        self.root = os.path.join(root, self.name)
        self.joint_nb = joint_nb
        self.reorder_idxs = np.array(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        )
        self.mini_factor = mini_factor
        self.full_image = full_image
        if fraction != 1:
            assert mini_factor in [1, 0, None], f"fraction and minifactor not simulatneously supported"

        self.all_queries = [
            BaseQueries.IMAGE,
            BaseQueries.JOINTS2D,
            BaseQueries.JOINTS3D,
            BaseQueries.OBJVERTS2D,
            BaseQueries.OBJVIS2D,
            BaseQueries.OBJVERTS3D,
            BaseQueries.OBJCORNERS2D,
            BaseQueries.OBJCORNERS3D,
            BaseQueries.OBJCANCORNERS,
            # BaseQueries.OBJCANROTCORNERS,
            # BaseQueries.OBJCANROTVERTS,
            BaseQueries.OBJFACES,
            BaseQueries.HANDVERTS2D,
            BaseQueries.HANDVIS2D,
            BaseQueries.HANDVERTS3D,
            BaseQueries.OBJCANVERTS,
            BaseQueries.SIDE,
            BaseQueries.CAMINTR,
            BaseQueries.JOINTVIS,
        ]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        # Fix dataset split
        valid_splits = ["train", "trainval", "val", "test"]
        assert split in valid_splits, "{} not in {}".format(split, valid_splits)
        self.split = split
        self.cam_intr = np.array([[617.343, 0.0, 312.42], [0.0, 617.343, 241.42], [0.0, 0.0, 1.0]]).astype(
            np.float32
        )
        self.cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root="assets/mano",
            center_idx=None,
            flat_hand_mean=True,
        )
        if self.split_mode == "objects":
            if self.split == "train":
                if like_v1:
                    seqs = {"SM5", "MC6", "MC4", "SM3", "SM4", "SS3", "SS2", "SM2", "SS1", "MC5", "MC1"}
                else:
                    seqs = {
                        "ABF11",
                        "ABF12",
                        "ABF13",
                        "ABF14",
                        "BB10",
                        "BB12",
                        "BB13",
                        "BB14",
                        "GPMF10",
                        "GPMF11",
                        "GPMF13",
                        "GPMF14",
                        "GSF10",
                        "GSF11",
                        "GSF12",
                        "GSF14",
                        "MC2",
                        "MC4",
                        "MC5",
                        "MC6",
                        "MDF10",
                        "MDF11",
                        "MDF12",
                        "MDF13",
                        "SB10",
                        "SB12",
                        "ShSu12",
                        "ShSu13",
                        "ShSu14",
                        "SiBF10",
                        "SiBF12",
                        "SiBF13",
                        "SiBF14",
                        "SM2",
                        "SM4",
                        "SM5",
                        "SMu40",
                        "SMu41",
                        "SS1",
                        "SS3",
                        "SMu42",
                    }
                subfolder = "train"
            elif self.split == "trainval":
                seqs = {
                    "ABF12",
                    "ABF13",
                    "ABF14",
                    "BB10",
                    "BB13",
                    "BB14",
                    "GPMF10",
                    "GPMF11",
                    "GPMF14",
                    "GSF10",
                    "GSF11",
                    "GSF12",
                    "MC2",
                    "MC4",
                    "MC5",
                    "MC6",
                    "MDF10",
                    "MDF11",
                    "MDF12",
                    "MDF13",
                    "SB10",
                    "SB12",
                    "ShSu12",
                    "ShSu13",
                    "ShSu14",
                    "SiBF10",
                    "SiBF12",
                    "SiBF13",
                    "SiBF14",
                    "SM2",
                    "SM4",
                    "SM5",
                    "SMu40",
                    "SMu41",
                    "SS1",
                    "SS3",
                    "SMu42",
                }
                subfolder = "train"
            elif self.split == "val":
                seqs = {"ABF11", "BB12", "GPMF13", "GSF14"}
                subfolder = "train"
            elif self.split == "test":
                if like_v1:
                    seqs = {"MC2"}
                else:
                    seqs = {
                        "ABF10",
                        "MC1",
                        "MDF14",
                        "BB11",
                        "GPMF12",
                        "GSF13",
                        "SB14",
                        "ShSu10",
                        "SM3",
                        "SMu1",
                        "SiBF11",
                        "SS2",
                    }
                subfolder = "train"
                warnings.warn(f"Using seqs {seqs} for evaluation")
                print(f"Using seqs {seqs} for evaluation")

        self.mode = mode

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
        else:
            if self.split_mode == "objects":
                all_seqs, seq_map, closeseq_map, strongs, weaks, idxs = ho3dv2utils.get_objectsplit_infos(
                    seqs, self.root, subfolder, fraction=fraction
                )
            elif self.split_mode == "paper":
                all_seqs, seq_map, closeseq_map, strongs, weaks, idxs = ho3dv2utils.get_papersplit_infos(
                    split, self.root
                )
            annotations = {
                "all_sequences": all_seqs,
                "closeseq_map": closeseq_map,
                "strongs": strongs,
                "weaks": weaks,
                "idxs": idxs,
                "seq_map": seq_map,
            }
            with open(cache_path, "wb") as p_f:
                pickle.dump(annotations, p_f)

        self.all_sequences = annotations["all_sequences"]
        self.closeseq_map = annotations["closeseq_map"]
        self.strongs = annotations["strongs"]
        self.weaks = annotations["weaks"]
        self.idxs = annotations["idxs"]
        self.seq_map = annotations["seq_map"]

        self.fulls = self.strongs + self.weaks

        assert len(self.fulls) == len(self.idxs)
        assert not len((set(self.weaks) | set(self.strongs)) - set(self.fulls))
        if len(self.weaks):
            weak_distances = [abs(key - self.closeseq_map[key]["closest"]) for key in self.weaks]
            assert min(weak_distances) == 1
        if len(self.strongs):
            strong_distances = [abs(key - self.closeseq_map[key]["closest"]) for key in self.strongs]
            assert min(strong_distances) == 0
            assert max(strong_distances) == 0
        self.obj_meshes = ho3dfullutils.load_objects(os.path.join(self.root, "modelsprocess"))

        # Get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def get_dataidx(self, idx):
        if self.mode == "strong":
            return self.strongs[idx]
        elif self.mode == "weak":
            return self.weaks[idx]
        elif self.mode == "full":
            return self.fulls[idx]
        else:
            raise ValueError(f"Mode {self.mode} not in [full|strong|weak]")

    def get_invdataidx(self, dataidx):
        return self.fulls.index(dataidx)

    def get_dist_idx(self, idx, dist=4):
        idx = self.get_dataidx(idx)
        if dist > 0:
            dist_dataidx = self.closeseq_map[idx]["next"]
        elif dist < 0:
            dist_dataidx = self.closeseq_map[idx]["previous"]
        elif dist == 0:
            dist_dataidx = self.closeseq_map[idx]["closest"]
        return self.get_invdataidx(dist_dataidx), abs(dist_dataidx - idx)

    def get_dist2strong(self, idx):
        data_idx = self.get_dataidx(idx)
        closest_dataidx = self.closeseq_map[data_idx]["closest"]
        dist = abs(data_idx - closest_dataidx)
        return dist

    def get_image(self, idx):
        data_idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[data_idx]
        img_path = self.seq_map[seq][img_idx]["img"]
        img = Image.open(img_path).convert("RGB")
        return img

    def get_jointvis(self, idx):
        return np.ones(self.joint_nb)

    def get_joints2d(self, idx):
        joints3d = self.get_joints3d(idx)
        cam_intr = self.get_camintr(idx)
        return self.project(joints3d, cam_intr)

    def get_joints3d(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        joints3d = annot["handJoints3D"]
        joints3d = self.cam_extr[:3, :3].dot(joints3d.transpose()).transpose()
        joints3d = joints3d[self.reorder_idxs]
        return joints3d.astype(np.float32)

    def get_obj_textures(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        textures = self.obj_meshes[obj_id]["textures"]
        return textures

    def get_hand_info(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        # Retrieve hand info
        handpose = annot["handPose"]
        handtrans = annot["handTrans"]
        handshape = annot["handBeta"]
        return handpose, handtrans, handshape

    def get_hand_verts3d(self, idx):
        handpose, handtrans, handshape = self.get_hand_info(idx)
        handverts, handjoints = self.layer(
            torch.Tensor(handpose).unsqueeze(0), torch.Tensor(handshape).unsqueeze(0)
        )
        handverts = handverts[0].numpy() / 1000 + handtrans
        trans_handverts = self.cam_extr[:3, :3].dot(handverts.transpose()).transpose()
        return trans_handverts

    def get_hand_verts2d(self, idx):
        verts3d = self.get_hand_verts3d(idx)
        cam_intr = self.get_camintr(idx)
        verts2d = self.project(verts3d, cam_intr)
        return verts2d

    def get_obj_faces(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        objfaces = self.obj_meshes[obj_id]["faces"]
        objfaces = np.array(objfaces).astype(np.int16)
        return objfaces

    def get_obj_verts_can(self, idx, rescale=True, no_center=False):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        obj_id = annot["objName"]
        verts = self.obj_meshes[obj_id]["verts"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()
        if rescale:
            return meshnorm.center_vert_bbox(verts, scale=False)
        elif no_center:
            return verts, np.array([0, 0]), 1
        else:
            return meshnorm.center_vert_bbox(verts, scale=False)

    def get_obj_rot(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        return rot

    def get_obj_verts_trans(self, idx):
        rot = self.get_obj_rot(idx)
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        trans = annot["objTrans"]
        obj_id = annot["objName"]
        verts = self.obj_meshes[obj_id]["verts"]
        trans_verts = rot.dot(verts.transpose()).transpose() + trans
        trans_verts = self.cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
        obj_verts = np.array(trans_verts).astype(np.float32)
        return obj_verts

    def get_obj_verts_can_rot(self, idx):
        verts, _, _ = self.get_obj_verts_can(idx)
        rot = self.get_obj_rot(idx)
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()
        verts = rot.dot(verts.transpose()).transpose()
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()
        obj_verts = np.array(verts).astype(np.float32)
        return obj_verts

    def get_obj_pose(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        trans = annot["objTrans"]
        rot = self.cam_extr[:3, :3].dot(rot.dot(self.cam_extr[:3, :3]))
        trans = trans * np.array([1, -1, -1])
        return rot, trans

    def get_obj_corners3d(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        trans = annot["objTrans"]
        corners = annot["objCorners3DRest"]
        trans_corners = rot.dot(corners.transpose()).transpose() + trans
        trans_corners = self.cam_extr[:3, :3].dot(trans_corners.transpose()).transpose()
        obj_corners = np.array(trans_corners).astype(np.float32)
        return obj_corners

    def get_obj_corners_can(self, idx):
        _, obj_cantrans, obj_canscale = self.get_obj_verts_can(idx)
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        annot = self.seq_map[seq][img_idx]
        corners = annot["objCorners3DRest"]
        corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()
        obj_cancorners = (corners - obj_cantrans) / obj_canscale
        return obj_cancorners

    def get_obj_corners_can_rot(self, idx):
        corners = self.get_obj_corners_can(idx)
        rot = self.get_obj_rot(idx)
        corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()
        corners = rot.dot(corners.transpose()).transpose()
        corners = self.cam_extr[:3, :3].dot(corners.transpose()).transpose()
        obj_corners = np.array(corners).astype(np.float32)
        return obj_corners

    def get_objcorners2d(self, idx):
        corners3d = self.get_obj_corners3d(idx)
        cam_intr = self.get_camintr(idx)
        return self.project(corners3d, cam_intr)

    def get_objverts2d(self, idx):
        objpoints3d = self.get_obj_verts_trans(idx)
        cam_intr = self.get_camintr(idx)
        verts2d = self.project(objpoints3d, cam_intr)
        return verts2d

    def get_objvis2d(self, idx):
        objvis = np.ones_like(self.get_objverts2d(idx)[:, 0])
        return objvis

    def get_handvis2d(self, idx):
        handvis = np.ones_like(self.get_hand_verts2d(idx)[:, 0])
        return handvis

    def get_sides(self, idx):
        return "right"

    def get_camintr(self, idx):
        idx = self.get_dataidx(idx)
        seq, img_idx = self.idxs[idx]
        cam_intr = self.seq_map[seq][img_idx]["camMat"]
        return cam_intr

    def __len__(self):
        if self.full_sequences:
            return len(self.all_sequences)
        else:
            if self.mode == "strong":
                return len(self.strongs)
            elif self.mode == "weak":
                return len(self.weaks)
            else:
                return len(self.fulls)

    def get_center_scale(self, idx):
        idx = self.get_dataidx(idx)
        if self.full_image:
            center = np.array([320, 240])
            scale = 640
        else:
            img_name = self.image_names[idx]
            bbox = self.box_infos[img_name]["bbox"]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[2]) / 2])
            scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        return center, scale

    def project(self, points3d, cam_intr):
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)
