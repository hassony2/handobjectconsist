import os
import pickle
import random

import numpy as np
from PIL import Image, ImageFile
import torch
from manopth import manolayer

from libyana.transformutils import handutils
from libyana.meshutils import meshnorm

from meshreg.datasets import fhbutils
from meshreg.datasets.queries import BaseQueries, get_trans_queries

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FHBHands(object):
    def __init__(
        self,
        root="data",
        split="train",
        split_type="actions",
        joint_nb=21,
        use_cache=True,
        mini_factor=None,
        use_objects=True,
        filter_object=None,
        center_hand=False,
        fraction=1,
        mode="strong",
    ):
        """
        Args:
            center_hand: to center on hand, else full image
            fraction: fraction of data with full annotation
            mode: [strong|full|weak]
            split_type: [actions|subjects]
        """
        super().__init__()
        self.center_hand = center_hand
        self.use_objects = use_objects
        self.fraction = fraction
        self.mode = mode
        self.has_dist2strong = True
        if fraction != 1:
            assert mini_factor in [1, 0, None], f"fraction and minifactor not simulatneously supported"
        # Set cache path
        self.use_cache = use_cache
        self.cache_folder = os.path.join("data", "cache", "fhb")
        os.makedirs(self.cache_folder, exist_ok=True)

        # Get queries
        self.all_queries = [
            BaseQueries.IMAGE,
            BaseQueries.JOINTS2D,
            BaseQueries.JOINTS3D,
            BaseQueries.OBJVERTS2D,
            # BaseQueries.OBJVIS2D,
            BaseQueries.OBJVERTS3D,
            BaseQueries.HANDVERTS3D,
            BaseQueries.HANDVERTS2D,
            # BaseQueries.OBJCANROTVERTS,
            BaseQueries.OBJFACES,
            BaseQueries.OBJCANVERTS,
            BaseQueries.SIDE,
            BaseQueries.CAMINTR,
            BaseQueries.JOINTVIS,
        ]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        # Get camera info
        self.cam_extr = np.array(
            [
                [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                [0, 0, 0, 1],
            ]
        )
        self.cam_intr = np.array([[1395.749023, 0, 935.732544], [0, 1395.749268, 540.681030], [0, 0, 1]])

        self.reorder_idx = np.array(
            [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]
        )
        self.name = "fhb"
        self.joint_nb = joint_nb
        self.mini_factor = mini_factor
        split_opts = ["actions", "objects", "subjects"]
        self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
        if split_type not in split_opts:
            raise ValueError(
                "Split for dataset {} should be in {}, got {}".format(self.name, split_opts, split_type)
            )

        self.split_type = split_type

        self.root = os.path.join(root, "fhbhands")
        self.info_root = os.path.join(self.root, "Subjects_info")
        self.info_split = os.path.join(self.root, "data_split_action_recognition.txt")
        self.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root="assets/mano",
            center_idx=None,
            flat_hand_mean=True,
        )
        self.reduce_res = True
        small_rgb = os.path.join(self.root, "Video_files_480")
        if os.path.exists(small_rgb) and self.reduce_res:
            print("Using reduced images for faster computations!")
            self.rgb_root = small_rgb
            self.reduce_factor = 1 / 4
        else:
            self.rgb_root = os.path.join(self.root, "Video_files")
            self.reduce_factor = 1
        self.skeleton_root = os.path.join(self.root, "Hand_pose_annotation_v1")
        self.filter_object = filter_object
        # Get file prefixes for images and annotations
        self.split = split
        self.rgb_template = "color_{:04d}.jpeg"
        # Joints are numbered from tip to base, we want opposite
        self.idxs = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19]
        self.load_dataset()

        # Infor for rendering
        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor
        self.image_size = [int(1920 * self.reduce_factor), int(1080 * self.reduce_factor)]
        print("Got {} samples for split {}".format(len(self.image_names), self.split))

        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def load_dataset(self):
        suffix = ""
        if not self.use_objects and self.split_type == "subjects":
            suffix = "{}_hand_all".format(suffix)
        if self.split_type == "subjects":
            suffix = suffix + "_or_subjects"
        if self.center_hand:
            suffix = suffix + "ch"
        cache_path = os.path.join(
            self.cache_folder,
            f"{self.split}_{self.mini_factor}_{suffix}_{self.split_type}"
            f"_filt{self.filter_object}_{self.reduce_factor}"
            f"_{self.fraction}_{self.mode}.pkl",
        )
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, "rb") as cache_f:
                annotations = pickle.load(cache_f)
            print("Cached information for dataset {} loaded from {}".format(self.name, cache_path))

        else:
            subjects_infos = {}
            for subject in self.subjects:
                subject_info_path = os.path.join(self.info_root, "{}_info.txt".format(subject))
                subjects_infos[subject] = {}
                with open(subject_info_path, "r") as subject_f:
                    raw_lines = subject_f.readlines()
                    for line in raw_lines[3:]:
                        line = " ".join(line.split())
                        action, action_idx, length = line.strip().split(" ")
                        subjects_infos[subject][(action, action_idx)] = length
            skel_info = fhbutils.get_skeletons(self.skeleton_root, subjects_infos)

            with open(self.info_split, "r") as annot_f:
                lines_raw = annot_f.readlines()
            train_list, test_list, all_infos = fhbutils.get_action_train_test(lines_raw, subjects_infos)
            all_objects = ["juice", "liquid_soap", "milk", "salt"]
            if self.filter_object:
                all_objects = [self.filter_object]

            if self.use_objects is True:
                self.fhb_objects = fhbutils.load_objects(
                    obj_root=os.path.join(self.root, "Object_models"), object_names=all_objects
                )
                obj_infos = fhbutils.load_object_infos(
                    os.path.join(self.root, "Object_6D_pose_annotation_v1_1")
                )

            if self.split_type == "actions":
                if self.split == "train":
                    sample_list = train_list
                elif self.split == "test":
                    sample_list = test_list
                elif self.split == "all":
                    sample_list = train_list + test_list
                else:
                    raise ValueError(
                        "Split {} not valid for fhbhands, should be [train|test|all]".format(self.split)
                    )
            elif self.split_type == "subjects":
                if self.split == "train":
                    subjects = ["Subject_1", "Subject_3", "Subject_4"]
                elif self.split == "test":
                    subjects = ["Subject_2", "Subject_5", "Subject_6"]
                else:
                    raise ValueError(f"Split {self.split} not in [train|test] for split_type subjects")
                self.subjects = subjects
                sample_list = all_infos
            elif self.split_type == "objects":
                sample_list = all_infos
            else:
                raise ValueError(
                    "split_type should be in [action|objects|subjects], got {}".format(self.split_type)
                )
            if self.split_type != "subjects":
                self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
            if self.use_objects and self.split_type != "objects":
                self.split_objects = self.fhb_objects

            image_names = []
            joints2d = []
            joints3d = []
            hand_sides = []
            clips = []
            sample_infos = []
            if self.use_objects:
                objnames = []
                objtransforms = []
            for subject, action_name, seq_idx, frame_idx in sample_list:
                img_path = os.path.join(
                    self.rgb_root, subject, action_name, seq_idx, "color", self.rgb_template.format(frame_idx)
                )
                skel = skel_info[subject][(action_name, seq_idx)][frame_idx]
                skel = skel[self.reorder_idx]

                skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                skel_camcoords = self.cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
                if subject in self.subjects:
                    if self.use_objects:
                        if subject in obj_infos and (action_name, seq_idx, frame_idx) in obj_infos[subject]:
                            obj, trans = obj_infos[subject][(action_name, seq_idx, frame_idx)]
                            if obj in self.split_objects:
                                clips.append((subject, action_name, seq_idx))
                                objtransforms.append(trans)
                                objnames.append(obj)
                            else:
                                continue
                        else:
                            # Skip samples without objects if object mode
                            continue
                else:
                    continue

                image_names.append(img_path)
                sample_infos.append(
                    {
                        "subject": subject,
                        "action_name": action_name,
                        "seq_idx": seq_idx,
                        "frame_idx": frame_idx,
                    }
                )
                joints3d.append(skel_camcoords)
                hom_2d = np.array(self.cam_intr).dot(skel_camcoords.transpose()).transpose()
                skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]

                joints2d.append(skel2d.astype(np.float32))
                hand_sides.append("right")

            mano_objs, mano_infos = fhbutils.load_manofits(sample_infos)
            annotations = {
                "image_names": image_names,
                "joints2d": joints2d,
                "joints3d": joints3d,
                "hand_sides": hand_sides,
                "sample_infos": sample_infos,
                "mano_infos": mano_infos,
                "mano_objs": mano_objs,
            }
            if self.use_objects:
                annotations["objnames"] = objnames
                annotations["objtransforms"] = objtransforms
                annotations["split_objects"] = self.split_objects
            if self.mini_factor and self.mini_factor != 1:
                idxs = list(range(len(image_names)))
                mini_nb = int(len(image_names) * self.mini_factor)
                random.Random(1).shuffle(idxs)
                idxs = idxs[:mini_nb]
                for key, vals in annotations.items():
                    if key != "split_objects":
                        annotations[key] = [vals[idx] for idx in idxs]
            with open(cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            print("Wrote cache for dataset {} to {}".format(self.name, cache_path))

            # Get image paths
        self.image_names = annotations["image_names"]
        self.joints2d = annotations["joints2d"]
        self.joints3d = annotations["joints3d"]
        self.hand_sides = annotations["hand_sides"]
        self.sample_infos = annotations["sample_infos"]
        self.mano_objs = annotations["mano_objs"]
        self.mano_infos = annotations["mano_infos"]
        self.seq_map, _, closeseqmap, strongs, weaks = fhbutils.get_seq_map(
            self.sample_infos, fraction=self.fraction
        )
        self.strongs = strongs
        self.weaks = weaks
        self.fulls = strongs + weaks
        self.viz_list = [479, 3010, 3820, 40, 1430, 1095, 3530, 175, 40, 4330]
        self.closeseqmap = closeseqmap
        if self.use_objects:
            self.objnames = annotations["objnames"]
            self.objtransforms = annotations["objtransforms"]
            self.split_objects = annotations["split_objects"]

    def get_dataidx(self, idx):
        if self.mode == "strong":
            return self.strongs[idx]
        elif self.mode == "weak":
            return self.weaks[idx]
        elif self.mode == "full":
            return self.fulls[idx]
        elif self.mode == "viz":
            return self.viz_list[idx]
        else:
            raise ValueError(f"Mode {self.mode} not in [full|strong|weak]")

    def get_invdataidx(self, dataidx):
        return self.fulls.index(dataidx)

    def get_dist_idx(self, idx, dist=4):
        idx = self.get_dataidx(idx)
        if dist > 0:
            dist_dataidx = self.closeseqmap[idx]["next"]
        elif dist < 0:
            dist_dataidx = self.closeseqmap[idx]["previous"]
        else:
            # If spacing is 0 --> fetch closest
            dist_dataidx = self.closeseqmap[idx]["closest"]
        return self.get_invdataidx(dist_dataidx), abs(dist_dataidx - idx)

    def get_dist2strong(self, idx):
        data_idx = self.get_dataidx(idx)
        closest_dataidx = self.closeseqmap[data_idx]["closest"]
        dist = abs(data_idx - closest_dataidx)
        return dist

    def get_image(self, idx):
        idx = self.get_dataidx(idx)
        img_path = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        return img

    def get_hand_verts3d(self, idx):
        pose, trans, shape = self.get_hand_info(idx)
        verts, _ = self.layer(torch.Tensor(pose).unsqueeze(0), torch.Tensor(shape).unsqueeze(0))
        verts = verts[0].numpy() / 1000 + trans
        return np.array(verts).astype(np.float32)

    def get_hand_verts2d(self, idx):
        verts = self.get_hand_verts3d(idx)
        hom_2d = np.array(self.cam_intr).dot(verts.transpose()).transpose()
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return np.array(verts2d).astype(np.float32)

    def get_hand_info(self, idx):
        idx = self.get_dataidx(idx)
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"], mano_info["trans"], mano_info["shape"]

    def get_obj_textures(self, idx):
        idx = self.get_dataidx(idx)
        obj = self.objnames[idx]
        objtextures = self.split_objects[obj]["textures"]
        return np.array(objtextures)

    def get_obj_faces(self, idx):
        idx = self.get_dataidx(idx)
        obj = self.objnames[idx]
        objfaces = self.split_objects[obj]["faces"]
        return np.array(objfaces).astype(np.int32)

    def get_obj_pose(self, idx):
        idx = self.get_dataidx(idx)
        trans = self.objtransforms[idx]
        return trans[:3, :3], trans[:3, 3] / 1000

    def get_obj_verts_trans(self, idx):
        idx = self.get_dataidx(idx)
        obj = self.objnames[idx]
        trans = self.objtransforms[idx]
        verts = self.split_objects[obj]["verts"]
        trans_verts = fhbutils.transform_obj_verts(verts, trans, self.cam_extr) / 1000
        return np.array(trans_verts).astype(np.float32)

    def get_jointvis(self, idx):
        return np.ones(self.joint_nb)

    def get_obj_verts_can(self, idx, no_center=False):
        idx = self.get_dataidx(idx)
        obj = self.objnames[idx]
        verts = self.split_objects[obj]["verts"]
        if no_center:
            return verts, None, None
        else:
            return meshnorm.center_vert_bbox(verts, scale=False)

    def get_obj_verts_can_rot(self, idx):
        idx = self.get_dataidx(idx)
        obj = self.objnames[idx]
        trans = self.objtransforms[idx]
        verts = self.split_objects[obj]["verts"]
        trans_verts = fhbutils.transform_obj_verts(verts, trans, self.cam_extr) / 1000
        return meshnorm.center_vert_bbox(trans_verts, scale=False)[0]

    def get_objverts2d(self, idx):
        objpoints3d = self.get_obj_verts_trans(idx)
        objpoints3d = objpoints3d * 1000
        hom_2d = np.array(self.cam_intr).dot(objpoints3d.transpose()).transpose()
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return verts2d.astype(np.float32)

    def get_joints3d(self, idx):
        idx = self.get_dataidx(idx)
        joints = self.joints3d[idx]
        return joints / 1000

    def get_joints2d(self, idx):
        idx = self.get_dataidx(idx)
        joints = self.joints2d[idx] * self.reduce_factor
        return joints

    def get_camintr(self, idx):
        idx = self.get_dataidx(idx)
        camintr = self.cam_intr
        return camintr.astype(np.float32)

    def get_sides(self, idx):
        idx = self.get_dataidx(idx)
        side = self.hand_sides[idx]
        return side

    def get_meta(self, idx):
        idx = self.get_dataidx(idx)
        meta = {"objname": self.objnames[idx]}
        return meta

    def get_center_scale(self, idx):
        if self.center_hand:
            joints2d = self.get_joints2d(idx)[0]
            center = handutils.get_annot_center(joints2d)
            scale = handutils.get_annot_scale(joints2d)
        else:
            center = np.array((480 / 2, 270 / 2))
            scale = 480
        return center, scale

    def get_objvis2d(self, idx):
        objvis = np.ones_like(self.get_objverts2d(idx))
        return objvis

    def get_handvis2d(self, idx):
        handvis = np.ones_like(self.get_handverts2d(idx))
        return handvis

    def __len__(self):
        if self.mode == "strong":
            return len(self.strongs)
        elif self.mode == "weak":
            return len(self.weaks)
        elif self.mode == "viz":
            return len(self.viz_list)
        else:
            return len(self.fulls)
