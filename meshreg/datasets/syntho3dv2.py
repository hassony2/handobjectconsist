import os
import pickle

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from meshreg.datasets.queries import BaseQueries, get_trans_queries
from meshreg.datasets import ho3dfullutils
from libyana.meshutils import meshnorm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SynthHO3Dv2:
    def __init__(
        self,
        version=2,
        split="train",
        joint_nb=21,
        mini_factor=None,
        use_cache=False,
        root_palm=False,
        mode="obj",
        segment=False,
        apply_obj_transform=True,
        root="data/syntho3dv2",
    ):
        # Set cache path
        self.split = split
        root = os.path.join(root, f"v{version}", split)
        self.ho3d_root = "data/ho3dv2"
        self.has_dist2strong = False
        self.all_queries = [
            BaseQueries.IMAGE,
            BaseQueries.JOINTS2D,
            BaseQueries.JOINTS3D,
            BaseQueries.JOINTVIS,
            BaseQueries.OBJVERTS2D,
            BaseQueries.OBJVERTS3D,
            BaseQueries.OBJVIS2D,
            BaseQueries.OBJCANROTVERTS,
            BaseQueries.OBJFACES,
            BaseQueries.HANDVERTS2D,
            BaseQueries.HANDVERTS3D,
            BaseQueries.HANDVIS2D,
            BaseQueries.OBJCANVERTS,
            BaseQueries.SIDE,
            BaseQueries.CAMINTR,
        ]
        self.root_palm = root_palm
        self.mode = mode
        self.segment = segment
        self.apply_obj_transform = apply_obj_transform

        self.rgb_folder = os.path.join(root, "rgb")

        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        # Cache information
        self.use_cache = use_cache
        self.name = f"syntho3dv2_{version}"
        self.cache_folder = os.path.join("data", "cache", self.name)
        os.makedirs(self.cache_folder, exist_ok=True)
        self.mini_factor = mini_factor

        self.joint_nb = joint_nb

        self.prefix_template = "{:08d}"
        self.meta_folder = os.path.join(root, "meta")
        self.coord2d_folder = os.path.join(root, "coords2d")
        self.obj_meshes = ho3dfullutils.load_objects(os.path.join(self.ho3d_root, "modelsprocess"))

        # Define links on skeleton
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

        # Object info
        self.load_dataset()

    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, "{}.jpg".format(prefix))

        return image_path

    def load_dataset(self):
        pkl_path = "/sequoia/data1/yhasson/code/\
                    pose_3d/mano_render/mano/models/MANO_RIGHT_v1.pkl"
        if not os.path.exists(pkl_path):
            pkl_path = "../" + pkl_path

        cache_path = os.path.join(
            self.cache_folder, "{}_{}_mode_{}.pkl".format(self.split, self.mini_factor, self.mode)
        )
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, "rb") as cache_f:
                annotations = pickle.load(cache_f)
            print("Cached information for dataset {} loaded from {}".format(self.name, cache_path))
        else:
            idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(self.meta_folder))]

            if self.mini_factor:
                mini_nb = int(len(idxs) * self.mini_factor)
                idxs = idxs[:mini_nb]

            prefixes = [self.prefix_template.format(idx) for idx in idxs]
            print("Got {} samples for split {}, generating cache !".format(len(idxs), self.split))

            image_names = []
            all_joints2d = []
            all_joints3d = []
            hand_sides = []
            hand_poses = []
            hand_pcas = []
            hand_verts3d = []
            sample_ids = []
            cam_intrs = []
            cam_extrs = []
            obj_transforms = []
            meta_infos = []
            depth_infos = []
            for idx, prefix in enumerate(tqdm(prefixes)):
                meta_path = os.path.join(self.meta_folder, "{}.pkl".format(prefix))
                with open(meta_path, "rb") as meta_f:
                    meta_info = pickle.load(meta_f)
                image_path = self._get_image_path(prefix)
                image_names.append(image_path)
                cam_intrs.append(meta_info["cam_calib"])
                cam_extrs.append(meta_info["cam_extr"])
                all_joints2d.append(meta_info["coords_2d"])
                all_joints3d.append(meta_info["coords_3d"])
                hand_verts3d.append(meta_info["verts_3d"])
                hand_sides.append(meta_info["side"])
                hand_poses.append(meta_info["hand_pose"])
                hand_pcas.append(meta_info["pca_pose"])
                sample_id = meta_info["sample_id"]

                sample_ids.append(sample_id)
                obj_transforms.append(meta_info["affine_transform"])
                meta_info_full = {
                    "obj_scale": meta_info["obj_scale"],
                    "obj_class_id": meta_info["class_id"],
                    "obj_sample_id": meta_info["sample_id"],
                }
                meta_infos.append(meta_info_full)

            annotations = {
                "image_names": image_names,
                "joints2d": all_joints2d,
                "joints3d": all_joints3d,
                "hand_sides": hand_sides,
                "hand_poses": hand_poses,
                "hand_pcas": hand_pcas,
                "hand_verts3d": hand_verts3d,
                "sample_ids": sample_ids,
                "obj_transforms": obj_transforms,
                "meta_infos": meta_infos,
                "cam_intrs": cam_intrs,
                "cam_extrs": cam_extrs,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            print("Wrote cache for dataset {} to {}".format(self.name, cache_path))

        # Set dataset attributes
        all_objects = [obj[:-7].split("/")[-1].split("_")[0] for obj in annotations["sample_ids"]]
        selected_idxs = list(range(len(all_objects)))
        sample_ids = [annotations["sample_ids"][idx] for idx in selected_idxs]
        image_names = [annotations["image_names"][idx] for idx in selected_idxs]
        joints3d = [annotations["joints3d"][idx] for idx in selected_idxs]
        joints2d = [annotations["joints2d"][idx] for idx in selected_idxs]
        hand_sides = [annotations["hand_sides"][idx] for idx in selected_idxs]
        cam_intrs = [annotations["cam_intrs"][idx] for idx in selected_idxs]
        cam_extrs = [annotations["cam_extrs"][idx] for idx in selected_idxs]
        hand_pcas = [annotations["hand_pcas"][idx] for idx in selected_idxs]
        hand_verts3d = [annotations["hand_verts3d"][idx] for idx in selected_idxs]
        obj_transforms = [annotations["obj_transforms"][idx] for idx in selected_idxs]
        meta_infos = [annotations["meta_infos"][idx] for idx in selected_idxs]
        if "depth_infos" in annotations:
            has_depth_info = True
            depth_infos = [annotations["depth_infos"][idx] for idx in selected_idxs]
        else:
            has_depth_info = False
        if has_depth_info:
            self.depth_infos = depth_infos
        self.image_names = image_names
        self.joints2d = joints2d
        self.joints3d = joints3d
        self.hand_sides = hand_sides
        self.hand_pcas = hand_pcas
        self.cam_extrs = cam_extrs
        self.cam_intrs = cam_intrs
        self.hand_verts3d = hand_verts3d
        self.sample_ids = sample_ids
        self.obj_transforms = obj_transforms
        self.meta_infos = meta_infos
        # Initialize cache for center and scale in case objects are used
        self.center_scale_cache = {}

    def get_image(self, idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path)
        img = img.convert("RGB")
        return img

    def get_joints2d(self, idx):
        return self.joints2d[idx].transpose().astype(np.float32)

    def get_joints3d(self, idx):
        joints3d = self.joints3d[idx]
        if self.root_palm:
            # Replace wrist with palm
            verts3d = self.hand_verts3d[idx]
            palm = (verts3d[95] + verts3d[218]) / 2
            joints3d = np.concatenate([palm[np.newaxis, :], joints3d[1:]])
        # No hom coordinates needed because no translation
        assert np.linalg.norm(self.cam_extrs[idx][:, 3]) == 0, "extr camera should have no translation"

        joints3d = self.cam_extrs[idx][:3, :3].dot(joints3d.transpose()).transpose()
        return joints3d

    def get_hand_verts3d(self, idx):
        verts3d = self.hand_verts3d[idx]
        verts3d = self.cam_extrs[idx][:3, :3].dot(verts3d.transpose()).transpose()
        return verts3d

    def get_hand_verts2d(self, idx):
        verts3d = self.get_hand_verts3d(idx)
        return self.project(verts3d, idx)

    def get_objverts2d(self, idx):
        objpoints3d = self.get_obj_verts_trans(idx)
        return self.project(objpoints3d, idx)

    def project(self, points3d, idx):
        hom_2d = np.array(self.cam_intrs[idx]).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)

    def get_obj_verts_trans(self, idx):
        sample_id = self.sample_ids[idx]
        verts = self.obj_meshes[sample_id]["verts"]
        obj_scale = self.meta_infos[idx]["obj_scale"]
        verts = verts * obj_scale

        # Apply transforms
        if self.apply_obj_transform:
            obj_transform = self.obj_transforms[idx]
            hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
            trans_verts = obj_transform.dot(hom_verts.T).T[:, :3]
            trans_verts = self.cam_extrs[idx][:3, :3].dot(trans_verts.transpose()).transpose()
        else:
            trans_verts = verts
        return np.array(trans_verts).astype(np.float32)

    def get_obj_verts_can(self, idx, rescale=True):
        sample_id = self.sample_ids[idx]
        verts = self.obj_meshes[sample_id]["verts"]
        obj_scale = self.meta_infos[idx]["obj_scale"]
        verts = verts * obj_scale

        # Apply transforms
        verts = self.cam_extrs[idx][:3, :3].dot(verts.transpose()).transpose()
        if rescale:
            return meshnorm.center_vert_bbox(verts, scale=False)
        else:
            return verts, np.array([0, 0]), 1

    def get_obj_verts_can_rot(self, idx):
        verts, _, _ = self.get_obj_verts_can(idx)
        rot = self.obj_transforms[idx][:3, :3]
        verts = self.cam_extrs[idx][:3, :3].dot(verts.transpose()).transpose()
        verts = rot.dot(verts.transpose()).transpose()
        verts = self.cam_extrs[idx][:3, :3].dot(verts.transpose()).transpose()
        obj_verts = np.array(verts).astype(np.float32)
        return obj_verts

    def get_obj_faces(self, idx):
        sample_id = self.sample_ids[idx]
        objfaces = self.obj_meshes[sample_id]["faces"]
        objfaces = np.array(objfaces).astype(np.int16)
        return objfaces

    def get_sides(self, idx):
        return self.hand_sides[idx]

    def get_camintr(self, idx):
        return self.cam_intrs[idx].astype(np.float32)

    def get_center_scale(self, idx, scale_factor=2.2):
        center = np.array([320, 240])
        scale = 640
        return center, scale

    def get_jointvis(self, idx):
        return np.ones(self.joint_nb)

    def get_objvis2d(self, idx):
        objvis = np.ones_like(self.get_objverts2d(idx)[:, 0])
        return objvis

    def get_handvis2d(self, idx):
        handvis = np.ones_like(self.get_hand_verts2d(idx)[:, 0])
        return handvis

    def __len__(self):
        return len(self.image_names)
