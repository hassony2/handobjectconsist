from collections import OrderedDict, defaultdict
from functools import lru_cache
import os
import pickle

import numpy as np
from tqdm import tqdm
import trimesh


@lru_cache(128)
def load_manoinfo(pkl_path):
    with open(pkl_path, "rb") as p_f:
        data = pickle.load(p_f)
    return data


def load_manofits(sample_infos, fit_root="assets/fhbhands_fits"):
    obj_paths = []
    metas = []
    for sample_info in sample_infos:
        hand_seq_path = os.path.join(
            fit_root, sample_info["subject"], sample_info["action_name"], sample_info["seq_idx"], "pkls.pkl"
        )
        mano_info = load_manoinfo(hand_seq_path)
        frame_name = f"{sample_info['frame_idx']:06d}.pkl"
        hand_info = mano_info[frame_name]
        metas.append(hand_info)
        hand_obj_path = os.path.join(
            fit_root,
            sample_info["subject"],
            sample_info["action_name"],
            sample_info["seq_idx"],
            "obj",
            f"{sample_info['frame_idx']:06d}.obj",
        )
        obj_paths.append(hand_obj_path)
    return obj_paths, metas


def load_objects(obj_root="data/fhbhands/Object_models", object_names=["juice"]):
    all_models = OrderedDict()
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, "{}_model".format(obj_name), "{}_model.ply".format(obj_name))
        mesh = trimesh.load(obj_path)
        if obj_name == "juice":
            obj_name = "juice_bottle"
        all_models[obj_name] = {"verts": np.array(mesh.vertices), "faces": np.array(mesh.faces)}
    return all_models


def load_object_infos(seq_root="data/fhbhands/Object_6D_pose_annotation_v1_1"):
    subjects = os.listdir(seq_root)
    annots = {}
    clip_lengths = {}
    for subject in subjects:
        subject_dict = {}
        subj_path = os.path.join(seq_root, subject)
        actions = os.listdir(subj_path)
        clips = 0
        for action in actions:
            object_name = "_".join(action.split("_")[1:])
            action_path = os.path.join(subj_path, action)
            seqs = os.listdir(action_path)
            clips += len(seqs)
            for seq in seqs:
                seq_path = os.path.join(action_path, seq, "object_pose.txt")
                with open(seq_path, "r") as seq_f:
                    raw_lines = seq_f.readlines()
                for raw_line in raw_lines:
                    line = raw_line.strip().split(" ")
                    frame_idx = int(line[0])
                    trans_matrix = np.array(line[1:]).astype(np.float32)
                    trans_matrix = trans_matrix.reshape(4, 4).transpose()
                    subject_dict[(action, seq, frame_idx)] = (object_name, trans_matrix)
        clip_lengths[subject] = clip_lengths
        annots[subject] = subject_dict
    return annots


def get_seq_map(sample_infos, fraction=1):
    seq_map = defaultdict(list)
    inv_seq_map = defaultdict()
    closeseqmap = defaultdict(dict)
    spacing = int(1 / fraction)
    cur_sample = sample_infos[0]
    cur_key = (cur_sample["subject"], cur_sample["action_name"], cur_sample["seq_idx"])
    idx_count = 0
    seq_count = 0
    previous = 0  # Keep track of idx of previous annotated frame
    stack_next = []  # Keep track of idxs that need to get assigned a next frame
    strong = []
    weak = []
    for sample_idx, sample_info in enumerate(sample_infos):
        next_key = (sample_info["subject"], sample_info["action_name"], sample_info["seq_idx"])
        if next_key != cur_key:
            # Get back last frame for sequence and mark as strong
            if fraction != 1 and len(weak):
                last_idx = weak.pop()
                strong.append(last_idx)
                closeseqmap.pop(last_idx)
            if fraction != 1 and len(stack_next):
                stack_next.pop()

            # Reinitialize sequence counter
            seq_count = 0
            previous = idx_count

            # Add last_idx to end of previous sequence for accumulated indices
            if fraction != 1:
                empty_stack_next(stack_next, idx_count - 1, closeseqmap)
                stack_next = []
            cur_key = next_key
            assert sample_info["frame_idx"] == 0
        if fraction != 1:
            if seq_count % spacing == 0:
                previous = idx_count
                empty_stack_next(stack_next, idx_count, closeseqmap)
                stack_next = []
            else:
                stack_next.append(idx_count)
                closeseqmap[idx_count]["previous"] = previous
        if (
            next_key != cur_key
            or seq_count % spacing == 0
            or (sample_idx == len(sample_infos) - 1)
            or (fraction == 1)
        ):
            strong.append(idx_count)
        else:
            weak.append(idx_count)

        full_key = (
            sample_info["subject"],
            sample_info["action_name"],
            sample_info["seq_idx"],
            sample_info["frame_idx"],
        )
        seq_map[cur_key].append((*full_key, idx_count))
        inv_seq_map[full_key] = idx_count
        idx_count += 1
        seq_count += 1
    if fraction != 1:
        empty_stack_next(stack_next, idx_count - 1, closeseqmap)
        if idx_count - 1 in closeseqmap:
            closeseqmap.pop(idx_count - 1)
        distances = [abs(key - val["closest"]) for key, val in closeseqmap.items()]
        assert min(distances) == 1
        assert len(weak) + len(strong) == len(sample_infos)
        assert not len((set(weak) | set(strong)) - set(range(len(sample_infos))))
    for strong_idx in strong:
        closeseqmap[strong_idx] = {"closest": strong_idx, "previous": strong_idx, "next": strong_idx}
    return dict(seq_map), dict(inv_seq_map), dict(closeseqmap), strong, weak


def empty_stack_next(stack, idx_count, closeseqmap):
    """
    Assign correct final frames and distances to accumulated indices
    since last anchor
    """
    for cand_next in stack:
        closeseqmap[cand_next]["next"] = idx_count
        dist_next = abs(cand_next - idx_count)
        dist_prev = abs(closeseqmap[cand_next]["previous"] - cand_next)
        if dist_prev > dist_next:
            closeseqmap[cand_next]["closest"] = closeseqmap[cand_next]["next"]
        else:
            closeseqmap[cand_next]["closest"] = closeseqmap[cand_next]["previous"]


def get_action_train_test(lines_raw, subjects_info):
    """
    Returns dicts of samples where key is
        subject: name of subject
        action_name: action class
        action_seq_idx: idx of action instance
        frame_idx
    and value is the idx of the action class
    """
    all_infos = []
    test_split = False
    test_samples = {}
    train_samples = {}
    for line in lines_raw[1:]:
        if line.startswith("Test"):
            test_split = True
            continue
        subject, action_name, action_seq_idx = line.split(" ")[0].split("/")
        action_idx = line.split(" ")[1].strip()  # Action classif index
        frame_nb = int(subjects_info[subject][(action_name, action_seq_idx)])
        for frame_idx in range(frame_nb):
            sample_info = (subject, action_name, action_seq_idx, frame_idx)
            if test_split:
                test_samples[sample_info] = action_idx
            else:
                train_samples[sample_info] = action_idx
            all_infos.append(sample_info)
    test_nb = len(np.unique(list((sub, act_n, act_seq) for (sub, act_n, act_seq, _) in test_samples), axis=0))
    assert test_nb == 575, "Should get 575 test samples, got {}".format(test_nb)
    train_nb = len(
        np.unique(list((sub, act_n, act_seq) for (sub, act_n, act_seq, _) in train_samples), axis=0)
    )
    # 600 - 1 Subject5/use_flash/6 discarded sample
    assert train_nb == 600 or train_nb == 599, "Should get 599 train samples, got {}".format(train_nb)
    assert len(test_samples) + len(train_samples) == len(all_infos)
    return train_samples, test_samples, all_infos


def transform_obj_verts(verts, trans, cam_extr):
    verts = verts * 1000
    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    trans_verts = trans.dot(hom_verts.T).T
    trans_verts = cam_extr.dot(trans_verts.transpose()).transpose()[:, :3]
    return trans_verts


def get_skeletons(skeleton_root, subjects_info, use_cache=True):
    cache_path = os.path.join("data/cache/fhb/skels.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and use_cache:
        with open(cache_path, "rb") as p_f:
            skelet_dict = pickle.load(p_f)
        print("Loaded fhb skel info from {}".format(cache_path))
    else:
        skelet_dict = defaultdict(dict)
        for subject, samples in tqdm(subjects_info.items(), desc="subj"):
            for (action, seq_idx) in tqdm(samples, desc="sample"):
                skeleton_path = os.path.join(skeleton_root, subject, action, seq_idx, "skeleton.txt")
                skeleton_vals = np.loadtxt(skeleton_path)
                if len(skeleton_vals):
                    assert np.all(
                        skeleton_vals[:, 0] == list(range(skeleton_vals.shape[0]))
                    ), "row idxs should match frame idx failed at {}".format(skeleton_path)
                    skelet_dict[subject][(action, seq_idx)] = skeleton_vals[:, 1:].reshape(
                        skeleton_vals.shape[0], 21, -1
                    )
                else:
                    # Handle sequences of size 0
                    skelet_dict[subject, action, seq_idx] = skeleton_vals
        with open(cache_path, "wb") as p_f:
            pickle.dump(skelet_dict, p_f)
    return skelet_dict
