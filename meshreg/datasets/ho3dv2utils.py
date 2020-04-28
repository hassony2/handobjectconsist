from collections import defaultdict
import json
import os
import pickle

from tqdm import tqdm
import numpy as np


def get_papersplit_infos(split, root="data", trainval_idx=60000):
    """
    Args:
        split (str): HO3DV2 split in [train|trainval|val|test]
            train = trainval U(nion) val
        trainval_idx (int): How many frames to include in training split when
            using trainval/val/test split
    """
    if split in ["train", "trainval", "val"]:
        info_path = os.path.join(root, "train.txt")
        subfolder = "train"
    elif split == "test":
        info_path = os.path.join(root, "evaluation.txt")
        subfolder = "evaluation"
    with open(info_path, "r") as f:
        lines = f.readlines()
    # Get list [[seq, frame_idx], ...]
    seq_frames = [line.strip().split("/") for line in lines]
    if split == "trainval":
        seq_frames = seq_frames[:trainval_idx]
    elif split == "val":
        seq_frames = seq_frames[trainval_idx:]
    idxs = []
    seq_map = defaultdict(list)
    closeseq_map = {}
    all_sequences = []
    strongs = []
    weaks = []
    seq_counts = defaultdict(int)
    for idx_count, (seq, frame_idx) in enumerate(tqdm(seq_frames)):
        seq_folder = os.path.join(root, subfolder, seq)
        meta_folder = os.path.join(seq_folder, "meta")
        rgb_folder = os.path.join(seq_folder, "rgb")

        meta_path = os.path.join(meta_folder, f"{frame_idx}.pkl")
        with open(meta_path, "rb") as p_f:
            annot = pickle.load(p_f)
            if annot["handJoints3D"].size == 3:
                annot["handTrans"] = annot["handJoints3D"]
                annot["handJoints3D"] = annot["handJoints3D"][np.newaxis, :].repeat(21, 0)
                annot["handPose"] = np.zeros(48)
                annot["handBeta"] = np.zeros(10)
        img_path = os.path.join(rgb_folder, f"{frame_idx}.png")
        annot["img"] = img_path
        seq_map[seq].append(annot)
        closeseq_map[idx_count] = {"previous": idx_count, "next": idx_count, "closest": idx_count}
        idxs.append((seq, seq_counts[seq]))
        seq_counts[seq] += 1

        all_sequences.append([idx_count])
        strongs.append(idx_count)

    return all_sequences, seq_map, closeseq_map, strongs, weaks, idxs


def get_objectsplit_infos(seqs, root, subfolder="train", fraction=1):
    seq_map = defaultdict(list)
    idxs = []
    seq_lengths = {}
    for seq in tqdm(sorted(seqs), desc="seq"):
        seq_folder = os.path.join(root, subfolder, seq)
        meta_folder = os.path.join(seq_folder, "meta")
        rgb_folder = os.path.join(seq_folder, "rgb")

        img_nb = len(os.listdir(meta_folder))
        seq_lengths[seq] = img_nb
        for img_idx in tqdm(range(img_nb), desc="img"):
            meta_path = os.path.join(meta_folder, f"{img_idx:04d}.pkl")
            with open(meta_path, "rb") as p_f:
                annot = pickle.load(p_f)
            img_path = os.path.join(rgb_folder, f"{img_idx:04d}.png")
            annot["img"] = img_path
            seq_map[seq].append(annot)
            idxs.append((seq, img_idx))

    spacing = int(len(idxs) / int(fraction * len(idxs)))
    weaks = []
    strongs = []
    idx_count = 0
    previous = 0
    closeseq_map = {}
    all_sequences = []
    trace_seqs = []

    for seq in sorted(seqs):
        seq_stack = []
        trace_seq = []
        img_nb = seq_lengths[seq]
        for image_idx in range(img_nb):
            if (image_idx == 0) or (image_idx % spacing == 0) or (image_idx == img_nb - 1):
                strongs.append(idx_count)
                previous = idx_count
                closeseq_map[idx_count] = {"previous": idx_count, "next": idx_count, "closest": idx_count}
                for stack_idx in seq_stack:
                    closeseq_map[stack_idx]["next"] = idx_count
                    if abs(closeseq_map[stack_idx]["next"] - stack_idx) < abs(
                        closeseq_map[stack_idx]["previous"] - stack_idx
                    ):
                        closeseq_map[stack_idx]["closest"] = closeseq_map[stack_idx]["next"]
                    else:
                        closeseq_map[stack_idx]["closest"] = closeseq_map[stack_idx]["previous"]
                seq_stack.append(idx_count)
                all_sequences.append(seq_stack)
                seq_stack = []
                trace_seq.append(seq)
                trace_seqs.append(trace_seq)
                trace_seq = []
            else:
                closeseq_map[idx_count] = {"previous": previous}
                seq_stack.append(idx_count)
                weaks.append(idx_count)
                trace_seq.append(seq)
            idx_count += 1
    return all_sequences, seq_map, closeseq_map, strongs, weaks, idxs


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file for official ho3dv2 evaluation. """
    # make sure its only lists
    def roundall(rows):
        return [[round(val, 5) for val in row] for row in rows]

    xyz_pred_list = [roundall(x.tolist()) for x in xyz_pred_list]
    verts_pred_list = [roundall(x.tolist()) for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, "w") as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print(
        "Dumped %d joints and %d verts predictions to %s"
        % (len(xyz_pred_list), len(verts_pred_list), pred_out_path)
    )
