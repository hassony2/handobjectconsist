"""
Script to resize all FPHAB images to [480x270] dimension
"""
import argparse
import os
from os.path import join as osj
from os import listdir as osls

from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

fhb_root = "data/fhbhands"
fhb_rgb_src = osj(fhb_root, "Video_files")
fhb_rgb_dst = osj(fhb_root, "Video_files_480")


def convert(src, dst, out_size=(480, 270)):
    """
    Moves image at src to dst
    """
    dst_folder = os.path.dirname(dst)
    os.makedirs(dst_folder, exist_ok=True)
    if not os.path.exists(dst):
        img = Image.open(src)
        dest_img = img.resize(out_size, Image.BILINEAR)
        dest_img.save(dst)


def main(args):
    subjects = [f"Subject_{subj_idx}" for subj_idx in range(1, 7)]
    # Gather all frame paths to convert
    frame_pairs = []
    for subj in subjects:
        subj_path = osj(fhb_rgb_src, subj)
        actions = sorted(osls(subj_path))
        for action in actions:
            action_path = osj(subj_path, action)
            sequences = sorted(osls(action_path))
            for seq in sequences:
                seq_path = osj(action_path, seq, "color")
                frames = sorted(osls(seq_path))
                for frame in frames:
                    frame_path_src = osj(seq_path, frame)
                    frame_path_dst = osj(fhb_rgb_dst, subj, action, seq, "color", frame)
                    frame_pairs.append((frame_path_src, frame_path_dst))

    # Resize all images
    print(f"Launching conversion for {len(frame_pairs)}")
    Parallel(n_jobs=args.workers, verbose=5)(
        delayed(convert)(frame_pair[0], frame_pair[1]) for frame_pair in frame_pairs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=10, type=int)
    args = parser.parse_args()
    main(args)
