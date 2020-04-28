import numpy as np

from meshreg.datasets import (
    handobjset,
    fhbhands,
    ho3dv2,
    syntho3dv2,
    syntho3d,
)


def get_dataset(
    dataset_name,
    split="train",
    no_augm=True,
    block_rot=False,
    center_idx=9,
    center_jittering=0,
    fraction=1,
    has_dist2strong=True,
    max_rot=np.pi,
    meta=None,
    mini_factor=1,
    mode="full",
    sample_nb=None,
    sides="right",
    spacing=2,
    scale_jittering=0,
    use_cache=False,
):
    if dataset_name == "ho3dv2":
        pose_dataset = ho3dv2.HO3DV2(
            split=split,
            split_mode=meta["split_mode"],
            root="data",
            mini_factor=mini_factor,
            fraction=fraction,
            mode=mode,
            like_v1=(meta["version"] == 1),
            full_sequences=False,
        )
        input_res = (640, 480)
    elif dataset_name == "syntho3d":
        pose_dataset = syntho3d.SynthHO3D(split="train", version=meta["version"], use_cache=use_cache)
        input_res = (640, 480)
    elif dataset_name == "syntho3dv2":
        pose_dataset = syntho3dv2.SynthHO3Dv2(split="train", version=meta["version"], use_cache=use_cache)
        input_res = (640, 480)
    elif dataset_name == "fhbhands":
        pose_dataset = fhbhands.FHBHands(
            split=split, use_cache=use_cache, mini_factor=mini_factor, fraction=fraction, mode=mode
        )
        input_res = (480, 270)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    sides = "right"
    dataset = handobjset.HandObjSet(
        pose_dataset=pose_dataset,
        train=not no_augm,
        block_rot=block_rot,
        max_rot=max_rot,
        center_idx=center_idx,
        queries=pose_dataset.all_queries,
        center_jittering=center_jittering,
        scale_jittering=scale_jittering,
        sides=sides,
        inp_res=input_res,
        sample_nb=sample_nb,
        spacing=spacing,
        has_dist2strong=has_dist2strong,
    )
    return dataset, input_res
