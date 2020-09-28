import argparse
from datetime import datetime
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import torch

from libyana.exputils.argutils import save_args
from libyana.modelutils import freeze

from meshreg.datasets import collate
from meshreg.netscripts import evalpass, reloadmodel, get_dataset


plt.switch_backend("agg")


def main(args):
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    # Initialize hosting
    dat_str = args.val_dataset
    now = datetime.now()
    exp_id = (
        f"checkpoints/{dat_str}_mini{args.mini_factor}/"
        f"{now.year}_{now.month:02d}_{now.day:02d}/"
        f"{args.com}_frac{args.fraction}_mode{args.mode}_bs{args.batch_size}_"
        f"objs{args.obj_scale_factor}_objt{args.obj_trans_factor}"
    )

    # Initialize local checkpoint folder
    save_args(args, exp_id, "opt")
    result_folder = os.path.join(exp_id, "results")
    os.makedirs(result_folder, exist_ok=True)
    pyapt_path = os.path.join(result_folder, f"{args.pyapt_id}__{now.strftime('%H_%M_%S')}")
    with open(pyapt_path, "a") as t_f:
        t_f.write(" ")

    val_dataset, input_size = get_dataset.get_dataset(
        args.val_dataset,
        split=args.val_split,
        meta={"version": args.version, "split_mode": "paper"},
        use_cache=args.use_cache,
        mini_factor=args.mini_factor,
        mode=args.mode,
        fraction=args.fraction,
        no_augm=True,
        center_idx=args.center_idx,
        scale_jittering=0,
        center_jittering=0,
        sample_nb=None,
        has_dist2strong=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
        collate_fn=collate.meshreg_collate,
    )

    opts = reloadmodel.load_opts(args.resume)
    model, epoch = reloadmodel.reload_model(args.resume, opts)
    if args.render_results:
        render_folder = os.path.join(exp_id, f"renders", f"epoch{epoch:04d}")
        os.makedirs(render_folder, exist_ok=True)
        print(f"Rendering to {render_folder}")
    else:
        render_folder = None
    img_folder = os.path.join(exp_id, "images", f"epoch{epoch:04d}")
    os.makedirs(img_folder, exist_ok=True)
    freeze.freeze_batchnorm_stats(model)  # Freeze batchnorm

    fig = plt.figure(figsize=(12, 4))
    save_results = {}
    save_results["opt"] = dict(vars(args))
    save_results["val_losses"] = []
    os.makedirs(args.json_folder, exist_ok=True)
    json_path = os.path.join(args.json_folder, f"{args.val_split}.json")
    evalpass.epoch_pass(
        val_loader,
        model,
        optimizer=None,
        scheduler=None,
        epoch=epoch,
        img_folder=img_folder,
        fig=fig,
        display_freq=args.display_freq,
        dump_results_path=json_path,
        render_folder=render_folder,
        render_freq=args.render_freq,
        true_root=args.true_root,
    )
    print(f"Saved results for split {args.val_split} to {json_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    # torch.multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser()
    parser.add_argument("--com", default="debug/")

    # Dataset params
    parser.add_argument("--val_dataset", choices=["ho3dv2"], default="ho3dv2")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--mini_factor", type=float, default=1)
    parser.add_argument("--max_verts", type=int, default=1000)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--synth", action="store_true")
    parser.add_argument("--version", default=3, type=int)
    parser.add_argument("--fraction", type=float, default=1)
    parser.add_argument("--mode", choices=["strong", "weak", "full"], default="strong")

    # Test options
    parser.add_argument("--dump_results", action="store_true")
    parser.add_argument("--render_results", action="store_true")
    parser.add_argument("--render_freq", type=int, default=10)

    # Model params
    parser.add_argument("--center_idx", default=9, type=int)
    parser.add_argument(
        "--true_root", action="store_true", help="Replace predicted wrist position with ground truth root"
    )
    parser.add_argument("--resume")

    # Training params
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--freeze_batchnorm", action="store_true")
    parser.add_argument("--pyapt_id")
    parser.add_argument("--criterion2d", choices=["l2", "l1", "smooth_l1"], default="l2")

    # Weighting
    parser.add_argument("--obj_trans_factor", type=float, default=1)
    parser.add_argument("--obj_scale_factor", type=float, default=1)

    # Evaluation params
    parser.add_argument("--mask_threshold", type=float, default=0.9)
    parser.add_argument("--json_folder", default="jsonres/res")

    # Weighting params
    parser.add_argument("--display_freq", type=int, default=100)
    parser.add_argument("--snapshot", type=int, default=50)

    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)
