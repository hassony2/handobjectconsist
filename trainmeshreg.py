import argparse
from datetime import datetime
import os
import pickle

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from libyana.exputils.argutils import save_args
from libyana.exputils.monitoring import Monitor
from libyana.modelutils import modelio
from libyana.modelutils import freeze
from libyana.randomutils import setseeds
from libyana.datautils import concatloader

from meshreg.datasets import collate
from meshreg.models.meshregnet import MeshRegNet
from meshreg.netscripts import epochpass
from meshreg.netscripts import reloadmodel, get_dataset


plt.switch_backend("agg")


def main(args):
    setseeds.set_all_seeds(args.manual_seed)
    # Initialize hosting
    dat_str = "_".join(args.train_datasets)
    now = datetime.now()
    split_str = "_".join(args.train_splits)
    exp_id = (
        f"checkpoints/{dat_str}_{split_str}_mini{args.mini_factor}/{now.year}_{now.month:02d}_{now.day:02d}/"
        f"{args.com}_frac{args.fraction:.1e}"
        f"lr{args.lr}_mom{args.momentum}_bs{args.batch_size}_"
        f"_lmv2d{args.mano_lambda_verts2d:.1e}_lmv3d{args.mano_lambda_verts3d:.1e}"
        f"_lmj2d{args.mano_lambda_joints2d:.1e}_lmj3d{args.mano_lambda_joints3d:.1e}"
        f"_lmbeta{args.mano_lambda_shape:.1e}"
        f"_lmpr{args.mano_lambda_pose_reg:.1e}"
        f"_lmrj3d{args.mano_lambda_recov_joints3d:.1e}"
        f"_lmrw3d{args.mano_lambda_recov_verts3d:.1e}"
        f"_lov2d{args.obj_lambda_verts2d:.1e}_lov3d{args.obj_lambda_verts3d:.1e}"
        f"_lovr3d{args.obj_lambda_recov_verts3d:.1e}"
        f"_cj{args.center_jittering:.1e}_sj{args.scale_jittering}"
        f"seed{args.manual_seed}"
    )
    if args.no_augm:
        exp_id = f"{exp_id}_no_augm"
    if args.block_rot:
        exp_id = f"{exp_id}_block_rot"
    if args.freeze_batchnorm:
        exp_id = f"{exp_id}_fbn"

    # Initialize local checkpoint folder
    print(f"Saving experiment logs, models, and training curves and images at {exp_id}")
    save_args(args, exp_id, "opt")
    monitor = Monitor(exp_id, hosting_folder=exp_id)
    img_folder = os.path.join(exp_id, "images")
    os.makedirs(img_folder, exist_ok=True)
    result_folder = os.path.join(exp_id, "results")
    result_path = os.path.join(result_folder, "results.pkl")
    os.makedirs(result_folder, exist_ok=True)
    pyapt_path = os.path.join(result_folder, f"{args.pyapt_id}__{now.strftime('%H_%M_%S')}")
    with open(pyapt_path, "a") as t_f:
        t_f.write(" ")

    loaders = []
    if not args.evaluate:
        for train_split, dat_name in zip(args.train_splits, args.train_datasets):
            train_dataset, _ = get_dataset.get_dataset(
                dat_name,
                split=train_split,
                meta={"version": args.version, "split_mode": args.split_mode},
                use_cache=args.use_cache,
                mini_factor=args.mini_factor,
                no_augm=args.no_augm,
                block_rot=args.block_rot,
                max_rot=args.max_rot,
                center_idx=args.center_idx,
                scale_jittering=args.scale_jittering,
                center_jittering=args.center_jittering,
                fraction=args.fraction,
                mode="strong",
                sample_nb=None,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=int(args.workers / len(args.train_datasets)),
                drop_last=True,
                collate_fn=collate.meshreg_collate,
            )
            loaders.append(train_loader)
        loader = concatloader.ConcatDataloader(loaders)
    if args.evaluate or args.eval_freq != -1:
        val_dataset, _ = get_dataset.get_dataset(
            args.val_dataset,
            split=args.val_split,
            meta={"version": args.version, "split_mode": args.split_mode},
            use_cache=args.use_cache,
            mini_factor=args.mini_factor,
            no_augm=True,
            block_rot=args.block_rot,
            max_rot=args.max_rot,
            center_idx=args.center_idx,
            scale_jittering=args.scale_jittering,
            center_jittering=args.center_jittering,
            sample_nb=None,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            drop_last=False,
            collate_fn=collate.meshreg_collate,
        )

    model = MeshRegNet(
        mano_center_idx=args.center_idx,
        mano_lambda_joints2d=args.mano_lambda_joints2d,
        mano_lambda_joints3d=args.mano_lambda_joints3d,
        mano_lambda_recov_joints3d=args.mano_lambda_recov_joints3d,
        mano_lambda_recov_verts3d=args.mano_lambda_recov_verts3d,
        mano_lambda_verts2d=args.mano_lambda_verts2d,
        mano_lambda_verts3d=args.mano_lambda_verts3d,
        mano_lambda_shape=args.mano_lambda_shape,
        mano_use_shape=args.mano_lambda_shape > 0,
        mano_lambda_pose_reg=args.mano_lambda_pose_reg,
        obj_lambda_recov_verts3d=args.obj_lambda_recov_verts3d,
        obj_lambda_verts2d=args.obj_lambda_verts2d,
        obj_lambda_verts3d=args.obj_lambda_verts3d,
        obj_trans_factor=args.obj_trans_factor,
        obj_scale_factor=args.obj_scale_factor,
        mano_fhb_hand="fhbhands" in args.train_datasets,
    )
    model.cuda()
    # Initalize model
    if args.resume is not None:
        opts = reloadmodel.load_opts(args.resume)
        model, epoch = reloadmodel.reload_model(args.resume, opts)
        model.cuda()
        if args.evaluate:
            args.epochs = epoch + 1
    else:
        epoch = 0
    if args.freeze_batchnorm:
        freeze.freeze_batchnorm_stats(model)  # Freeze batchnorm

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    if args.resume is not None:
        reloadmodel.reload_optimizer(args.resume, optimizer)
    if args.lr_decay_gamma:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)
    fig = plt.figure(figsize=(10, 10))
    save_results = {}
    save_results["opt"] = dict(vars(args))
    save_results["train_losses"] = []
    save_results["val_losses"] = []
    for epoch_idx in tqdm(range(epoch, args.epochs), desc="epoch"):
        if not args.freeze_batchnorm:
            model.train()
        else:
            model.eval()
        if not args.evaluate:
            save_dict, avg_meters, _ = epochpass.epoch_pass(
                loader,
                model,
                train=True,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_idx,
                img_folder=img_folder,
                fig=fig,
                display_freq=args.display_freq,
                epoch_display_freq=args.epoch_display_freq,
                lr_decay_gamma=args.lr_decay_gamma,
            )
            monitor.log_train(epoch_idx + 1, {key: val.avg for key, val in avg_meters.average_meters.items()})
            monitor.metrics.save_metrics(epoch_idx + 1, save_dict)
            monitor.metrics.plot_metrics()
            save_results["train_losses"].append(save_dict)
            with open(result_path, "wb") as p_f:
                pickle.dump(save_results, p_f)
            modelio.save_checkpoint(
                {
                    "epoch": epoch_idx + 1,
                    "network": "correspnet",
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler,
                },
                is_best=True,
                checkpoint=exp_id,
                snapshot=args.snapshot,
            )
        if args.evaluate or (args.eval_freq != -1 and epoch_idx % args.eval_freq == 0):
            val_save_dict, val_avg_meters, _ = epochpass.epoch_pass(
                val_loader,
                model,
                train=False,
                optimizer=None,
                scheduler=None,
                epoch=epoch_idx,
                img_folder=img_folder,
                fig=fig,
                display_freq=args.display_freq,
                epoch_display_freq=args.epoch_display_freq,
                lr_decay_gamma=args.lr_decay_gamma,
            )

            save_results["val_losses"].append(val_save_dict)
            monitor.log_val(
                epoch_idx + 1, {key: val.avg for key, val in val_avg_meters.average_meters.items()}
            )
            monitor.metrics.save_metrics(epoch_idx + 1, val_save_dict)
            monitor.metrics.plot_metrics()
            if args.evaluate:
                print(val_save_dict)
                break


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser()
    parser.add_argument("--com", default="debug/", help="Prefix for experimental results")

    # Dataset params
    parser.add_argument(
        "--train_datasets",
        choices=["syntho3d", "syntho3dv2", "ho3dv2", "fhbhands"],
        default=["fhbhands"],
        nargs="+",
    )
    parser.add_argument(
        "--val_dataset", choices=["ho3dv2", "fhbhands"], default="fhbhands",
    )
    parser.add_argument("--train_splits", default=["train"], nargs="+")
    parser.add_argument("--val_split", default="test", choices=["test", "train", "val", "trainval"])
    parser.add_argument(
        "--split_mode",
        default="objects",
        choices=["objects", "paper"],
        help="HO3D possible splits, 'paper' for hand baseline, 'objects' for photometric consistency",
    )
    parser.add_argument(
        "--mini_factor", type=float, default=1, help="Work on fraction of the datase for debugging purposes"
    )
    parser.add_argument("--fraction", type=float, default=1, help="Fraction of data with full supervision")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--no_augm", action="store_true", help="Prevent all data augmenation")
    parser.add_argument("--block_rot", action="store_true", help="Prevent rotation during data augmentation")
    parser.add_argument("--max_rot", default=0, type=float, help="Max rotation for data augmentation")
    parser.add_argument("--version", default=1, type=int, help="Version of HO3D dataset to use")
    parser.add_argument("--center_idx", default=9, type=int)
    parser.add_argument(
        "--center_jittering", type=float, default=0.1, help="Controls magnitude of center jittering"
    )
    parser.add_argument(
        "--scale_jittering", type=float, default=0, help="Controls magnitude of scale jittering"
    )

    # Model parameters
    parser.add_argument("--resume")
    parser.add_argument(
        "--freeze_batchnorm",
        action="store_true",
        help="Freeze batchnorm layer values" "to ImageNet initialized values",
    )

    # Training parameters
    parser.add_argument("--evaluate", action="store_true", help="Only evaluation run")
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for multiprocessing")
    parser.add_argument("--pyapt_id")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr_decay_step", type=float, default=100)
    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        default=1,
        help="Learning rate decay" "factor, if 1, no decay is effectively applied",
    )
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0)

    # Loss parameters
    parser.add_argument("--criterion2d", choices=["l2", "l1", "smooth_l1"], default="l2")
    parser.add_argument(
        "--mano_lambda_verts3d", type=float, default=0, help="Weight for 3D vertices supervision"
    )
    parser.add_argument(
        "--mano_lambda_joints3d", type=float, default=0, help="Weight for 3D joints supervision, centered"
    )
    parser.add_argument(
        "--mano_lambda_recov_joints3d",
        type=float,
        default=0.5,
        help="Weight for 3D vertices supervision in camera space",
    )
    parser.add_argument(
        "--mano_lambda_recov_verts3d",
        type=float,
        default=0,
        help="Weight for 3D joints supervision, in camera space",
    )
    parser.add_argument(
        "--mano_lambda_joints2d", type=float, default=0, help="Weight for 2D joints supervision"
    )
    parser.add_argument(
        "--mano_lambda_verts2d", type=float, default=0, help="Weight for 2D vertices supervision"
    )
    parser.add_argument(
        "--mano_lambda_shape", type=float, default=5e-07, help="Weight for hand shape regularization"
    )
    parser.add_argument(
        "--mano_lambda_pose_reg", type=float, default=5e-06, help="Weight for hand pose regularization"
    )
    parser.add_argument(
        "--obj_lambda_recov_verts3d",
        type=float,
        default=0.5,
        help="Weight for object vertices supervision, in camera space",
    )
    parser.add_argument("--obj_lambda_verts3d", type=float, default=0)
    parser.add_argument("--obj_lambda_verts2d", type=float, default=0)
    parser.add_argument(
        "--obj_trans_factor", type=float, default=100, help="Multiplier for translation prediction"
    )
    parser.add_argument(
        "--obj_scale_factor", type=float, default=0.0001, help="Multiplier for scale prediction"
    )

    # Evaluation params
    parser.add_argument(
        "--eval_freq", type=int, default=10, help="How often to evaluate, 1 for always -1 for never"
    )

    # Weighting params
    parser.add_argument(
        "--display_freq", type=int, default=500, help="How often to generate visualizations (training steps)"
    )
    parser.add_argument(
        "--epoch_display_freq", type=int, default=1, help="How often to generate visualizations (epochs)"
    )
    parser.add_argument(
        "--snapshot", type=int, default=100, help="How often to save intermediate models (epochs)"
    )

    args = parser.parse_args()
    args.train_datasets = tuple(args.train_datasets)
    args.train_splits = tuple(args.train_splits)
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)
