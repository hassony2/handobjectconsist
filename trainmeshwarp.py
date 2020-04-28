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

from meshreg.datasets import collate
from meshreg.loadutils import concatloader
from meshreg.datasets.queries import BaseQueries, TransQueries
from meshreg.models.meshregnet import MeshRegNet
from meshreg.netscripts import epochpassconsist, epochpass
from meshreg.netscripts import reloadmodel, get_dataset
from meshreg.models import warpreg


plt.switch_backend("agg")

extend_queries = [
    TransQueries.OBJVERTS3D,
    BaseQueries.OBJVERTS3D,
    BaseQueries.OBJCANVERTS,
    BaseQueries.OBJCANROTVERTS,
    TransQueries.OBJCANROTVERTS,
    BaseQueries.OBJVERTS2D,
    TransQueries.OBJVERTS2D,
    BaseQueries.OBJVIS2D,
    BaseQueries.OBJFACES,
]


def collate_fn(seq, extend_queries=extend_queries):
    return collate.seq_extend_collate(seq, extend_queries)


def main(args):
    setseeds.set_all_seeds(args.manual_seed)
    train_dat_str = "_".join(args.train_datasets)
    dat_str = f"{train_dat_str}_warp{args.consist_dataset}"
    now = datetime.now()
    exp_id = (
        f"checkpoints/{dat_str}_mini{args.mini_factor}/{now.year}_{now.month:02d}_{now.day:02d}/"
        f"{args.com}_frac{args.fraction:.1e}"
        f"_ld{args.lambda_data:.1e}_lc{args.lambda_consist:.1e}"
        f"crit{args.consist_criterion}sca{args.consist_scale}"
        f"opt{args.optimizer}_lr{args.lr}_crit{args.criterion2d}"
        f"_mom{args.momentum}_bs{args.batch_size}"
        f"_lmj3d{args.mano_lambda_joints3d:.1e}"
        f"_lmbeta{args.mano_lambda_shape:.1e}"
        f"_lmpr{args.mano_lambda_pose_reg:.1e}"
        f"_lmrj3d{args.mano_lambda_recov_joints3d:.1e}"
        f"_lmrw3d{args.mano_lambda_recov_verts3d:.1e}"
        f"_lov2d{args.obj_lambda_verts2d:.1e}_lov3d{args.obj_lambda_verts3d:.1e}"
        f"_lovr3d{args.obj_lambda_recov_verts3d:.1e}"
        f"cj{args.center_jittering}seed{args.manual_seed}"
        f"sample_nb{args.sample_nb}_spac{args.spacing}"
        f"csteps{args.progressive_consist_steps}"
    )
    if args.no_augm:
        exp_id = f"{exp_id}_no_augm"
    if args.no_consist_augm:
        exp_id = f"{exp_id}_noconsaugm"
    if args.block_rot:
        exp_id = f"{exp_id}_block_rot"
    if args.consist_gt_refs:
        exp_id = f"{exp_id}_gt_refs"

    # Initialize local checkpoint folder
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

    train_loader_nb = 0
    if len(args.train_datasets):
        train_loader_nb = train_loader_nb + len(args.train_datasets)
    if args.consist_dataset is not None:
        train_loader_nb = train_loader_nb + 1

    loaders = []
    if len(args.train_datasets) is not None:
        for train_split, dat_name in zip(args.train_splits, args.train_datasets):
            train_dataset, input_res = get_dataset.get_dataset(
                dat_name,
                split=train_split,
                meta={"version": args.version, "split_mode": "objects"},
                block_rot=args.block_rot,
                max_rot=args.max_rot,
                center_idx=args.center_idx,
                center_jittering=args.center_jittering,
                fraction=args.fraction,
                mini_factor=args.mini_factor,
                mode="strong",
                no_augm=args.no_augm,
                scale_jittering=args.scale_jittering,
                sample_nb=1,
                use_cache=args.use_cache,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=int(args.workers / train_loader_nb),
                drop_last=True,
                collate_fn=collate_fn,
            )
            loaders.append(train_loader)

    if args.consist_dataset is not None:
        consist_dataset, consist_input_res = get_dataset.get_dataset(
            args.consist_dataset,
            split=args.consist_split,
            meta={"version": args.version, "split_mode": "objects"},
            use_cache=args.use_cache,
            mini_factor=args.mini_factor,
            block_rot=args.block_rot,
            center_idx=args.center_idx,
            center_jittering=args.center_jittering,
            fraction=args.fraction,
            max_rot=args.max_rot,
            mode="full",
            no_augm=args.no_consist_augm,
            sample_nb=args.sample_nb,
            scale_jittering=args.scale_jittering,
            spacing=args.spacing,  # Otherwise black padding gets included on warps
        )
        print(f"Got consist dataset {args.consist_dataset} of size {len(consist_dataset)}")
        if input_res != consist_input_res:
            raise ValueError(
                f"train and consist dataset should have same input sizes"
                f"but got {input_res} and {consist_input_res}"
            )
        consist_loader = torch.utils.data.DataLoader(
            consist_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers / train_loader_nb),
            drop_last=True,
            collate_fn=collate_fn,
        )
        loaders.append(consist_loader)
    loader = concatloader.ConcatLoader(loaders)
    if args.eval_freq != -1:
        val_dataset, input_res = get_dataset.get_dataset(
            args.val_dataset,
            split=args.val_split,
            meta={"version": args.version, "split_mode": "objects"},
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
        mano_lambda_joints3d=args.mano_lambda_joints3d,
        mano_lambda_joints2d=args.mano_lambda_joints2d,
        mano_lambda_pose_reg=args.mano_lambda_pose_reg,
        mano_lambda_recov_joints3d=args.mano_lambda_recov_joints3d,
        mano_lambda_recov_verts3d=args.mano_lambda_recov_verts3d,
        mano_lambda_verts2d=args.mano_lambda_verts2d,
        mano_lambda_verts3d=args.mano_lambda_verts3d,
        mano_lambda_shape=args.mano_lambda_shape,
        mano_use_shape=args.mano_lambda_shape > 0,
        obj_lambda_verts2d=args.obj_lambda_verts2d,
        obj_lambda_verts3d=args.obj_lambda_verts3d,
        obj_lambda_recov_verts3d=args.obj_lambda_recov_verts3d,
        obj_trans_factor=args.obj_trans_factor,
        obj_scale_factor=args.obj_scale_factor,
        mano_fhb_hand="fhbhands" in args.train_datasets,
    )
    model.cuda()
    # Initalize model
    if args.resume is not None:
        opts = reloadmodel.load_opts(args.resume)
        model, epoch = reloadmodel.reload_model(args.resume, opts)
    else:
        epoch = 0
    if args.freeze_batchnorm:
        freeze.freeze_batchnorm_stats(model)  # Freeze batchnorm

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    pmodel = warpreg.WarpRegNet(
        input_res,
        model,
        lambda_consist=args.lambda_consist,
        lambda_data=args.lambda_data,
        criterion=args.consist_criterion,
        consist_scale=args.consist_scale,
        gt_refs=args.consist_gt_refs,
        progressive_steps=args.progressive_consist_steps,
        use_backward=args.consist_use_backward,
    )
    pmodel.cuda()
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
        save_dict, avg_meters, _ = epochpassconsist.epoch_pass(
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
            premodel=pmodel,
            loader_nb=train_loader_nb,
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
        if args.eval_freq != -1 and epoch_idx % args.eval_freq == 0:
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


if __name__ == "__main__":
    # Experiment parameters
    # torch.multiprocessing.set_start_method("forkserver")
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser()
    parser.add_argument("--com", default="debug/")
    parser.add_argument("--pyapt_id")

    # Dataset parameters
    valid_datasets = [
        "syntho3d",
        "syntho3dv2",
        "ho3dv2",
        "fhbhands",
    ]
    parser.add_argument(
        "--fraction",
        type=float,
        default=1,
        help="Fraction of dataset for which"
        "full supervision is used (the remaining frames being supervised with"
        "photometric consistency)",
    )
    parser.add_argument("--train_datasets", choices=valid_datasets, default=["fhbhands"], nargs="+")
    parser.add_argument("--consist_dataset", choices=valid_datasets, default="fhbhands")
    parser.add_argument("--val_dataset", choices=valid_datasets, default="fhbhands")
    parser.add_argument("--train_splits", default=["train"], nargs="+")
    parser.add_argument("--consist_split", default="train")
    parser.add_argument("--val_split", default="test")
    parser.add_argument("--mini_factor", type=float, default=1)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--no_augm", action="store_true")
    parser.add_argument("--no_consist_augm", action="store_true")
    parser.add_argument("--block_rot", action="store_true")
    parser.add_argument("--max_rot", default=0, type=float)
    parser.add_argument("--version", default=3, type=int)
    parser.add_argument("--center_idx", default=9, type=int)
    parser.add_argument("--center_jittering", type=float, default=0.1)
    parser.add_argument("--scale_jittering", type=float, default=0)

    # Model parameters
    parser.add_argument("--consist_mode", choices=["flow", "warp"], default="warp")
    parser.add_argument("--consist_scale", type=float, default=1)
    parser.add_argument("--consist_criterion", choices=["l1", "l2", "ssim"], default="l1")
    parser.add_argument("--progressive_consist_steps", type=int, default=1000)
    parser.add_argument("--consist_gt_refs", action="store_true")
    parser.add_argument(
        "--consist_use_backward",
        action="store_true",
        help="Supervise with photometric consistency"
        "in both directions (from annotated to unannotated and from"
        "unannotated to annotated)",
    )
    parser.add_argument("--obj_trans_factor", type=float, default=100)
    parser.add_argument("--obj_scale_factor", type=float, default=0.0001)
    parser.add_argument("--resume")
    parser.add_argument(
        "--sample_nb",
        default=2,
        type=int,
        help="Number of frames to compare"
        "If 2, compare unnanotated frame to closest annotated frame,"
        "if 3, compare to previous and next annotated frame",
    )
    parser.add_argument(
        "--spacing",
        default=0,
        type=int,
        help="If spacing is 0, get closest ground truth"
        "frame for reference, if positive, sample first frame from future"
        "if negative, sample first frame from the past.",
    )

    # Training params
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size during training and evaluation")
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of workers for multiprocessing" "during data loading"
    )
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr_decay_step", type=float, default=100)
    parser.add_argument("--lr_decay_gamma", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--freeze_batchnorm", action="store_true")
    parser.add_argument("--criterion2d", choices=["l2", "l1", "smooth_l1"], default="l2")

    # Weighting between different losses
    parser.add_argument("--lambda_consist", type=float, default=0.001)
    parser.add_argument("--lambda_data", type=float, default=0.999)
    parser.add_argument("--mano_lambda_verts3d", type=float, default=0)
    parser.add_argument("--mano_lambda_verts2d", type=float, default=0)
    parser.add_argument("--mano_lambda_recov_joints3d", type=float, default=0.5)
    parser.add_argument("--mano_lambda_recov_verts3d", type=float, default=0)
    parser.add_argument("--mano_lambda_pose_reg", type=float, default=5e-06)
    parser.add_argument("--mano_lambda_joints3d", type=float, default=0)
    parser.add_argument("--mano_lambda_joints2d", type=float, default=0)
    parser.add_argument("--mano_lambda_shape", type=float, default=5e-07)
    parser.add_argument("--obj_lambda_recov_verts3d", type=float, default=0.5)
    parser.add_argument("--obj_lambda_verts3d", type=float, default=0)
    parser.add_argument("--obj_lambda_verts2d", type=float, default=0)

    # Evaluation parameters
    parser.add_argument(
        "--eval_freq", type=int, default=10, help="How often to evaluate, 1 for always -1 for never"
    )

    # Display parameters
    parser.add_argument("--display_freq", type=int, default=500)
    parser.add_argument("--epoch_display_freq", type=int, default=20)

    parser.add_argument(
        "--snapshot", type=int, default=100, help="How often to save intermediate models (epochs)"
    )

    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)
