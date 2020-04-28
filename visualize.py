import argparse
import os

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from tqdm import tqdm

from libyana.exputils.argutils import save_args
from libyana.modelutils import freeze
from libyana.randomutils import setseeds

from meshreg.datasets import collate
from meshreg.datasets.queries import BaseQueries
from meshreg.models.meshregnet import MeshRegNet
from meshreg.netscripts import reloadmodel, get_dataset
from meshreg.neurender import fastrender
from meshreg.visualize import vizdemo


plt.switch_backend("agg")


def main(args):
    setseeds.set_all_seeds(args.manual_seed)
    # Initialize hosting
    exp_id = f"checkpoints/{args.dataset}/" f"{args.com}"

    # Initialize local checkpoint folder
    print(f"Saving info about experiment at {exp_id}")
    save_args(args, exp_id, "opt")
    render_folder = os.path.join(exp_id, "images")
    os.makedirs(render_folder, exist_ok=True)
    # Load models
    models = []
    for resume in args.resumes:
        opts = reloadmodel.load_opts(resume)
        model, epoch = reloadmodel.reload_model(resume, opts)
        models.append(model)
        freeze.freeze_batchnorm_stats(model)  # Freeze batchnorm

    dataset, input_res = get_dataset.get_dataset(
        args.dataset,
        split=args.split,
        meta={},
        mode=args.mode,
        use_cache=args.use_cache,
        no_augm=True,
        center_idx=opts["center_idx"],
        sample_nb=None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
        collate_fn=collate.meshreg_collate,
    )

    model = MeshRegNet(
        mano_center_idx=opts["center_idx"],
        mano_lambda_joints2d=opts["mano_lambda_joints2d"],
        mano_lambda_joints3d=opts["mano_lambda_joints3d"],
        mano_lambda_recov_joints3d=opts["mano_lambda_recov_joints3d"],
        mano_lambda_recov_verts3d=opts["mano_lambda_recov_verts3d"],
        mano_lambda_verts2d=opts["mano_lambda_verts2d"],
        mano_lambda_verts3d=opts["mano_lambda_verts3d"],
        mano_lambda_shape=opts["mano_lambda_shape"],
        mano_use_shape=opts["mano_lambda_shape"] > 0,
        mano_lambda_pose_reg=opts["mano_lambda_pose_reg"],
        obj_lambda_recov_verts3d=opts["obj_lambda_recov_verts3d"],
        obj_lambda_verts2d=opts["obj_lambda_verts2d"],
        obj_lambda_verts3d=opts["obj_lambda_verts3d"],
        obj_trans_factor=opts["obj_trans_factor"],
        obj_scale_factor=opts["obj_scale_factor"],
        mano_fhb_hand="fhbhands" in args.dataset,
    )

    fig = plt.figure(figsize=(10, 10))
    save_results = {}
    save_results["opt"] = dict(vars(args))
    # Put models on GPU and evaluation mode
    for model in models:
        model.cuda()
        model.eval()
    render_step = 0
    for batch in tqdm(loader):
        all_results = []
        # Compute model outputs
        with torch.no_grad():
            for model in models:
                _, results, _ = model(batch)
                all_results.append(results)

        # Densely render error map for the meshes
        for results in all_results:
            render_results, cmap_obj = fastrender.comp_render(
                batch, all_results, rotate=True, modes=("all", "obj", "hand"), max_val=args.max_val
            )

        for img_idx, img in enumerate(batch[BaseQueries.IMAGE]):
            # Get rendered results for current image
            render_ress = [res[img_idx] for res in render_results["all"]]
            renderot_ress = [res[img_idx] for res in render_results["all_rotated"]]
            # Initialize figure
            fig.clf()
            row_nb = len(models) + 1
            col_nb = 3
            axes = fig.subplots(row_nb, col_nb)
            # Display cmap
            cmap = cm.get_cmap("jet")
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            cax = fig.add_axes([0.27, 0.05, 0.5, 0.02])

            cb = matplotlib.colorbar.ColorbarBase(
                cax, cmap=cmap, norm=norm, ticks=[0, 1], orientation="horizontal"
            )
            cb.ax.set_xticklabels(["0", str(args.max_val * 100)])
            cb.set_label("3D mesh error (cm)")

            # Get masks for hand and object in current image
            obj_masks = [res.cpu()[img_idx][:, :].sum(2).numpy() for res in render_results["obj"]]
            hand_masks = [res.cpu()[img_idx][:, :].sum(2).numpy() for res in render_results["hand"]]
            # Compute bounding boxes of masks
            crops = [vizdemo.get_crop(render_res) for render_res in render_ress]
            rot_crops = [vizdemo.get_crop(renderot_res) for renderot_res in renderot_ress]
            # Get crop that encompasses the spatial extent of all results
            crop = vizdemo.get_common_crop(crops)
            rot_crop = vizdemo.get_common_crop(rot_crops)
            for model_idx, (render_res, renderot_res) in enumerate(zip(render_ress, renderot_ress)):
                # Draw input image with predicted contours in column 1
                ax = vizdemo.get_axis(axes, row_nb, col_nb, model_idx, 0)
                # Initialize white background and copy input image
                viz_img = 255 * img.new_ones(max(img.shape), max(img.shape), 3)
                viz_img[: img.shape[0], : img.shape[1], :3] = img

                # Clamp so that displayed image values are in [0, 255]
                render_res = render_res.clamp(0, 1)
                renderot_res = renderot_res.clamp(0, 1)

                obj_mask = obj_masks[model_idx] > 0
                hand_mask = hand_masks[model_idx] > 0
                # Draw hand and object contours
                contoured_img = vizdemo.draw_contours(viz_img.numpy(), hand_mask, color=(0, 210, 255))
                contoured_img = vizdemo.draw_contours(contoured_img, obj_mask, color=(255, 50, 50))
                ax.imshow(contoured_img[crop[0] : crop[2], crop[1] : crop[3]])
                # Image with rendering overlay
                ax = vizdemo.get_axis(axes, row_nb, col_nb, model_idx, 1)
                ax.set_title(args.model_names[model_idx])
                ax.imshow(viz_img[crop[0] : crop[2], crop[1] : crop[3]])
                ax.imshow(render_res[crop[0] : crop[2], crop[1] : crop[3]],)

                # Render rotated
                ax = vizdemo.get_axis(axes, row_nb, col_nb, model_idx, 2)
                ax.imshow(renderot_res[rot_crop[0] : rot_crop[2], rot_crop[1] : rot_crop[3]])
            fig.tight_layout()
            save_path = os.path.join(render_folder, f"render{render_step:06d}.png")
            fig.savefig(save_path)
            print(f"Saved demo visualization at {save_path}")
            render_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--com", default="debug/", help="Prefix for experimental results")
    parser.add_argument("--manual_seed", default=1, help="Fixed random seed")

    # Dataset params
    parser.add_argument("--dataset", default="fhbhands")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--mode", default="viz", help="[viz|full], 'viz' for selected dataset samples, 'full' for random ones"
    )
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument(
        "--max_val", default=0.1, type=float, help="Max value (in meters) for colormap error range"
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--workers", default=0, type=int)

    # Model parameters
    parser.add_argument(
        "--resumes",
        nargs="+",
        default=[
            "releasemodels/fphab/hands_and_objects/checkpoint_200.pth.tar",
            "releasemodels/fphab/warp_effect/no_consist/frac_6.3e-03/checkpoint_1200.pth.tar",
            "releasemodels/fphab/warp_effect/with_consist/frac_6.3e-03/checkpoint_1200.pth.tar",
        ],
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "Supervised data: 100%",
            "Supervised data: 0.65%",
            "Supervised data: 0.65% + photometric consistency",
        ],
    )

    # Loss parameters
    parser.add_argument("--criterion2d", choices=["l2", "l1", "smooth_l1"], default="l2")
    parser.add_argument(
        "--display_freq", type=int, default=500, help="How often to generate visualizations (training steps)"
    )

    args = parser.parse_args()
    args.model_names = ["Ground Truth"] + args.model_names
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)
