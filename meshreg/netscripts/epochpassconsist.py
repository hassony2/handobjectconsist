import os
import pickle

from tqdm import tqdm
import torch

from libyana.evalutils.avgmeter import AverageMeters
from libyana.evalutils.zimeval import EvalUtil

from meshreg.visualize import evalvis, warpvis
from meshreg.netscripts import evaluate


def epoch_pass(
    loader,
    model,
    train=False,
    optimizer=None,
    scheduler=None,
    epoch=0,
    img_folder=None,
    fig=None,
    display_freq=10,
    epoch_display_freq=1,
    lr_decay_gamma=0,
    loader_nb=2,
    freeze_batchnorm=True,
    premodel=None,
):
    if train:
        prefix = "train"
    else:
        prefix = "val"
    evaluators = {
        # "joints2d_trans": EvalUtil(),
        "joints2d_base": EvalUtil(),
        "corners2d_base": EvalUtil(),
        "verts2d_base": EvalUtil(),
        "joints3d_cent": EvalUtil(),
        "joints3d": EvalUtil(),
    }
    consist_evaluators = {
        # "joints2d_trans": EvalUtil(),
        "joints2d_base": EvalUtil(),
        "corners2d_base": EvalUtil(),
        "verts2d_base": EvalUtil(),
        "joints3d_cent": EvalUtil(),
        "joints3d": EvalUtil(),
    }
    avg_meters = AverageMeters()
    consist_avg_meters = AverageMeters()
    if train and not freeze_batchnorm:
        model.train()
    else:
        model.eval()
    for batch_idx, batch in enumerate(tqdm(loader)):
        if batch_idx % loader_nb == 0:
            losses = []
        loss, all_losses, results, pair_results = premodel.forward(batch)
        losses.append(loss.flatten())
        if train and ((batch_idx % loader_nb) == (loader_nb - 1)):
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {all_losses} became nan!")
            optimizer.zero_grad()
            loss = torch.stack(losses).sum()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
        if "data" in batch["supervision"]:
            for loss_name, loss_val in all_losses.items():
                if loss_val is not None:
                    avg_meters.add_loss_value(loss_name, loss_val.mean().item())
            for sample, res in zip(batch["data"], results):
                evaluate.feed_avg_meters(avg_meters, sample, res)
                evaluate.feed_evaluators(evaluators, sample, res)
        elif "consist" in batch["supervision"]:
            for loss_name, loss_val in all_losses.items():
                if loss_val is not None:
                    consist_avg_meters.add_loss_value(loss_name, loss_val.mean().item())
            # Only compute metrics for weakly supervised frames (e.g. the first in the sequence)
            evaluate.feed_avg_meters(consist_avg_meters, batch["data"][0], results[0])
            evaluate.feed_evaluators(consist_evaluators, batch["data"][0], results[0])
        else:
            raise ValueError(f"Supervision {batch['supervision']} not in [data|consist]")
        if (batch_idx % display_freq) < loader_nb and epoch % epoch_display_freq == 0:
            img_prefix = f"{prefix}_epoch{epoch:04d}_batch{batch_idx:06d}"
            save_img_path = os.path.join(img_folder, img_prefix)
            warpvis.sample_vis(batch, results, pair_results, fig=fig, save_img_prefix=save_img_path)
    if lr_decay_gamma and scheduler is not None:
        scheduler.step()
    save_dict = {}
    for loss_name, avg_meter in avg_meters.average_meters.items():
        save_dict[loss_name] = {}
        loss_val = avg_meter.avg
        save_dict[loss_name][prefix] = loss_val
    for loss_name, avg_meter in consist_avg_meters.average_meters.items():
        if loss_name not in save_dict:
            save_dict[loss_name] = {}
        loss_val = avg_meter.avg
        save_dict[loss_name]["consist"] = loss_val
    evaluator_results = evaluate.parse_evaluators(evaluators)
    show_metrics = ["epe_mean"]  # "auc"
    for eval_name, eval_res in evaluator_results.items():
        for met in show_metrics:
            loss_name = f"{eval_name}_{met}"
            # Filter nans
            if eval_res[met] == eval_res[met]:
                save_dict[loss_name] = {}
                save_dict[loss_name][prefix] = eval_res[met]
    consist_evaluator_results = evaluate.parse_evaluators(consist_evaluators)
    for eval_name, eval_res in consist_evaluator_results.items():
        for met in show_metrics:
            loss_name = f"{eval_name}_{met}"
            # Filter nans
            if eval_res[met] == eval_res[met]:
                if loss_name not in save_dict:
                    save_dict[loss_name] = {}
                save_dict[loss_name]["consist"] = eval_res[met]
    img_filepath = f"{prefix}_epoch{epoch:04d}_eval.png"
    save_img_path = os.path.join(img_folder, img_filepath)
    # Filter out Nan pck curves
    evaluator_results = {
        eval_name: res for eval_name, res in evaluator_results.items() if res["epe_mean"] == res["epe_mean"]
    }
    evalvis.eval_vis(evaluator_results, save_img_path, fig=fig)
    pickle_path = save_img_path.replace(".png", ".pkl")
    with open(pickle_path, "wb") as p_f:
        pickle.dump(evaluator_results, p_f)
    return save_dict, avg_meters, evaluator_results
