import os
import pickle

from tqdm import tqdm
import torch

from libyana.evalutils.avgmeter import AverageMeters
from libyana.evalutils.zimeval import EvalUtil

from meshreg.datasets.queries import BaseQueries
from meshreg.netscripts import evaluate
from meshreg.visualize import samplevis, evalvis
from meshreg.visualize import consistdisplay


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
    freeze_batchnorm=True,
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
    if train and not freeze_batchnorm:
        model.train()
    else:
        model.eval()
    avg_meters = AverageMeters()
    render_step = 0
    # Loop over dataset
    for batch_idx, batch in enumerate(tqdm(loader)):
        # Compute outputs and losses
        if train:
            loss, results, losses = model(batch)
        else:
            with torch.no_grad():
                loss, results, losses = model(batch)
        # Optimize model if needed
        if train:
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {losses} became nan!")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for loss_name, loss_val in losses.items():
            if loss_val is not None:
                avg_meters.add_loss_value(loss_name, loss_val.mean().item())
        evaluate.feed_avg_meters(avg_meters, batch, results)

        # Visualize outputs
        if batch_idx % display_freq == 0 and epoch % epoch_display_freq == 0:
            img_filepath = f"{prefix}_epoch{epoch:04d}_batch{batch_idx:06d}.png"
            save_img_path = os.path.join(img_folder, img_filepath)
            samplevis.sample_vis(batch, results, fig=fig, save_img_path=save_img_path)
        evaluate.feed_evaluators(evaluators, batch, results)
    if lr_decay_gamma and scheduler is not None:
        scheduler.step()
    save_dict = {}
    for loss_name, avg_meter in avg_meters.average_meters.items():
        save_dict[loss_name] = {}
        loss_val = avg_meter.avg
        save_dict[loss_name][prefix] = loss_val
    evaluator_results = evaluate.parse_evaluators(evaluators)
    for eval_name, eval_res in evaluator_results.items():
        for met in ["epe_mean", "auc"]:
            loss_name = f"{eval_name}_{met}"
            # Filter nans
            if eval_res[met] == eval_res[met]:
                save_dict[loss_name] = {}
                save_dict[loss_name][prefix] = eval_res[met]
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
