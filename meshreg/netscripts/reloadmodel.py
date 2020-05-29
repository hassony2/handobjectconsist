import os
import pickle
import traceback
import warnings

import torch

from libyana.modelutils import modelio

from meshreg.models.meshregnet import MeshRegNet


def load_opts(resume_checkpoint):
    # Identify if folder or checkpoint is provided
    if resume_checkpoint.endswith(".pth"):
        resume_checkpoint = os.path.join(*resume_checkpoint.split("/")[:-1])
    opt_path = os.path.join(resume_checkpoint, "opt.pkl")
    with open(opt_path, "rb") as p_f:
        opts = pickle.load(p_f)
    return opts


def reload_model(resume_checkpoint, opts, optimizer=None, mano_lambda_pose_reg=None, mano_lambda_shape=None):
    opts = load_opts(resume_checkpoint)
    if mano_lambda_pose_reg is not None:
        opts["mano_lambda_pose_reg"] = mano_lambda_pose_reg
    if mano_lambda_shape is not None:
        opts["mano_lambda_shape"] = mano_lambda_shape
    if "train_dataset" in opts:
        opts["train_datasets"] = opts["train_dataset"]
    # Initialize model
    model = MeshRegNet(
        mano_lambda_verts2d=opts["mano_lambda_verts2d"],
        mano_lambda_verts3d=opts["mano_lambda_verts3d"],
        mano_lambda_joints3d=opts["mano_lambda_joints3d"],
        mano_lambda_recov_joints3d=opts["mano_lambda_recov_joints3d"],
        mano_lambda_recov_verts3d=opts["mano_lambda_recov_verts3d"],
        mano_lambda_joints2d=opts["mano_lambda_joints2d"],
        mano_lambda_shape=opts["mano_lambda_shape"],
        mano_use_shape=opts["mano_lambda_shape"] > 0,
        mano_lambda_pose_reg=opts["mano_lambda_pose_reg"],
        obj_lambda_recov_verts3d=opts["obj_lambda_recov_verts3d"],
        obj_lambda_verts2d=opts["obj_lambda_verts2d"],
        obj_lambda_verts3d=opts["obj_lambda_verts3d"],
        obj_trans_factor=opts["obj_trans_factor"],
        obj_scale_factor=opts["obj_scale_factor"],
        mano_fhb_hand="fhbhands" in opts["train_datasets"],
    )
    # model = torch.nn.DataParallel(model)
    if resume_checkpoint:
        start_epoch, _ = modelio.load_checkpoint(
            model, optimizer=optimizer, resume_path=resume_checkpoint, strict=False, as_parallel=False
        )
    else:
        start_epoch = 0
    return model, start_epoch


def reload_optimizer(resume_path, optimizer):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
    try:
        missing_states = set(optimizer.state_dict().keys()) - set(checkpoint["optimizer"].keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys in optimizer ! : {}".format(missing_states))
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError:
        traceback.print_exc()
        warnings.warn("Couldn' load optimizer from {}".format(resume_path))
