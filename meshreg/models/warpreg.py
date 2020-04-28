import torch

from manopth import manolayer

from meshreg.neurender import renderer
from meshreg.optim import pyramidloss
from meshreg.models import warpbranch, manoutils


class WarpRegNet(torch.nn.Module):
    def __init__(
        self,
        image_size,
        model,
        fill_back=True,
        use_backward=True,
        lambda_data=1,
        lambda_consist=1,
        criterion="l1",
        consist_scale=1,
        first_only=True,
        gt_refs=True,
        progressive_consist=True,
        progressive_steps=1000,
    ):
        super().__init__()
        self.fill_back = fill_back
        self.use_backward = use_backward
        max_size = max(image_size)
        self.image_size = image_size
        self.lambda_data = lambda_data
        self.lambda_consist = lambda_consist
        self.consist_scale = consist_scale
        self.criterion = pyramidloss.PyramidCriterion(criterion)
        self.first_only = first_only
        self.progressive_consist = progressive_consist
        self.progressive_steps = progressive_steps
        self.gt_refs = gt_refs
        self.step_count = 0
        neurenderer = renderer.Renderer(
            image_size=max_size,
            R=torch.eye(3).unsqueeze(0).cuda(),
            t=torch.zeros(1, 3).cuda(),
            K=torch.ones(1, 3, 3).cuda(),
            orig_size=max_size,
            anti_aliasing=False,
            fill_back=fill_back,
            near=0.1,
            no_light=True,
            light_intensity_ambient=0.8,
        )
        self.renderer = neurenderer
        self.model = model
        self.mano_layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root="assets/mano",
            center_idx=None,
            flat_hand_mean=True,
        )
        closed_faces, hand_ignore_faces = manoutils.get_closed_faces()
        self.hand_ignore_faces = hand_ignore_faces
        self.mano_layer.register_buffer("th_faces", closed_faces)
        self.fill_back = fill_back

    def warp_forward(self, samples, all_results):
        warp_loss, pair_results = warpbranch.forward(
            samples,
            all_results,
            self.mano_layer.th_faces,
            self.renderer,
            self.image_size,
            self.criterion,
            gt_refs=self.gt_refs,
            first_only=self.first_only,
            hand_ignore_faces=self.hand_ignore_faces,
            use_backward=self.use_backward,
        )
        return warp_loss, pair_results

    def forward(self, batch):
        samples = batch["data"]
        all_results = []
        all_losses = []
        mesh_losses = []
        for sample in samples:
            loss, results, losses = self.model(sample)
            mesh_losses.append(loss)
            all_losses.append(losses)
            all_results.append(results)

        if "consist" in batch["supervision"]:
            warp_loss, pair_results = self.warp_forward(samples, all_results)
        else:
            pair_results = None

        aggregate_losses = {}
        for key in all_losses[0]:
            if all_losses[0][key] is not None:
                losses = [sample_loss[key] for sample_loss in all_losses]
                aggregate_losses[key] = torch.stack(losses).mean()
        loss = 0
        if self.progressive_consist:
            lambda_consist = min(
                self.lambda_consist * self.step_count / self.progressive_steps, self.lambda_consist
            )
            lambda_data = self.lambda_data - lambda_consist
        else:
            lambda_data = self.lambda_data
            lambda_consist = self.lambda_consist
        if "data" in batch["supervision"]:
            reg_loss = torch.cat(mesh_losses).mean()
            data_loss = lambda_data * reg_loss
            aggregate_losses["reg_loss"] = reg_loss
            loss += data_loss
        if "consist" in batch["supervision"]:
            # Add pose and shape regularization losses
            reg_loss = torch.mean(torch.stack([loss["mano_reg_loss"] for loss in all_losses]))
            data_loss = lambda_data * reg_loss
            loss += data_loss

            # Add consistency supervision
            consist_loss = lambda_consist * warp_loss
            loss += consist_loss
            aggregate_losses["warp_consist"] = warp_loss
            self.step_count += 1
        return loss, aggregate_losses, all_results, pair_results
