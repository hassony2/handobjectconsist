from collections import defaultdict
import torch

from libyana.renderutils import textutils, catmesh
from matplotlib import cm

from meshreg.neurender import renderer
from meshreg.datasets.queries import BaseQueries, TransQueries
from meshreg.visualize import consistdisplay, rotateverts
from meshreg.visualize import samplevis
from meshreg.models import manoutils


def render(
    verts,
    faces,
    input_res,
    camintrs=None,
    colors=None,
    fill_back=True,
    near=0.05,
    far=2,
    crop_to_img=True,
    bg_color=None,
):
    """
    Given vertices, faces, resolution input_res of image and camera
    intrinsic parameters, render the corresponding image using
    neural_renderer as a backend
    """
    max_size = max(input_res)
    neurenderer = renderer.Renderer(
        image_size=max_size,
        R=torch.eye(3).unsqueeze(0).cuda(),
        t=torch.zeros(1, 3).cuda(),
        K=camintrs,
        orig_size=max_size,
        anti_aliasing=False,
        fill_back=fill_back,
        near=near,
        far=far,
        no_light=False,
        light_intensity_ambient=0.8,
    )
    if colors is None:
        colors = torch.ones_like(verts)
    textures = textutils.batch_vertex_textures(faces, colors[:, :, :3])

    renderout = neurenderer(verts, faces, textures)
    # Crop to original image dimensions
    render_rgb = renderout["rgb"]
    render_alpha = renderout["alpha"]
    if crop_to_img:
        render_rgb = render_rgb[:, :, : input_res[1], : input_res[0]]
        render_alpha = render_alpha[:, : input_res[1], : input_res[0]]
    if bg_color is not None:
        render_rgb = (render_rgb * render_alpha) + bg_color * (1 - render_alpha) * torch.ones_like(render_rgb)
    render = torch.cat([render_rgb, render_alpha.unsqueeze(1)], 1)
    return render.permute(0, 2, 3, 1)


def comp_render(
    sample,
    all_results,
    fill_back=True,
    near=0.05,
    far=2,
    modes=("all"),
    rotate=True,
    crop_to_img=False,
    max_val=0.1,
):
    images = sample[TransQueries.IMAGE].permute(0, 2, 3, 1).cpu() + 0.5
    camintrs = sample[TransQueries.CAMINTR].cuda()
    hand_faces, _ = manoutils.get_closed_faces()
    batch_size = images.shape[0]
    hand_faces_b = hand_faces.unsqueeze(0).repeat(batch_size, 1, 1).long().cuda()
    all_hand_verts = []
    all_obj_verts = []
    all_obj_faces = []
    for results in all_results:
        hand_verts = samplevis.get_check_none(results, "recov_handverts3d", cpu=False)
        obj_verts = samplevis.get_check_none(results, "recov_objverts3d", cpu=False)
        obj_faces = samplevis.get_check_none(sample, BaseQueries.OBJFACES, cpu=False).long()
        all_hand_verts.append(hand_verts)
        all_obj_verts.append(obj_verts)
        all_obj_faces.append(obj_faces)

    obj_verts_gt = samplevis.get_check_none(sample, BaseQueries.OBJVERTS3D, cpu=False)
    hand_verts_gt = samplevis.get_check_none(sample, BaseQueries.HANDVERTS3D, cpu=False)
    hand_colors_gt = consistdisplay.get_verts_colors(hand_verts_gt, [0.3, 0.3, 1])
    obj_colors_gt = consistdisplay.get_verts_colors(obj_verts_gt, [1, 0.3, 0.3])
    all_colors_gt = torch.cat([hand_colors_gt, obj_colors_gt], 1)
    # Render
    jet_cmap = cm.get_cmap("jet")
    all_obj_errs = torch.stack([pred_vert - obj_verts_gt for pred_vert in all_obj_verts]).norm(2, -1)
    all_hand_errs = torch.stack([pred_vert - hand_verts_gt for pred_vert in all_hand_verts]).norm(2, -1)

    all_obj_colors = all_obj_errs / max_val
    all_hand_colors = all_hand_errs / max_val
    cmap_objs = jet_cmap(all_obj_colors.cpu().numpy())
    cmap_hands = jet_cmap(all_hand_colors.cpu().numpy())
    obj_colors = obj_verts_gt.new(cmap_objs)
    hand_colors = obj_verts_gt.new(cmap_hands)
    all_colors = torch.cat([hand_colors, obj_colors], 2)
    input_res = (images.shape[2], images.shape[1])
    all_render_res = defaultdict(list)
    all_verts_gt, all_faces, _ = catmesh.batch_cat_meshes(
        [hand_verts_gt, obj_verts_gt], [hand_faces_b, obj_faces]
    )
    # Render ground truth meshes
    for mode in modes:
        if mode == "all":
            render_res = render(
                all_verts_gt,
                all_faces,
                input_res,
                camintrs=camintrs,
                colors=all_colors_gt,
                near=near,
                far=far,
                fill_back=fill_back,
                crop_to_img=crop_to_img,
            )
            if rotate:
                rotated_verts = rotateverts.rotate_verts(all_verts_gt)
                renderot_res = render(
                    rotated_verts,
                    all_faces,
                    input_res,
                    camintrs=camintrs,
                    colors=all_colors_gt,
                    near=near,
                    far=far,
                    fill_back=fill_back,
                    crop_to_img=crop_to_img,
                )
        elif mode == "hand":
            render_res = render(
                hand_verts_gt,
                hand_faces_b,
                input_res,
                camintrs=camintrs,
                colors=hand_colors_gt,
                near=near,
                far=far,
                fill_back=fill_back,
                crop_to_img=crop_to_img,
            )
        elif mode == "obj":
            render_res = render(
                obj_verts_gt,
                obj_faces,
                input_res,
                camintrs=camintrs,
                colors=obj_colors_gt,
                near=near,
                far=far,
                fill_back=fill_back,
                crop_to_img=crop_to_img,
            )
        all_render_res[mode].append(render_res.cpu())
        all_render_res[f"{mode}_rotated"].append(renderot_res.cpu())
    # Render predictions
    for model_idx, (hand_verts, obj_verts, obj_faces) in enumerate(
        zip(all_hand_verts, all_obj_verts, all_obj_faces)
    ):
        all_verts, all_faces, _ = catmesh.batch_cat_meshes([hand_verts, obj_verts], [hand_faces_b, obj_faces])
        for mode in modes:
            if mode == "all":
                render_res = render(
                    all_verts,
                    all_faces,
                    input_res,
                    camintrs=camintrs,
                    colors=all_colors[model_idx],
                    near=near,
                    far=far,
                    fill_back=fill_back,
                    crop_to_img=crop_to_img,
                )
                if rotate:
                    rotated_verts = rotateverts.rotate_verts(all_verts)
                    renderot_res = render(
                        rotated_verts,
                        all_faces,
                        input_res,
                        camintrs=camintrs,
                        colors=all_colors[model_idx],
                        near=near,
                        far=far,
                        fill_back=fill_back,
                        crop_to_img=crop_to_img,
                    )
            elif mode == "obj":
                render_res = render(
                    obj_verts,
                    obj_faces,
                    input_res,
                    camintrs=camintrs,
                    colors=obj_colors[model_idx],
                    near=near,
                    far=far,
                    fill_back=fill_back,
                    crop_to_img=crop_to_img,
                )
            elif mode == "hand":
                render_res = render(
                    hand_verts,
                    hand_faces_b,
                    input_res,
                    camintrs=camintrs,
                    # colors=hand_colors[model_idx],
                    colors=hand_colors_gt,
                    near=near,
                    far=far,
                    fill_back=fill_back,
                    crop_to_img=crop_to_img,
                )
            all_render_res[f"{mode}"].append(render_res.cpu())
            all_render_res[f"{mode}_rotated"].append(renderot_res.cpu())
    return all_render_res, cmap_objs


def hand_obj_render(
    sample,
    results,
    hand_colors=(0.3, 0.3, 1),
    obj_colors=(1, 0.3, 0.3),
    fill_back=True,
    near=0.05,
    far=2,
    modes=("all"),
    rotate=True,
    crop_to_img=True,
):
    images = sample[TransQueries.IMAGE].permute(0, 2, 3, 1).cpu() + 0.5
    camintrs = sample[TransQueries.CAMINTR].cuda()
    batch_size = images.shape[0]
    hand_verts = samplevis.get_check_none(results, "recov_handverts3d", cpu=False)
    obj_verts = samplevis.get_check_none(results, "recov_objverts3d", cpu=False)
    obj_faces = samplevis.get_check_none(sample, BaseQueries.OBJFACES, cpu=False)

    # Render
    render_results = {}
    if hand_verts is not None:
        # Initialize faces and textures, TODO use closed hand compatible with model vertices
        hand_faces, _ = manoutils.get_closed_faces()
        hand_faces_b = hand_faces.unsqueeze(0).repeat(batch_size, 1, 1).long().cuda()
        input_res = (images.shape[2], images.shape[1])
        hand_colors = consistdisplay.get_verts_colors(hand_verts, [0.3, 0.3, 1])
        if "hand" in modes:
            render_res = render(
                hand_verts,
                hand_faces_b,
                input_res,
                camintrs=camintrs,
                colors=hand_colors,
                near=near,
                far=far,
                fill_back=fill_back,
                crop_to_img=crop_to_img,
            )
            render_results["hand"] = render_res
    if obj_verts is not None:
        obj_colors = consistdisplay.get_verts_colors(obj_verts, [1, 0.3, 0.3])
        obj_faces = obj_faces.long()
        if "obj" in modes:
            render_res = render(
                obj_verts,
                obj_faces,
                input_res,
                camintrs=camintrs,
                colors=obj_colors,
                near=near,
                far=far,
                fill_back=fill_back,
                crop_to_img=crop_to_img,
            )
            render_results["obj"] = render_res
    if obj_verts is not None and hand_verts is not None:
        colors = torch.cat([hand_colors, obj_colors], 1)
        all_verts, all_faces, _ = catmesh.batch_cat_meshes([hand_verts, obj_verts], [hand_faces_b, obj_faces])
        if "all" in modes:
            render_res = render(
                all_verts,
                all_faces,
                input_res,
                camintrs=camintrs,
                colors=colors,
                near=near,
                far=far,
                fill_back=fill_back,
                crop_to_img=crop_to_img,
            )
            render_results["all"] = render_res
            if rotate:
                rotated_verts = rotateverts.rotate_verts(all_verts)
                render_res = render(
                    rotated_verts,
                    all_faces,
                    input_res,
                    camintrs=camintrs,
                    colors=colors,
                    near=near,
                    far=far,
                    fill_back=fill_back,
                    crop_to_img=crop_to_img,
                )
                render_results["all_rotated"] = render_res
    return dict(render_results)
