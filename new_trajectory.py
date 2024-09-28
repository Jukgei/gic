import taichi as ti
from utils.general_utils import safe_state
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from simulator import Simulator
import time
import torch
import os
import subprocess
from scene import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.reg_utils import mini_batch_knn
from argparse import ArgumentParser, Namespace
import numpy as np
import trimesh as tm
import torchvision
import json
from utils.system_utils import mkdir_p, write_particles
from tqdm import tqdm

# dataset.model_path
def load_pcd_file(file_path, iteration):
    # path = os.path.join(file_path, 'point_cloud_fix_pcd', f'iteration_{iteration}', 'point_cloud.ply')
    path = os.path.join(file_path, "mpm", "static_0.ply")
    pcd = tm.load_mesh(path)
    np_pcd = np.array(pcd.vertices)
    vol = torch.from_numpy(np_pcd).to('cuda',dtype=torch.float32).contiguous()
    return vol

def gen_xyz_list(simulator, frames, diff=True, save_ply=False, path=None):
    print('Generate xyz list')
    seq = []
    xyz0 = simulator.vol.detach()#.cpu()
    simulator.initialize()
    print(f'Init velocity: {simulator.vel.tolist()}')
    if diff:
        seq.append(torch.zeros_like(xyz0).to(simulator.vol.device))
    else:
        seq.append(xyz0)
        
    for f in tqdm(range(frames), desc="Generate trajectory"):
        xyz = simulator.forward(f)
        if f > 0:
            if diff:
                seq.append((xyz - xyz0).detach())
            else:
                seq.append(xyz.detach())
    
    if save_ply and path:
        length = len(seq)
        for f in range(length):
            write_particles(seq[f], f, path, name='prediction')
    return seq

def read_estimation_result(dataset: ModelParams, phys_args):
    files = os.listdir(dataset.model_path)
    pred_json = [f for f in files if "-pred.json" in f]
    result_json = None
    result = None
    if len(pred_json) == 0:
        print('Cannot find estimation result')

    with open(os.path.join(dataset.model_path, "{}-pred.json".format(str(phys_args.config_id))), 'r') as f:
        print(f'Load file: {result_json}')
        result = json.load(f)
    return result

def gen_bg(scene, color="white"):
    view = scene.getTrainCameras(scale=1.0)[0]
    w, h = view.image_width, view.image_height
    fn = torch.ones if color=="white" else torch.zeros
    bg = fn((3, h, w), device="cuda")
    return bg

def read_bg(phys_args, scene, dataset):
    use_orign_bg = getattr(phys_args, "origin_bg", False)
    if not use_orign_bg:
        return gen_bg(scene)
    cam_index = phys_args.view_id
    real_bg_path = os.path.join(dataset.source_path, "data", f"r_{cam_index}_-1.png")
    img = torchvision.io.read_image(real_bg_path)
    return img.to('cuda') / 255.0

def gen_surround_cam(center, r, num_points, FoVx, FoVy, image, fid):
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    c_x, c_y, c_z = center
    camera.lookat(*center)
    camera.up(0, 1, 0)
    # d_theta = np.pi / num_points
    d_phi = 2 * np.pi / num_points
    views = []
    for i in range(num_points):
        phi = i * d_phi
        x=c_x + r * np.cos(phi)
        y=c_y + 3.0#+ r * np.sin(theta) * np.sin(phi)
        z=c_z + r * np.sin(phi)
        camera.position(x,y,z)
        matrix=camera.get_view_matrix().T
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        if np.isnan(R).any() or np.isnan(T).any():
            continue
        cam = Camera(-1, R, T, FoVx, FoVy, image, image.cpu().numpy(), None, -1, fid=fid)
        views.append(cam)
    return views

def render_new(dataset: ModelParams, pipeline: PipelineParams, phys_args, scene, simulator, bg):
    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[1.0], load_fix_pcd=True)
    gaussians = scene.gaussians
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras(scale=1.0)
    surround_view = getattr(phys_args, "surround_view", False)
    
    cam_index = phys_args.view_id
    min_fid = min([view.fid for view in views])
    cams = [view for view in views if view.fid == min_fid]#[cam_index]
    cam = [v for v in cams if v.uid == cam_index][0]
    camera_center = cam.camera_center
    if not surround_view:
        render_views = [v for v in views if torch.equal(camera_center, v.camera_center)]
    else:
        centroid = simulator.vol.sum(dim=0) / simulator.vol.shape[0]
        r = (torch.sqrt(((camera_center - centroid) ** 2).sum()).to('cpu').item()) * 1.5
        cam_count = getattr(phys_args, "sur_traj_points", 120)
        render_views = gen_surround_cam(centroid.tolist(), r, cam_count, views[cam_index].FoVx, views[cam_index].FoVy, views[cam_index].original_image, views[cam_index].fid.item())
        bg = gen_bg(scene)

    # if getattr(phys_args, "n_frames", None):
    #     render_views = render_views[:phys_args.n_frames]
    views.sort(key=lambda cam: cam.fid)

    path = os.path.join(dataset.model_path, 'render')
    if not os.path.exists(path):
        mkdir_p(path)
    else:
        # delete exist image
        files = os.listdir(path)
        for file in files:
            os.remove(os.path.join(path, file))

    # knn gs filling
    # vertex_colors = np.concatenate([(gaussians.get_opacity.detach().cpu().numpy()*255).astype(np.uint8)]*3, axis=1)
    # trimesh.Trimesh(
    #     gaussians.get_xyz.detach().cpu().numpy(), 
    #     vertex_colors=vertex_colors, 
    #     ).export('gs.ply')
    if getattr(phys_args, "use_knn", False):
        with torch.no_grad():
            xyz = gaussians.get_xyz
            ids = torch.arange(xyz.shape[0]).to(xyz.device)
            opacity = gaussians.get_opacity.squeeze()
            th = 0.1
            ids1 = ids[opacity > th]
            ids2 = ids[opacity <= th]
            i = 0
            while ids2.shape[0] != 0 and i < 20:
                print(ids2.shape[0])
                dist, min_ids = mini_batch_knn(xyz[ids2], xyz[ids1], num_knn=1, bs=128)
                if dist is None:
                    i+=1
                    continue
                dist, min_ids = dist.T[0], min_ids.T[0]
                flag = dist <= 0.16 / 16
                min_ids = ids1[min_ids]
                ids[ids2[flag]] = min_ids[flag]
                ids1 = torch.cat([ids1, ids2[flag]])
                ids2 = ids2[torch.logical_not(flag)]
                i += 1
            gaussians._features_dc = gaussians._features_dc[ids]
            gaussians._features_rest = gaussians._features_rest[ids]
            gaussians._scaling = gaussians._scaling[ids]
            gaussians._opacity = gaussians._opacity[ids]
    # vertex_colors = np.concatenate([(gaussians.get_opacity.detach().cpu().numpy()*255).astype(np.uint8)]*3, axis=1)
    # trimesh.Trimesh(
    #     gaussians.get_xyz.detach().cpu().numpy(), 
    #     vertex_colors=vertex_colors, 
    #     ).export('gs2.ply')

    # for view in render_views:
    with torch.no_grad():
        simulator.initialize()
        max_f = phys_args.predict_frames if not surround_view else cam_count
        for f in tqdm(range(max_f), desc="Rendering"):
            if surround_view:
                view = render_views[f if f < len(render_views) else len(render_views)-1]
            else:
                view = render_views[0]
            xyz = simulator.forward(f)
            d_xyz = xyz - gaussians.get_xyz.detach()
            results = render(view, gaussians, pipeline, background, d_xyz, 0.0, 0.0, False)
            rendering = results["render"]
            alpha = results["alpha"]
            blend_img = alpha * rendering + (1.0 - alpha) * bg
            torchvision.utils.save_image(blend_img, os.path.join(dataset.model_path, 'render', '{0:05d}'.format(f) + ".png"))


if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Generate new trajectory")
    parser.add_argument('-vid', '--view_id', type=int, default=0)
    parser.add_argument('-knn', '--use_knn', type=bool, default=False)
    parser.add_argument('-cid', '--config_id', type=int, default=0)
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    phys_args.view_id = gs_args.view_id
    setattr(phys_args, "use_knn", gs_args.use_knn)
    setattr(phys_args, "config_id", gs_args.config_id)
    print(phys_args)
    safe_state(gs_args.quiet)
    dataset = model.extract(gs_args)
    
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.4)

    # 0. Load trained pcd
    vol = load_pcd_file(dataset.model_path, gs_args.iteration)

    # 1. (Optional ?) train again (improve the image quality)
    estimation_params = Namespace(**read_estimation_result(dataset, phys_args))
    simulator = Simulator(estimation_params, vol)
    d_xyz_list = gen_xyz_list(simulator, gs_args.train_frames) if gs_args.train_frames > 1 else None
    scene = train_gs_with_fixed_pcd(vol, dataset, op.extract(gs_args), pipeline.extract(gs_args), gs_args.test_iterations + list(range(10000, 40001, 1000)), gs_args.save_iterations, 
                            d_xyz_list, estimation_params.fps, force_train=getattr(gs_args, 'force_train', False), grid_size=estimation_params.density_grid_size)

    torch.cuda.empty_cache()
    # 2. Generate new trajectory
    simulator = Simulator(phys_args, vol)
    render_new(dataset, pipeline.extract(gs_args), phys_args, scene, simulator, read_bg(phys_args, scene, dataset))

    # 3. Generate *.mp4 file
    os.chdir(os.path.join(dataset.model_path, 'render'))
    # cmd = f'ffmpeg -y -framerate {phys_args.fps} -i %05d.png -pix_fmt yuv420p output.mp4'
    cmd = ['ffmpeg', '-y', '-framerate', f'{phys_args.fps}', '-i', '%05d.png', '-pix_fmt', 'yuv420p', 'output.mp4']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd = ['ffmpeg', '-y', '-framerate', f'{phys_args.fps}', '-i', '%05d.png', '-pix_fmt', 'yuv420p', 'output.gif']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # os.system(cmd)
    # gen video cmd: ffmpeg -y -framerate 24 -i %05d.png -pix_fmt yuv420p output.mp4