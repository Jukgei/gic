import os
import time
import scipy
import torch
import torchvision
import random
import subprocess
import numpy as np
import taichi as ti
import open3d as o3d
import trimesh as tm
from tqdm import tqdm
from simulator import Simulator
from utils.general_utils import safe_state
from pytorch3d.loss import chamfer_distance
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from new_trajectory import load_pcd_file, read_estimation_result, gen_xyz_list, render_new
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from utils.image_utils import psnr
from utils.loss_utils import ssim
from gaussian_renderer import render
from pathlib import Path

def discretize(pcd, vs):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=vs)
    voxels = voxel_grid.get_voxels()
    np_pcd = np.zeros((len(voxels), 3))
    for j in range(len(voxels)):
        c = voxel_grid.get_voxel_center_coordinate(voxels[j].grid_index)
        np_pcd[j, :] = c
    return torch.from_numpy(np_pcd).to('cuda',dtype=torch.float32).contiguous()

def load_gt_pcds(path):
    gts = []
    indices = [ply.split('.')[0] for ply in os.listdir(path)]
    indices.sort()
    n_digits = len(indices[0])
    indices = [int(idx) for idx in indices]
    indices.sort()
    for idx in range(len(indices)):
        pcd = tm.load_mesh(os.path.join(path, f"{idx:0{n_digits}d}.ply"))
        np_pcd = np.array(pcd.vertices)
        gts.append(torch.from_numpy(np_pcd).to('cuda',dtype=torch.float32).contiguous())
    return gts

def emd_func(x, y, pkg="torch"):
    if pkg == "numpy":
        # numpy implementation
        x_ = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y_ = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        cost_matrix = np.linalg.norm(x_ - y_, 2, axis=2)
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = np.mean(np.linalg.norm(x[ind1] - y[ind2], 2, axis=1))
    else:
        # torch implementation
        x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
        cost_matrix = dis.detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))

    return emd

def evaluate(preds, gts, train_frames, loss_type='CD'):
    print(f"Prediction sequence {len(preds)}, gts sequence {len(gts)}")
    if len(preds) != len(gts):
        print("[Error]: The prediction sequence is not align with the gt sequence.")
        return
    print(f"Prediction pcd particles cnt {preds[0].shape[0]}, gt pcd particles cnt {gts[0].shape[0]}")
    max_f = len(preds)
    fit_loss = 0.0
    predict_loss = 0.0
    for f in tqdm(range(max_f), desc=f"Evaluate {loss_type} Loss"):
        # pcd1 = discretize(preds[f], 0.02)
        # pcd2 = discretize(gts[f], 0.02)
        # cd = chamfer_distance(preds[f], gts[f])
        # print(f"frames: {f}, cd: {cd}")

        # cd align with https://zlicheng.com/spring_gaus/
        n_sample = 2048 if loss_type == 'EMD' else 8192
        pcd0 = preds[f]
        pcd1 = gts[f]
        n_sample = min(n_sample, pcd0.shape[0], pcd1.shape[0])

        pcd0 = pcd0[random.sample(range(pcd0.shape[0]), n_sample), :]
        pcd1 = pcd1[random.sample(range(pcd1.shape[0]), n_sample), :]

        if loss_type == "CD":
            loss = (chamfer_distance(pcd0[None], pcd1[None])[0] * 1e6).item()
        elif loss_type == "EMD":
            loss = emd_func(pcd0, pcd1)
        else:
            print("[Error]: undefined error type.")
        if f < train_frames:
            fit_loss+= loss
        else:
            predict_loss+= loss

    fit_loss /= train_frames
    if max_f - train_frames > 0.0:
        predict_loss /= (max_f - train_frames)
    print(f"{loss_type} loss train: {fit_loss}, {loss_type} loss predict: {predict_loss}")
    return fit_loss, predict_loss

if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Prediction")
    parser.add_argument("--predict_frames", default=30, type=int)
    parser.add_argument("--train_frames", type=int)#, default=14, type=int)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument('-cid', '--config_id', type=int, default=0)
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    setattr(phys_args, "config_id", gs_args.config_id)
    # print(phys_args)
    safe_state(gs_args.quiet)
    dataset = model.extract(gs_args)
    opt = op.extract(gs_args)
    pipe = pipeline.extract(gs_args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    model_path = Path(dataset.model_path)
    obj_name = dataset.model_path.split('/')[-1]
    (model_path/f'{obj_name}_img_render').mkdir(exist_ok=True)
    (model_path/f'{obj_name}_img_gt').mkdir(exist_ok=True)

    # 0. Load trained pcd
    vol = load_pcd_file(dataset.model_path, gs_args.iteration)

    
    estimation_params = Namespace(**read_estimation_result(dataset, phys_args))
    simulator = Simulator(estimation_params, vol)
    d_xyz_list = gen_xyz_list(simulator, gs_args.predict_frames, diff=True, save_ply=False, path=dataset.model_path)
    d_xyz_list = d_xyz_list[:20]
    scene = train_gs_with_fixed_pcd(
        vol, dataset, 
        opt, pipe, 
        gs_args.test_iterations + list(range(10000, 40001, 1000)), gs_args.save_iterations, 
        d_xyz_list, estimation_params.fps, 
        force_train=True, 
        grid_size=estimation_params.density_grid_size
    )
    views = scene.getTrainCameras(scale=dataset.res_scale).copy()
    # views = [view for view in views if view.fid > gs_args.train_frames / estimation_params.fps - 1e-4]
    simulator = Simulator(estimation_params, vol)
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        simulator.initialize()
        max_f = gs_args.predict_frames
        for f in range(max_f):
            xyz = simulator.forward(f)
            curr_views = [view for view in views if torch.abs(view.fid - f / estimation_params.fps) < 1e-4]
            d_xyz = xyz - scene.gaussians.get_xyz.detach()
            for view in curr_views:
                results = render(view, scene.gaussians, pipeline, background, d_xyz, 0.0, 0.0, False)
                image = results["render"]
                gt_image = view.original_image.cuda()
                if view.uid == 3:
                    torchvision.utils.save_image(image, model_path/f'{obj_name}_img_render'/f'{f:05d}.png')
                    torchvision.utils.save_image(gt_image, model_path/f'{obj_name}_img_gt'/f'{f:05d}.png')
                if f < gs_args.train_frames:
                    continue
                psnr_list.append(psnr(image, gt_image))
                ssim_list.append(ssim(image, gt_image))
    mean_psnr = torch.mean(torch.stack(psnr_list))
    mean_ssim = torch.mean(torch.stack(ssim_list))            

    # 1. Predicting trajectory
    seq = gen_xyz_list(simulator, gs_args.predict_frames, diff=False, save_ply=False, path=dataset.model_path)

    # 2. Load Gt
    gts = load_gt_pcds(gs_args.gt_path)

    # 3.evaluate CD loss & EMD loss
    evaluate(seq, gts, gs_args.train_frames, 'CD')
    evaluate(seq, gts, gs_args.train_frames, 'EMD')

    print(f'average psnr: {mean_psnr}')
    print(f'average ssim: {mean_ssim}')


    render_abs_path = (model_path/f'{obj_name}_img_render').resolve()
    gt_abs_path = (model_path/f'{obj_name}_img_gt').resolve()
    os.chdir(render_abs_path)
    cmd = ['ffmpeg', '-y', '-framerate', f'30', '-i', '%05d.png', '-pix_fmt', 'yuv420p', 'render.gif']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = ['ffmpeg', '-y', '-framerate', f'30', '-i', '%05d.png', '-pix_fmt', 'yuv420p', 'render.mp4']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.chdir(gt_abs_path)
    cmd = ['ffmpeg', '-y', '-framerate', f'30', '-i', '%05d.png', '-pix_fmt', 'yuv420p', 'gt.gif']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = ['ffmpeg', '-y', '-framerate', f'30', '-i', '%05d.png', '-pix_fmt', 'yuv420p', 'gt.mp4']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
