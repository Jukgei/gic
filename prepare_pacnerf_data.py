import torch
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from matting import MattingRefine


device = torch.device("cuda:0")
torch.manual_seed(123)
np.random.seed(123)


import argparse
def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', required=True,
                        help='path to the object, eg, data/pacnerf/torus')
    return parser


def preprare_mask(data_folder):
    data_folder = Path(data_folder)
    matting_model = MattingRefine(backbone='resnet101',
                    backbone_scale=1/2,
                    refine_mode = 'sampling',
                    refine_sample_pixels = 100_000)
    matting_model.load_state_dict(torch.load('data/checkpoint/pytorch_resnet101.pth', map_location=device))
    matting_model = matting_model.eval().to(torch.float32).to(device)
    with open(data_folder/"all_data.json") as f:
        data_info = json.load(f)
    num_views = len(list(set([int(entry["file_path"].split('_')[1]) for entry in data_info])))

    intrinsic = data_info[0]['intrinsic']
    W, H = int(intrinsic[0][2])*2, int(intrinsic[1][2])*2
    focal_length = intrinsic[0][0]
    fov = np.arctan(0.5 * H / focal_length) * 2
    print(f'fov: {fov/np.pi*180}')
    out_dict = {
        "camera_angle_x": fov, 
        "frames": [], 
        }
        
    backgroud_all = np.zeros([num_views, H, W, 3])
    for entry in data_info:
        cam_id, frame_id = [int(i) for i in entry["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")]
        if frame_id == -1:
            backgroud_all[cam_id] = np.array(Image.open(data_folder/entry["file_path"]))[..., :3]
    for entry in tqdm(data_info):
        cam_id, frame_id = [int(i) for i in entry["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")]
        if frame_id < 0:
            continue
        img = np.array(Image.open(data_folder/entry["file_path"]))[..., :3]
        bgr = backgroud_all[cam_id]
        with torch.no_grad():
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)[None]
            bgr_tensor = torch.from_numpy(bgr.transpose(2, 0, 1).astype(np.float32) / 255).to(device)[None]
            pha = matting_model(img_tensor, bgr_tensor)[0][0,0].cpu().numpy()
            mask = pha < 0.9
        img[mask] = 255
        Image.fromarray(img).save(data_folder/f"data/m_{cam_id}_{frame_id}.png")

        file_path = entry['file_path'].replace('r', 'm')
        alpha = np.logical_not(mask).astype(np.uint8) * 255
        image = np.concatenate([img, alpha[..., None]], axis=-1)
        file_path = entry['file_path'].replace('r', 'a')
        Image.fromarray(image).save(data_folder/file_path)
        file_path = file_path[0:-4]  # remove .png
        entry['c2w'].append([0.0, 0.0, 0.0, 1.0])
        frame = {
                "file_path": file_path, 
                "rotation": 0.0, 
                "time": entry['time'], 
                "transform_matrix": entry['c2w'], 
            }
        out_dict['frames'].append(frame)
    with open(data_folder/f'transforms_train.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
    test_frames = out_dict.copy()
    test_frames['frames'] = []
    for frame in out_dict['frames']:
        if frame['file_path'].split('_')[-2] == '0':
            test_frames['frames'].append(frame)
    with open(data_folder/f'transforms_test.json', 'w') as f:
        json.dump(test_frames, f, indent=4)
    with open(data_folder/f'transforms_val.json', 'w') as f:
        json.dump(test_frames, f, indent=4)


if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    preprare_mask(args.data_folder)
