#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def combined_score(ssim_val, psnr_val, lpips_val, psnr_max):
    """
    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    where PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)
    """
    psnr_norm = torch.clamp(psnr_val / psnr_max, 0.0, 1.0)
    return 0.4 * (1.0 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm


def evaluate(model_paths, psnr_max=40.0):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    print(f"Using PSNR_max = {psnr_max} for normalization")
    print("Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm")
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                scores = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssim_val = ssim(renders[idx], gts[idx])
                    psnr_val = psnr(renders[idx], gts[idx])
                    lpips_val = lpips(renders[idx], gts[idx], net_type='vgg')
                    score_val = combined_score(ssim_val, psnr_val, lpips_val, psnr_max)

                    ssims.append(ssim_val)
                    psnrs.append(psnr_val)
                    lpipss.append(lpips_val)
                    scores.append(score_val)

                ssim_mean = torch.tensor(ssims).mean()
                psnr_mean = torch.tensor(psnrs).mean()
                lpips_mean = torch.tensor(lpipss).mean()
                score_mean = torch.tensor(scores).mean()
                # Also report score from mean metrics (same formula on averages)
                score_from_means = combined_score(ssim_mean, psnr_mean, lpips_mean, psnr_max)

                print("  SSIM      : {:>12.7f}".format(ssim_mean.item()))
                print("  PSNR      : {:>12.7f}".format(psnr_mean.item()))
                print("  LPIPS     : {:>12.7f}".format(lpips_mean.item()))
                print("  PSNR_norm : {:>12.7f}".format(torch.clamp(psnr_mean / psnr_max, 0.0, 1.0).item()))
                print("  Score     : {:>12.7f}  (mean of per-view scores)".format(score_mean.item()))
                print("  Score_avg : {:>12.7f}  (score on mean SSIM/PSNR/LPIPS)".format(score_from_means.item()))
                print("")

                full_dict[scene_dir][method].update({
                    "SSIM": ssim_mean.item(),
                    "PSNR": psnr_mean.item(),
                    "LPIPS": lpips_mean.item(),
                    "PSNR_norm": torch.clamp(psnr_mean / psnr_max, 0.0, 1.0).item(),
                    "Score": score_mean.item(),
                    "Score_avg": score_from_means.item(),
                    "PSNR_max": psnr_max,
                })
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: v for v, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: v for v, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: v for v, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "Score": {name: v for v, name in zip(torch.tensor(scores).tolist(), image_names)},
                })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, ":", e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics evaluation")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--psnr_max', type=float, default=40.0,
                        help="PSNR ceiling for normalization: psnr_norm = clamp(psnr / psnr_max, 0, 1)")
    args = parser.parse_args()
    evaluate(args.model_paths, psnr_max=args.psnr_max)
