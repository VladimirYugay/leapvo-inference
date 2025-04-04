import os
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from decord import VideoReader
from omegaconf import DictConfig
from tqdm import tqdm

from main.leapvo import LEAPVO
from main.utils import save_trajectory_tum_format


def scannet_image_stream(imagedir):
    """ Image generator for ScanNet """
    imagedir = Path(imagedir)
    video_path = imagedir / "video.mp4"
    pose_path = imagedir / "poses.npz"
    intrinsics_path = imagedir / "intrinsics.npz"

    vr = VideoReader(str(video_path))
    c2ws = np.load(pose_path)["poses"]

    # Take everything until first invalid pose
    inf_ids = np.where(np.isinf(c2ws).any(axis=(1, 2)))[0]
    if inf_ids.size > 0:
        c2ws = c2ws[:inf_ids.min()]

    # Move to the origin for visualization
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws

    # Load intrinsics
    intrinsics = np.load(intrinsics_path)["poses"]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    intrinsics = np.array([fx, fy, cx, cy])

    for i in range(50):
        image = vr[i].asnumpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        yield (float(i), image, c2ws[i], intrinsics)


@hydra.main(version_base=None, config_path="configs", config_name="demo")
def main(cfg: DictConfig):

    slam = None

    imagedir = cfg.data.imagedir
    dataloader = scannet_image_stream(imagedir)

    gt_poses = []
    image_list = []
    intrinsics_list = []
    for i, (t, image, gt_pose, intrinsics) in enumerate(tqdm(dataloader)):
        if t < 0:
            break

        image_list.append(image)
        intrinsics_list.append(intrinsics)
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = LEAPVO(cfg, ht=image.shape[1], wd=image.shape[2])

        gt_poses.append(gt_pose)
        slam(t, image, intrinsics)

    # tx ty tz qw qx qy qz
    pred_traj = slam.terminate()

    os.makedirs(f"{cfg.data.savedir}", exist_ok=True)
    pred_traj = list(pred_traj)

    save_trajectory_tum_format(pred_traj, f"{cfg.data.savedir}/pred_traj.txt")
    np.save(f"{cfg.data.savedir}/gt_traj.npy", gt_poses)


if __name__ == "__main__":
    main()
