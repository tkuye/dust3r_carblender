from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.utils.device import to_numpy    
import torch
import trimesh
import numpy as np
import os

model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
device = 'cuda'
schedule = 'cosine'
lr = 0.01
niter = 1000
max_images = 20
directory = "data/images"

all_images = os.listdir(directory)
skip_iter = len(all_images) // max_images
images_to_load = []

for i in range(0, len(all_images), skip_iter):
    images_to_load.append(f"{directory}/{all_images[i]}")

model = load_model(model_path, device)

images = load_images(images_to_load, size=512)

pairs = make_pairs(images, scene_graph='complete', symmetrize=True)
output = inference(pairs, model, device, batch_size=1)

scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks()

def summary_report(scene: PointCloudOptimizer, niter: int, lr: float, schedule: str):
    print(f'Optimization settings: niter={niter}, lr={lr}, schedule={schedule}')
    print('Number of images:', scene.n_imgs)
    print(f'Final focal lengths: {scene.get_focals().shape}')
    print(f'Final 3D points length: {len(scene.get_pts3d())}')


def create_pointcloud(pts3d, imgs, mask=None):
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    imgs = to_numpy(imgs)
    
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    
    pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    
    return pct

# Summary report
summary_report(scene, niter, lr, schedule)

if imgs:
    print('Creating point cloud')
    point_cloud = create_pointcloud(pts3d, imgs, confidence_masks)
    ## save point cloud to file
    print('Saving point cloud to file')
    point_cloud.export('point_cloud.ply')


print('Saving focals, poses, and 3D points to files')

if isinstance(focals, torch.Tensor):
    focals = focals.cpu()

if isinstance(poses, torch.Tensor):
    poses = poses.cpu()
elif isinstance(poses, list):
    for i in range(len(poses)):
        if isinstance(poses[i], torch.Tensor):
            poses[i] = poses[i].cpu()

if isinstance(pts3d, torch.Tensor):
    pts3d = pts3d.cpu()
elif isinstance(pts3d, list):
    for i in range(len(pts3d)):
        if isinstance(pts3d[i], torch.Tensor):
            pts3d[i] = pts3d[i].cpu()
            
torch.save(pts3d, 'pts3d.bin')
torch.save(poses, 'poses.bin')
torch.save(focals, 'focals.bin')   

