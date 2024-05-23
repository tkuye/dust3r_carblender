import os.path as osp
import itertools
from collections import deque
from pathlib import Path
import numpy as np
import pyexr
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import json 
import argparse

def get_args(args=None):
    parser = argparse.ArgumentParser(description='Render a scene with multiple cameras')
    parser.add_argument('--num_cameras', type=int, default=36, help='Number of cameras to render per level')
    parser.add_argument('--num_levels', type=int, default=5, help='Number of levels to render')
    parser.add_argument('--start_radius', type=float, default=8, help='Radius of the circle around the object')
    parser.add_argument('--base_path', type=str, default='./', help='Base path for output files')
    parser.add_argument('--obj_name', type=str, help='Name of the target object')
    parser.add_argument('--radius_decay', type=float, default=1.4, help='Decay rate of the radius for each level')
    parser.add_argument('--intrinsics_path', type=str, default='intrinsics.json', help='Path to save camera intrinsics file')
    parser.add_argument('--extrinsics_path', type=str, default='cameras.json', help='Path to save camera extrinsics file')
    parser.add_argument('--jitter', action='store_true', help='Add jitter to camera positions')
    parser.add_argument('--z_offset', type=float, default=0.8, help='Offset of the camera from the object')
    parser.add_argument('--z_scale_multiplier', type=float, default=1.4, help='Multiplier for the z offset')
    parser.add_argument('--camera_rotation', type=float, default=180, help='Rotation of the camera around the object in degrees')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--layer_name', type=str, default='ViewLayer', help='Name of the view layer to use for rendering')
    parser.add_argument('--resolution_x', type=int, default=1920, help='Resolution of the rendered image in x')
    parser.add_argument('--resolution_y', type=int, default=1440, help='Resolution of the rendered image in y')
    
    return parser.parse_args(args)

class Carblender(BaseStereoViewDataset):
    
    def __init__(self, ROOT:str, *args, **kwargs):
        self.ROOT = Path(ROOT)
        super().__init__(*args, **kwargs)
        self.scenes = self.load_scenes()
        self.scene_list = list(self.scenes.keys())
        self.combinations = self.get_combinations()
        self.invalidate = {scene: {} for scene in self.scene_list}
        
    def load_scenes(self):
        scenes = dict()
        for scene in self.ROOT.iterdir():
            scene_name = scene.name
            scene_images = []
            scene_positions = []
            scene_depths = []
            for image in scene.glob('image/*'):
                scene_images.append(str(image))
            for position in scene.glob('position/*'):
                scene_positions.append(str(position))
            for depth in scene.glob('depth/*'):
                scene_depths.append(str(depth))
            cameras_file_path = scene / 'cameras.npz'

            settings_file = scene / 'settings.json'
            settings = json.load(open(settings_file))['settings']
            
            args = get_args(settings)
            num_cameras = args.num_cameras
            num_levels = args.num_levels
                
            scenes[scene_name] = {
                'name': scene_name,
                'rgbs': scene_images,
                'positions': scene_positions,
                'depths': scene_depths,
                'length': len(scene_images),
                'cameras_file_path': cameras_file_path,
                'num_cameras': num_cameras,
                'num_levels': num_levels,
            }
        
        return scenes
    
    def get_combinations(self):
        combinations = dict()
        existing_combos = dict()
        for scene in self.scenes.values():
            num_cameras = scene['num_cameras']
            num_levels = scene['num_levels']
            if (num_levels, num_cameras) in existing_combos:
                combs = existing_combos[(num_levels, num_cameras)]
            else:
                ## 90 degrees 
                factor = 360 / num_cameras
                idx_to_degrees = {i: (i % num_cameras) * factor for i in range(num_cameras * num_levels)}
                combs = [(i, j) for i, j in itertools.combinations(range(num_cameras * num_levels), 2) if abs(idx_to_degrees[i] - idx_to_degrees[j]) <= 90]
                existing_combos[(num_levels, num_cameras)] = combs
            combinations[scene['name']] = combs
        return combinations
    
    def __len__(self):
        return sum([len(combs) for combs in self.combinations.values()])
    
    def convert_to_K(self, intrinsics):
        if isinstance(intrinsics, np.ndarray):
            if intrinsics.shape == (3, 3):
                return intrinsics
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K
    
    def _get_views(self, idx, resolution, rng):
        
        scene = self.scene_list[int((idx / sum([len(combs) for combs in self.combinations.values()])) * len(self.scene_list))]
        
        scene = self.scenes[scene]
        image_pool = scene['rgbs']
        depth_pool = scene['depths']
        position_pool = scene['positions']
        scene_cameras_data = np.load(scene['cameras_file_path'])
        ## get the remainder 
        idx = idx % len(self.combinations[scene['name']])
        im1_idx, im2_idx = self.combinations[scene['name']][idx]
        
        last = scene['length'] - 1
        
        if resolution not in self.invalidate[scene['name']]:
            self.invalidate[scene['name']][resolution] = [False for _ in range(scene['length'])]
        
        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        
        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            impath = image_pool[im_idx]
            orig_rgb_image = imread_cv2(impath)
            depthmap = pyexr.read(depth_pool[im_idx]).astype(np.float32)
            intrinsics = scene_cameras_data['intrinsics']
            K = self.convert_to_K(intrinsics).astype(np.float32)
            camera_pose = scene_cameras_data['cam_worlds'][im_idx]

            rgb_image, depthmap, K, _ = self._crop_resize_if_necessary(
                orig_rgb_image, depthmap, K, resolution, rng=rng, info=impath)
            
            #TODO Must resize positionmap as well
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[scene][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                v2=True,
                angle_x=scene_cameras_data['fov_x'].astype(np.float32),
                clip_start=scene_cameras_data['clip_start'].astype(np.float32),
                clip_end=scene_cameras_data['clip_end'].astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=K,
                dataset='CarblenderDataset',
                label=scene['name'],
                instance=osp.split(impath)[1].split('.')[0],
            ))
        
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import os 
    from PIL import Image
    from pyglet import gl
    import io
    os.environ['DISPLAY'] = ':1'
    
    dataset = Carblender(ROOT='/home/azureuser/cloudfiles/code/Users/tosin/blender_data/data', resolution=(1024, 768), aug_crop=16)
    
    
    print("Dataset Size: ", len(dataset))
    
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        
        window_conf = gl.Config(double_buffer=True, depth_size=24)
        byts = viz.scene.save_image((1920, 1440),  window_conf=window_conf)
        img = Image.open(io.BytesIO(byts))
        image_1, image_2 = views[0]['orig_img'], views[1]['orig_img']
        ## add images to the image

        image_1 = Image.fromarray(image_1, 'RGB')
        image_2 = Image.fromarray(image_2, 'RGB')
        new_img = Image.new('RGB', (img.width, image_1.height + image_2.height + img.height))
        new_img.paste(image_1, (0, 0))
        new_img.paste(image_2, (0, image_1.height))
        new_img.paste(img, (0, image_2.height + image_1.height))
        new_img.save(f'{idx}.png')
        
        break