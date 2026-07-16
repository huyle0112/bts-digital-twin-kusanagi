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

import os
import sys
import csv
from PIL import Image
from typing import NamedTuple, Optional, Tuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # Ideal models (undistorted). Distorted models are approximated as pinhole
        # by dropping radial/tangential terms — fine when |k| is small.
        if intr.model == "SIMPLE_PINHOLE":
            # params: f, cx, cy
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            # params: fx, fy, cx, cy
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            # params: f, cx, cy, k
            focal_length_x = intr.params[0]
            k = float(intr.params[3]) if len(intr.params) > 3 else 0.0
            if abs(k) > 1e-6 and idx == 0:
                print(f"\n[Warning] Camera model {intr.model} has distortion k={k:.6g}; "
                      f"approximating as SIMPLE_PINHOLE (ignore k). "
                      f"For best quality run COLMAP image_undistorter.")
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ("RADIAL", "RADIAL_FISHEYE"):
            # params: f, cx, cy, k1, k2
            focal_length_x = intr.params[0]
            if idx == 0:
                print(f"\n[Warning] Camera model {intr.model}; approximating as SIMPLE_PINHOLE (ignore k1,k2).")
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            # params: fx, fy, cx, cy, ...distortion
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            if idx == 0:
                print(f"\n[Warning] Camera model {intr.model}; approximating as PINHOLE (ignore distortion). "
                      f"For best quality run COLMAP image_undistorter.")
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                f"Colmap camera model not handled: {intr.model}. "
                f"Supported (exact): PINHOLE, SIMPLE_PINHOLE. "
                f"Supported (approx, ignore distortion): SIMPLE_RADIAL, RADIAL, OPENCV. "
                f"Otherwise undistort with COLMAP image_undistorter first."
            )

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def resolve_colmap_and_test_paths(path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Resolve COLMAP root + optional test CSV layout.

    Supports:
      1) Competition layout:
           <path>/train/sparse/0  +  <path>/test/test_poses.csv, <path>/test/images
      2) source_path points at train/:
           <path>/sparse/0  +  sibling <path>/../test/test_poses.csv
      3) Standard 3DGS layout:
           <path>/sparse/0  +  optional <path>/test_poses.csv, <path>/test_images
    """
    path = os.path.abspath(path)

    # Competition root: scenes/{train,test}
    train_sparse = os.path.join(path, "train", "sparse")
    if os.path.isdir(train_sparse):
        colmap_root = os.path.join(path, "train")
        test_csv = os.path.join(path, "test", "test_poses.csv")
        test_images = os.path.join(path, "test", "images")
        if os.path.isfile(test_csv):
            return colmap_root, test_csv, test_images
        return colmap_root, None, None

    # source_path is the train folder (or standard scene root with sparse/)
    if os.path.isdir(os.path.join(path, "sparse")):
        colmap_root = path
        sibling_csv = os.path.join(os.path.dirname(path), "test", "test_poses.csv")
        sibling_images = os.path.join(os.path.dirname(path), "test", "images")
        if os.path.isfile(sibling_csv):
            return colmap_root, sibling_csv, sibling_images

        root_csv = os.path.join(path, "test_poses.csv")
        if os.path.isfile(root_csv):
            for images_dir in (
                os.path.join(path, "test_images"),
                os.path.join(path, "test", "images"),
            ):
                if os.path.isdir(images_dir):
                    return colmap_root, root_csv, images_dir
            return colmap_root, root_csv, os.path.join(path, "test_images")

        nested_csv = os.path.join(path, "test", "test_poses.csv")
        if os.path.isfile(nested_csv):
            return colmap_root, nested_csv, os.path.join(path, "test", "images")

        return colmap_root, None, None

    return path, None, None


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8, load_test=True):
    """
    load_test: If False, only build train cameras (for optimization). Test poses/images
    are left for render/compare pipelines and are not returned.
    """
    colmap_root, test_csv, test_images_folder = resolve_colmap_and_test_paths(path)
    sparse_dir = os.path.join(colmap_root, "sparse", "0")

    try:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(sparse_dir, "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # When test poses come from CSV, train set is the full COLMAP reconstruction.
    if test_csv is not None:
        test_cam_names_list = []
    elif eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(sparse_dir, "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(colmap_root, reading_dir),
        depths_folder=os.path.join(colmap_root, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # Train set = COLMAP / reconstruction images that have poses + files under images/.
    # Never mix test CSV poses into training.
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    print(f"Train cameras (optimization targets): {len(train_cam_infos)}")

    if not load_test:
        test_cam_infos = []
        if test_csv is not None:
            print(f"[Info] Not loading test poses during this step (CSV present: {test_csv}). "
                  f"Use render.py later for novel-view / comparison.")
    elif test_csv is not None:
        print(f"Loading test poses for render/compare (not training targets): {test_csv}")
        test_cam_infos = readTestPoseCSV(test_csv, test_images_folder)
    else:
        test_cam_infos = [c for c in cam_infos if c.is_test]
        if test_cam_infos:
            print(f"Held-out test cameras from COLMAP split: {len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(sparse_dir, "points3D.ply")
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    pcd = None

    # Prefer existing .ply; otherwise load points from COLMAP .bin/.txt directly.
    # On read-only filesystems (e.g. Kaggle /kaggle/input) we must NOT require writing .ply.
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except Exception as e:
            print(f"[Warning] Failed to read {ply_path}: {e}")
            pcd = None

    if pcd is None and (os.path.exists(bin_path) or os.path.exists(txt_path)):
        print("Loading point cloud from points3D.bin/.txt (no writable points3D.ply required).")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        pcd = BasicPointCloud(
            points=xyz,
            colors=rgb.astype(np.float64) / 255.0,
            normals=np.zeros_like(xyz),
        )
        # Best-effort: cache .ply next to sparse if the folder is writable
        try:
            storePly(ply_path, xyz, rgb)
            print(f"Cached points3D.ply at {ply_path}")
        except OSError:
            print(f"[Info] Sparse dir is read-only; using points from .bin in memory. "
                  f"input.ply will be written under the model output folder.")
            ply_path = ""  # Scene will write input.ply from point_cloud

    if pcd is None:
        print("[Warning] No point cloud loaded from sparse model.")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}

def readTestPoseCSV(csv_path, test_image_folder):
    """
    Build CameraInfo list from competition test_poses.csv.

    Expected columns:
      image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height

    Quaternion (w,x,y,z) and translation (x,y,z) use COLMAP world-to-camera
    convention, matching train sparse reconstruction.
    fx/fy are focal lengths in pixels; width/height are render resolution.
    cx/cy are principal points (assumed image center by the 3DGS rasterizer).
    """
    if test_image_folder is None:
        test_image_folder = os.path.dirname(csv_path)

    cam_infos = []
    missing_images = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
                    "fx", "fy", "width", "height"}
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {csv_path}")
        fields = {name.strip() for name in reader.fieldnames}
        missing_cols = required - fields
        if missing_cols:
            raise ValueError(
                f"test_poses.csv missing columns {sorted(missing_cols)}. "
                f"Found: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader):
            image_name = row["image_name"].strip()
            qw = float(row["qw"])
            qx = float(row["qx"])
            qy = float(row["qy"])
            qz = float(row["qz"])
            tx = float(row["tx"])
            ty = float(row["ty"])
            tz = float(row["tz"])
            fx = float(row["fx"])
            fy = float(row["fy"])
            width = int(float(row["width"]))
            height = int(float(row["height"]))

            # COLMAP W2C: R_store is transposed for glm, T is tvec
            qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
            R = np.transpose(qvec2rotmat(qvec))
            T = np.array([tx, ty, tz], dtype=np.float64)

            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)

            image_path = os.path.join(test_image_folder, image_name)
            if not os.path.isfile(image_path):
                missing_images += 1

            cam_infos.append(CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                depth_params=None,
                image_path=image_path,
                image_name=image_name,
                depth_path="",
                width=width,
                height=height,
                is_test=True,
            ))

    print(f"Loaded {len(cam_infos)} test cameras from {csv_path}")
    if missing_images:
        print(f"Warning: {missing_images}/{len(cam_infos)} GT images not found under {test_image_folder}")
        print("Missing files will use a black placeholder of CSV width/height.")
    return cam_infos
