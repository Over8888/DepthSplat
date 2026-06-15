from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R


@dataclass
class PoseData:
    image_paths: list[Path]
    extrinsics: torch.Tensor  # [V, 4, 4] c2w, OpenCV convention
    intrinsics: torch.Tensor  # [V, 3, 3] normalized (fx/W, fy/H)
    image_size: tuple[int, int]  # (H, W)


# ---------------------------------------------------------------------------
# Format detectors
# ---------------------------------------------------------------------------

def detect_pose_format(pose_path: Path) -> str:
    if pose_path.suffix == ".npz":
        return "npz"
    if pose_path.suffix == ".npy":
        return "npy"
    with open(pose_path) as f:
        content = f.read(1024)
    if '"frames"' in content:
        return "transforms_json"
    if "QW" in content and "TX" in content:
        return "colmap_images_txt"
    raise ValueError(f"Cannot detect pose format for {pose_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENGL_TO_OPENCV = torch.tensor(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=torch.float32,
)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()


def load_image_to_tensor(path: Path) -> torch.Tensor:
    from torchvision.transforms import ToTensor

    return ToTensor()(Image.open(path))[:3]


def estimate_near_far(
    extrinsics: torch.Tensor,
    multiplier_near: float = 0.02,
    multiplier_far: float = 3.0,
    min_near: float = 0.05,
    max_far: float = 500.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = extrinsics[:, :3, 3]  # [V, 3]
    if positions.shape[0] < 2:
        near = torch.full((positions.shape[0],), min_near)
        far = torch.full((positions.shape[0],), max_far)
        return near, far
    max_baseline = torch.cdist(positions, positions).max().item()
    near_val = max(max_baseline * multiplier_near, min_near)
    far_val = min(max_baseline * multiplier_far, max_far)
    near = torch.full((positions.shape[0],), near_val)
    far = torch.full((positions.shape[0],), far_val)
    return near, far


def clamp_to_divisible_pair(h: int, w: int, divisor: int) -> tuple[int, int]:
    h_new = (h // divisor) * divisor
    w_new = (w // divisor) * divisor
    return h_new, w_new


def center_crop_divisible(
    images: torch.Tensor,  # [V, 3, H, W]
    intrinsics: torch.Tensor,  # [V, 3, 3]
    divisor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, h, w = images.shape
    h_new, w_new = clamp_to_divisible_pair(h, w, divisor)
    if (h_new, w_new) == (h, w):
        return images, intrinsics
    row = (h - h_new) // 2
    col = (w - w_new) // 2
    images = images[:, :, row : row + h_new, col : col + w_new]
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 0] *= w / w_new
    intrinsics[:, 1, 1] *= h / h_new
    intrinsics[:, 0, 2] -= col / w_new
    intrinsics[:, 1, 2] -= row / h_new
    return images, intrinsics


def sort_by_name(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: p.name)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_transforms_json(
    path: Path,
    image_dir: Optional[Path] = None,
    load_images: bool = True,
) -> PoseData:
    import json

    with open(path) as f:
        data = json.load(f)

    frames = data["frames"]
    extrinsics_list = []
    image_paths: list[Path] = []
    image_size: Optional[tuple[int, int]] = None
    raw_intrinsics: Optional[tuple[float, float, float, float]] = None

    # If data contains camera intrinsics info (nerfstudio style)
    if "fl_x" in data:
        raw_intrinsics = (
            data["fl_x"],
            data.get("fl_y", data["fl_x"]),
            data.get("cx", data.get("w", 1.0) / 2),
            data.get("cy", data.get("h", 1.0) / 2),
        )

    for frame in frames:
        img_path = Path(frame["file_path"])
        if not img_path.is_absolute():
            if image_dir is not None:
                img_path = image_dir / img_path
            else:
                img_path = path.parent / img_path
        image_paths.append(img_path)

        mat = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        # Assume OpenGL convention -> convert to OpenCV
        mat = mat @ OPENGL_TO_OPENCV
        extrinsics_list.append(mat)

    extrinsics = torch.stack(extrinsics_list)  # [V, 4, 4]

    if load_images and image_paths:
        test_img = load_image_to_tensor(image_paths[0])
        _, h, w = test_img.shape
        image_size = (h, w)
    elif "h" in data and "w" in data:
        h, w = data["h"], data["w"]
        image_size = (h, w)
    else:
        h, w = 256, 256
        image_size = (h, w)

    if raw_intrinsics is not None:
        fx, fy, cx, cy = raw_intrinsics
        # Normalize intrinsics
        K = torch.tensor(
            [[fx / w, 0, cx / w], [0, fy / h, cy / h], [0, 0, 1]],
            dtype=torch.float32,
        )
    else:
        # Default projection from fov_y if available
        fov_y = data.get("camera_angle_y", data.get("fov_y", None))
        if fov_y is not None:
            fy = 0.5 * h / np.tan(0.5 * fov_y)
            fx = fy  # assume square pixels
            K = torch.tensor(
                [[fx / w, 0, 0.5], [0, fy / h, 0.5], [0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            # Placeholder; user must provide intrinsics
            K = torch.tensor(
                [[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]],
                dtype=torch.float32,
            )

    intrinsics = K.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)

    return PoseData(
        image_paths=sort_by_name(image_paths),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_size=image_size,
    )


def load_colmap(
    cameras_path: Path,
    images_path: Path,
    image_dir: Optional[Path] = None,
) -> PoseData:
    cameras = {}
    with open(cameras_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            w = int(parts[2])
            h = int(parts[3])
            params = list(map(float, parts[4:]))
            if model in ("SIMPLE_PINHOLE", "PINHOLE"):
                fx, cx, cy = params[0], params[1], params[2]
                fy = params[3] if model == "PINHOLE" else fx
            elif model == "SIMPLE_RADIAL":
                fx, cx, cy = params[0], params[1], params[2]
                fy = fx
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            cameras[cam_id] = {
                "w": w,
                "h": h,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }

    image_paths: list[Path] = []
    intrinsics_list = []
    extrinsics_list = []

    with open(images_path) as f:
        lines = f.readlines()

    # COLMAP images.txt format: every image line is followed by a points line.
    # Image line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # Points line: POINTS2D[] - just skip it
    image_lines = []
    skip_next = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if skip_next:
            skip_next = False
            continue
        parts = line.split()
        if len(parts) < 9:
            skip_next = False
            continue
        image_lines.append(line)
        skip_next = True

    for line in image_lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        cam_id = int(parts[-2])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        img_name = parts[-1]
        img_path = Path(img_name.strip())
        if not img_path.is_absolute():
            if image_dir is not None:
                img_path = image_dir / img_path
            else:
                img_path = images_path.parent / img_path

        cam = cameras[cam_id]
        w, h = cam["w"], cam["h"]

        # Rotation: COLMAP stores w2c rotation as quaternion
        R_w2c = qvec2rotmat(np.array([qw, qx, qy, qz]))
        T_world = np.array([tx, ty, tz])

        # c2w rotation = R_w2c^T, c2w translation = camera position in world
        R_c2w = R_w2c.T
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = T_world

        # COLMAP uses OpenCV-like convention; no axis flip needed
        extrinsics_list.append(torch.tensor(c2w, dtype=torch.float32))

        # Normalized intrinsics
        K = torch.tensor(
            [
                [cam["fx"] / w, 0, cam["cx"] / w],
                [0, cam["fy"] / h, cam["cy"] / h],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        intrinsics_list.append(K)
        image_paths.append(img_path)

    if len(image_paths) == 0:
        raise ValueError(f"No valid images found in {images_path}")

    extrinsics = torch.stack(extrinsics_list)
    intrinsics = torch.stack(intrinsics_list)

    image_size = (cameras[cam_id]["h"], cameras[cam_id]["w"])

    return PoseData(
        image_paths=sort_by_name(image_paths),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_size=image_size,
    )


def load_npz(path: Path) -> PoseData:
    data = np.load(path)
    extrinsics = torch.tensor(data["extrinsics"], dtype=torch.float32)
    intrinsics = torch.tensor(data["intrinsics"], dtype=torch.float32)
    if extrinsics.dim() == 2 and extrinsics.shape == (4, 4):
        extrinsics = extrinsics.unsqueeze(0)
    if intrinsics.dim() == 2 and intrinsics.shape == (3, 3):
        intrinsics = intrinsics.unsqueeze(0)
    if "image_paths" in data:
        image_paths = [Path(p) for p in data["image_paths"]]
    else:
        image_paths = [Path(f"frame_{i:04d}.png") for i in range(extrinsics.shape[0])]

    test_img = load_image_to_tensor(image_paths[0])
    image_size = (test_img.shape[1], test_img.shape[2])

    return PoseData(
        image_paths=image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_size=image_size,
    )


# ---------------------------------------------------------------------------
# Target poses (no images needed)
# ---------------------------------------------------------------------------

def load_target_poses(pose_path: Path, format: Optional[str] = None) -> PoseData:
    fmt = format or detect_pose_format(pose_path)
    if fmt == "transforms_json":
        return load_transforms_json(pose_path, image_dir=None, load_images=False)
    elif fmt in ("npz", "npy"):
        data = np.load(pose_path)
        extrinsics = torch.tensor(data["extrinsics"], dtype=torch.float32)
        intrinsics = torch.tensor(data["intrinsics"], dtype=torch.float32)
        if extrinsics.dim() == 2:
            extrinsics = extrinsics.unsqueeze(0)
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0)
        h = data.get("h", 256)
        w = data.get("w", 256)
        return PoseData(
            image_paths=[],
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_size=(h, w),
        )
    else:
        raise ValueError(f"Unsupported format for target poses: {fmt}")


# ---------------------------------------------------------------------------
# Unified load
# ---------------------------------------------------------------------------

def load_poses(
    pose_path: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    colmap_cameras: Optional[Path] = None,
    colmap_images: Optional[Path] = None,
    format: Optional[str] = None,
    image_paths: Optional[list[Path]] = None,
) -> PoseData:
    """Unified pose-loading entry point.

    If ``image_paths`` is provided, uses those files directly (sorted by name).
    Otherwise loads images from ``images_dir`` (sorted glob *.png, *.jpg, *.jpeg),
    using the pose file to filter/map.

    The ``pose_path`` can be:
      - transforms.json (Nerfstudio style)
      - .npz with ``extrinsics`` and ``intrinsics`` arrays
      - (not needed when using separate colmap files)

    For COLMAP, use ``colmap_cameras`` and ``colmap_images``.
    """
    if image_paths is not None:
        image_paths = sorted(image_paths)
    elif images_dir is not None:
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        image_paths = []
        for pat in patterns:
            image_paths.extend(sorted(images_dir.glob(pat)))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {images_dir}")
    else:
        raise ValueError("Either image_paths or images_dir must be provided")

    if colmap_cameras is not None and colmap_images is not None:
        pose_data = load_colmap(colmap_cameras, colmap_images, images_dir)
    elif pose_path is not None:
        fmt = format or detect_pose_format(pose_path)
        if fmt == "transforms_json":
            pose_data = load_transforms_json(pose_path, images_dir)
        elif fmt in ("npz", "npy"):
            pose_data = load_npz(pose_path)
        else:
            raise ValueError(f"Unsupported pose format: {fmt}")
    else:
        raise ValueError("No pose data provided")

    # Map image_paths to the pose data image paths
    if pose_data.image_paths != image_paths:
        # Match by filename
        name_map = {p.name: p for p in image_paths}
        ordered = []
        for p in pose_data.image_paths:
            if p.name in name_map:
                ordered.append(name_map[p.name])
            elif p in image_paths:
                ordered.append(p)
            else:
                raise FileNotFoundError(
                    f"Image {p.name} referenced in poses but not found in image list"
                )
        pose_data.image_paths = ordered

    return pose_data


# ---------------------------------------------------------------------------
# Individual camera.json loader (for Family/DL3DV-like scenes)
# ---------------------------------------------------------------------------

def load_single_camera_json(path: Path) -> dict:
    import json

    with open(path) as f:
        data = json.load(f)

    w = data["width"]
    h = data["height"]
    fx = data["fx"]
    fy = data["fy"]
    cx = data["cx"]
    cy = data["cy"]

    intrinsics = torch.tensor(
        [[fx / w, 0, cx / w],
         [0, fy / h, cy / h],
         [0, 0, 1]],
        dtype=torch.float32,
    )

    c2w = torch.tensor(data["camera_to_world"], dtype=torch.float32)

    return {
        "extrinsics": c2w,
        "intrinsics": intrinsics,
        "image_name": data["image_name"],
        "image_size": (h, w),
    }


def load_camera_jsons_from_range(
    data_dir: Path,
    start_frame: int,
    end_frame: int,
    pad_width: int = 5,
) -> PoseData:
    extrinsics_list = []
    intrinsics_list = []
    image_size = None

    for fidx in range(start_frame, end_frame + 1):
        fname = f"{fidx:0{pad_width}d}_camera.json"
        cam_path = data_dir / fname
        if not cam_path.exists():
            print(f"Warning: missing camera file {cam_path}, skipping")
            continue

        cdata = load_single_camera_json(cam_path)
        extrinsics_list.append(cdata["extrinsics"])
        intrinsics_list.append(cdata["intrinsics"])
        if image_size is None:
            image_size = cdata["image_size"]

    if len(extrinsics_list) == 0:
        raise FileNotFoundError(
            f"No camera.json files found in {data_dir} for frames [{start_frame}, {end_frame}]"
        )

    return PoseData(
        image_paths=[],
        extrinsics=torch.stack(extrinsics_list),
        intrinsics=torch.stack(intrinsics_list),
        image_size=image_size,
    )


def extract_frame_id(filename: str) -> int:
    import re

    stem = Path(filename).stem
    match = re.search(r"(\d+)", stem)
    if match is None:
        raise ValueError(f"Cannot extract frame ID from {filename}")
    return int(match.group(1))
