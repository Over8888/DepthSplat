from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


GS_PROPERTIES = [
    ("x", "float"), ("y", "float"), ("z", "float"),
    ("nx", "float"), ("ny", "float"), ("nz", "float"),
    ("f_dc_0", "float"), ("f_dc_1", "float"), ("f_dc_2", "float"),
    ("opacity", "float"),
    ("scale_0", "float"), ("scale_1", "float"), ("scale_2", "float"),
    ("rot_0", "float"), ("rot_1", "float"), ("rot_2", "float"), ("rot_3", "float"),
]

PC_PROPERTIES = [
    ("x", "float"), ("y", "float"), ("z", "float"),
    ("red", "float"), ("green", "float"), ("blue", "float"),
    ("confidence", "float"),
]


def write_gaussian_ply(
    path: str | Path,
    centers: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    sh_dc: np.ndarray,
    opacity_threshold: float | None = None,
):
    N = len(centers)
    if opacity_threshold is not None:
        mask = opacities >= opacity_threshold
        centers = centers[mask]
        opacities = opacities[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        sh_dc = sh_dc[mask]
        N = len(centers)

    if N == 0:
        raise ValueError("No gaussians to write (all filtered out)")

    opacities_logit = np.log(np.maximum(opacities, 1e-10) / np.maximum(1 - opacities, 1e-10))
    opacities_logit = np.clip(opacities_logit, -10, 10)
    scales_log = np.log(np.maximum(scales, 1e-10))
    normals = np.zeros((N, 3), dtype=np.float32)

    sh_dc_converted = np.zeros_like(sh_dc)
    C0 = 0.28209479177387814
    sh_dc_converted = (sh_dc - 0.5) / C0

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
    ]
    for name, dtype in GS_PROPERTIES:
        header_lines.append(f"property {dtype} {name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            data = struct.pack(
                "<fffffffffffffffff",
                centers[i, 0], centers[i, 1], centers[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                sh_dc_converted[i, 0], sh_dc_converted[i, 1], sh_dc_converted[i, 2],
                float(opacities_logit[i]),
                scales_log[i, 0], scales_log[i, 1], scales_log[i, 2],
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3],
            )
            f.write(data)

    print(f"[PLY] Wrote {N} gaussians to {path}")


def write_pointcloud_ply(
    path: str | Path,
    centers: np.ndarray,
    colors: np.ndarray,
    confidences: np.ndarray,
    confidence_threshold: float | None = None,
):
    N = len(centers)
    if confidence_threshold is not None:
        mask = confidences >= confidence_threshold
        centers = centers[mask]
        colors = colors[mask]
        confidences = confidences[mask]
        N = len(centers)

    if N == 0:
        raise ValueError("No points to write (all filtered out)")

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
    ]
    for name, dtype in PC_PROPERTIES:
        header_lines.append(f"property {dtype} {name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            data = struct.pack(
                "<fffffff",
                centers[i, 0], centers[i, 1], centers[i, 2],
                colors[i, 0], colors[i, 1], colors[i, 2],
                float(confidences[i]),
            )
            f.write(data)

    print(f"[PLY] Wrote {N} points to {path}")


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            header_lines.append(line)

    properties = []
    num_vertices = 0
    for line in header_lines:
        if line.startswith("element vertex "):
            num_vertices = int(line.split()[-1])
        elif line.startswith("property "):
            parts = line.split()
            dtype = parts[1]
            name = parts[2]
            properties.append((name, dtype))

    fmt_map = {
        "float": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uint": ("I", 4),
        "uchar": ("B", 1),
        "int8": ("b", 1),
    }

    row_fmt = "<" + "".join(fmt_map[d][0] for _, d in properties)
    row_size = struct.calcsize(row_fmt)

    data = {}
    for name, _ in properties:
        data[name] = np.zeros(num_vertices, dtype=np.float32)

    with open(path, "rb") as f:
        header_len = 0
        while True:
            line = f.readline().decode("ascii").strip()
            header_len = f.tell()
            if line == "end_header":
                break

        for i in range(num_vertices):
            row_bytes = f.read(row_size)
            values = struct.unpack(row_fmt, row_bytes)
            for j, (name, _) in enumerate(properties):
                data[name][i] = float(values[j])

    return data


def compute_max_eigenvalue(scales: np.ndarray) -> np.ndarray:
    return np.max(np.abs(scales), axis=1)
