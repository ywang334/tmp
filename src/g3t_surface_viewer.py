"""Interactive RGB surface visualization for G3T/G3T-Long outputs.

The G3T point map is image-structured, so adjacent pixels can be connected into
triangles. Long 3D edges are rejected to avoid bridging foreground/background
discontinuities. The result is a compact multi-view surface viewer rather than
a sparse point cloud.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.visualization import add_object_overlays, load_object_records


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BACKGROUND = "#f4f5f7"
DEFAULT_SCENE_BACKGROUND = "#fafbfc"


def visualize_g3t_surface(
    g3t_dir: str | Path,
    output_dir: str | Path | None = None,
    frame_dir: str | Path | None = None,
    output_name: str = "g3t_surface",
    max_keyframes: int = 6,
    pixel_stride: int = 5,
    min_point_conf: float = 1.0,
    max_edge_factor: float = 3.5,
    mesh_opacity: float = 0.96,
    show_cameras: bool = True,
    objects_json_path: str | Path | None = None,
    show_object_bboxes: bool = False,
    show_object_orientations: bool = False,
) -> dict[str, Any]:
    """Create a continuous RGB surface view from saved G3T point maps."""

    if max_keyframes < 1:
        raise ValueError("max_keyframes must be greater than or equal to 1.")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be greater than or equal to 1.")
    if min_point_conf < 0.0:
        raise ValueError("min_point_conf must be greater than or equal to 0.")
    if max_edge_factor <= 0.0:
        raise ValueError("max_edge_factor must be greater than 0.")
    if not 0.0 < mesh_opacity <= 1.0:
        raise ValueError("mesh_opacity must be in (0, 1].")

    np, go = _load_dependencies()
    g3t_dir = Path(g3t_dir)
    output_dir = Path(output_dir) if output_dir is not None else g3t_dir.parent
    frame_dir = Path(frame_dir) if frame_dir is not None else None
    point_map_path = g3t_dir / "point_map.npy"
    point_conf_path = g3t_dir / "point_conf.npy"
    _require_file(point_map_path)
    _require_file(point_conf_path)

    point_map = np.load(point_map_path, mmap_mode="r")
    point_conf = np.load(point_conf_path, mmap_mode="r")
    _validate_geometry(point_map=point_map, point_conf=point_conf)
    frame_paths = _collect_frame_paths(frame_dir) if frame_dir is not None else []
    keyframes = _select_keyframes(point_map.shape[0], max_keyframes=max_keyframes, np=np)

    meshes: list[dict[str, Any]] = []
    for frame_index in keyframes:
        rgb = _load_frame_rgb(
            frame_paths[frame_index] if frame_index < len(frame_paths) else None,
            width=point_map.shape[2],
            height=point_map.shape[1],
            np=np,
        )
        mesh = _build_frame_mesh(
            points=np.asarray(point_map[frame_index]),
            confidence=np.asarray(point_conf[frame_index]),
            rgb=rgb,
            pixel_stride=pixel_stride,
            min_point_conf=min_point_conf,
            max_edge_factor=max_edge_factor,
            np=np,
        )
        if mesh["num_triangles"] > 0:
            mesh["frame_index"] = int(frame_index)
            meshes.append(mesh)

    if not meshes:
        raise ValueError("No valid G3T surface triangles passed the configured filters.")

    camera_poses = _load_optional_array(g3t_dir / "camera_pose.npy", np=np)
    intrinsics = _load_optional_array(g3t_dir / "intrinsic.npy", np=np)
    figure = _build_surface_figure(
        meshes=meshes,
        camera_poses=camera_poses,
        intrinsics=intrinsics,
        keyframes=keyframes,
        image_shape=point_map.shape[1:3],
        mesh_opacity=mesh_opacity,
        show_cameras=show_cameras,
        title="G3T-Long RGB Surface Reconstruction" if "long" in g3t_dir.name else "G3T RGB Surface Reconstruction",
        go=go,
        np=np,
    )
    overlay_objects: list[dict[str, Any]] = []
    if objects_json_path is not None:
        overlay_objects = load_object_records(objects_json_path)
        if show_object_bboxes or show_object_orientations:
            add_object_overlays(
                figure=figure,
                objects=overlay_objects,
                show_bboxes=show_object_bboxes,
                show_orientations=show_object_orientations,
                scene_overlay=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{output_name}.html"
    report_path = output_dir / f"{output_name}_report.json"
    figure.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    report = {
        "geometry_backend": "g3t_long" if "long" in g3t_dir.name else "g3t",
        "visualization": "structured_multi_view_rgb_surface",
        "paths": {
            "g3t_dir": str(g3t_dir),
            "frame_dir": str(frame_dir) if frame_dir is not None else None,
            "html": str(html_path),
        },
        "geometry_shape": {
            "point_map": list(point_map.shape),
            "point_conf": list(point_conf.shape),
        },
        "parameters": {
            "max_keyframes": max_keyframes,
            "pixel_stride": pixel_stride,
            "min_point_conf": min_point_conf,
            "max_edge_factor": max_edge_factor,
            "mesh_opacity": mesh_opacity,
            "show_cameras": show_cameras,
        },
        "selected_keyframes": [int(index) for index in keyframes],
        "num_meshes": len(meshes),
        "num_vertices": int(sum(mesh["num_vertices"] for mesh in meshes)),
        "num_triangles": int(sum(mesh["num_triangles"] for mesh in meshes)),
        "object_overlay": {
            "objects_json": str(objects_json_path) if objects_json_path is not None else None,
            "object_count": len(overlay_objects),
            "bbox_count": sum(1 for obj in overlay_objects if obj.get("bbox_3d") is not None),
            "orientation_count": sum(1 for obj in overlay_objects if obj.get("orientation_3d") is not None),
            "show_bboxes": bool(show_object_bboxes),
            "show_orientations": bool(show_object_orientations),
        },
        "per_frame": [
            {
                "frame_index": int(mesh["frame_index"]),
                "num_vertices": int(mesh["num_vertices"]),
                "num_triangles": int(mesh["num_triangles"]),
                "median_neighbor_edge": float(mesh["median_neighbor_edge"]),
                "max_triangle_edge": float(mesh["max_triangle_edge"]),
            }
            for mesh in meshes
        ],
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return {"html_path": html_path, "report_path": report_path, "report": report}


def _build_frame_mesh(
    points: Any,
    confidence: Any,
    rgb: Any,
    pixel_stride: int,
    min_point_conf: float,
    max_edge_factor: float,
    np: Any,
) -> dict[str, Any]:
    """Triangulate one image-structured point map with discontinuity rejection."""

    points = np.asarray(points[::pixel_stride, ::pixel_stride], dtype=float)
    confidence = np.asarray(confidence[::pixel_stride, ::pixel_stride], dtype=float)
    rgb = np.asarray(rgb[::pixel_stride, ::pixel_stride], dtype=np.uint8)
    valid = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(confidence)
        & (confidence >= min_point_conf)
    )

    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    horizontal_edges = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)[horizontal_valid]
    vertical_edges = np.linalg.norm(points[1:] - points[:-1], axis=-1)[vertical_valid]
    edge_values = np.concatenate([horizontal_edges, vertical_edges])
    edge_values = edge_values[np.isfinite(edge_values) & (edge_values > 0.0)]
    if edge_values.size == 0:
        return _empty_mesh(np=np)
    median_edge = float(np.median(edge_values))
    max_triangle_edge = max(median_edge * max_edge_factor, 1e-6)

    grid_ids = np.full(valid.shape, -1, dtype=np.int64)
    compact_points = points[valid]
    compact_rgb = rgb[valid]
    grid_ids[valid] = np.arange(compact_points.shape[0], dtype=np.int64)

    tl = points[:-1, :-1]
    tr = points[:-1, 1:]
    bl = points[1:, :-1]
    br = points[1:, 1:]
    v_tl = valid[:-1, :-1]
    v_tr = valid[:-1, 1:]
    v_bl = valid[1:, :-1]
    v_br = valid[1:, 1:]

    tri1_valid = v_tl & v_tr & v_bl
    tri1_valid &= _triangle_edges_within(tl, tr, bl, max_triangle_edge, np=np)
    tri2_valid = v_tr & v_br & v_bl
    tri2_valid &= _triangle_edges_within(tr, br, bl, max_triangle_edge, np=np)

    tri1 = np.stack(
        [grid_ids[:-1, :-1][tri1_valid], grid_ids[:-1, 1:][tri1_valid], grid_ids[1:, :-1][tri1_valid]],
        axis=1,
    ) if np.any(tri1_valid) else np.empty((0, 3), dtype=np.int64)
    tri2 = np.stack(
        [grid_ids[:-1, 1:][tri2_valid], grid_ids[1:, 1:][tri2_valid], grid_ids[1:, :-1][tri2_valid]],
        axis=1,
    ) if np.any(tri2_valid) else np.empty((0, 3), dtype=np.int64)
    triangles = np.concatenate([tri1, tri2], axis=0)
    colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in compact_rgb.tolist()]
    return {
        "vertices": compact_points,
        "triangles": triangles,
        "colors": colors,
        "num_vertices": int(compact_points.shape[0]),
        "num_triangles": int(triangles.shape[0]),
        "median_neighbor_edge": median_edge,
        "max_triangle_edge": max_triangle_edge,
    }


def _triangle_edges_within(a: Any, b: Any, c: Any, threshold: float, np: Any) -> Any:
    return (
        (np.linalg.norm(a - b, axis=-1) <= threshold)
        & (np.linalg.norm(a - c, axis=-1) <= threshold)
        & (np.linalg.norm(b - c, axis=-1) <= threshold)
    )


def _empty_mesh(np: Any) -> dict[str, Any]:
    return {
        "vertices": np.empty((0, 3), dtype=float),
        "triangles": np.empty((0, 3), dtype=np.int64),
        "colors": [],
        "num_vertices": 0,
        "num_triangles": 0,
        "median_neighbor_edge": 0.0,
        "max_triangle_edge": 0.0,
    }


def _build_surface_figure(
    meshes: list[dict[str, Any]],
    camera_poses: Any | None,
    intrinsics: Any | None,
    keyframes: list[int],
    image_shape: tuple[int, int],
    mesh_opacity: float,
    show_cameras: bool,
    title: str,
    go: Any,
    np: Any,
) -> Any:
    figure = go.Figure()
    for mesh_index, mesh in enumerate(meshes):
        vertices = mesh["vertices"]
        triangles = mesh["triangles"]
        frame_index = int(mesh["frame_index"])
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                vertexcolor=mesh["colors"],
                opacity=mesh_opacity,
                flatshading=False,
                name=f"RGB surfaces (frame {frame_index})",
                legendgroup="rgb-surfaces",
                showlegend=mesh_index == 0,
                hoverinfo="skip",
                lighting={
                    "ambient": 0.88,
                    "diffuse": 0.62,
                    "specular": 0.06,
                    "roughness": 0.92,
                    "fresnel": 0.04,
                },
                lightposition={"x": 0.0, "y": -4.0, "z": 8.0},
            )
        )

    surface_trace_count = len(figure.data)
    camera_trace_count = 0
    all_vertices = np.concatenate([mesh["vertices"] for mesh in meshes], axis=0)
    scene_scale = _robust_scene_scale(all_vertices, np=np)
    if show_cameras and camera_poses is not None:
        poses = _normalize_camera_pose_array(camera_poses, np=np)
        centers = _camera_centers(poses, np=np)
        figure.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="lines+markers",
                name="camera path",
                line={"color": "#d1495b", "width": 4},
                marker={"color": "#d1495b", "size": 3},
                hovertemplate="camera %{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
                text=[str(index) for index in range(centers.shape[0])],
            )
        )
        camera_trace_count += 1
        frustum = _camera_frustum_coordinates(
            poses=poses,
            intrinsics=intrinsics,
            frame_indices=keyframes,
            image_shape=image_shape,
            scale=scene_scale * 0.035,
            np=np,
        )
        figure.add_trace(
            go.Scatter3d(
                x=frustum[0],
                y=frustum[1],
                z=frustum[2],
                mode="lines",
                name="keyframe cameras",
                line={"color": "#d1495b", "width": 2},
                opacity=0.8,
                hoverinfo="skip",
            )
        )
        camera_trace_count += 1

    visibility_all = [True] * (surface_trace_count + camera_trace_count)
    visibility_surface = [True] * surface_trace_count + [False] * camera_trace_count
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        paper_bgcolor=DEFAULT_BACKGROUND,
        plot_bgcolor=DEFAULT_BACKGROUND,
        font={"family": "Arial, sans-serif", "color": "#20242a"},
        scene={
            "xaxis": {"title": "X", "backgroundcolor": DEFAULT_SCENE_BACKGROUND, "gridcolor": "#d9dde3", "zerolinecolor": "#8b929c"},
            "yaxis": {"title": "Y (gravity/down)", "backgroundcolor": DEFAULT_SCENE_BACKGROUND, "gridcolor": "#d9dde3", "zerolinecolor": "#8b929c"},
            "zaxis": {"title": "Z (first-view forward)", "backgroundcolor": DEFAULT_SCENE_BACKGROUND, "gridcolor": "#d9dde3", "zerolinecolor": "#8b929c"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.35, "y": -1.15, "z": 1.05}},
        },
        margin={"l": 0, "r": 0, "t": 58, "b": 0},
        legend={"orientation": "h", "x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.78)"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.02,
                "y": 1.08,
                "showactive": True,
                "buttons": [
                    {"label": "Surface + cameras", "method": "update", "args": [{"visible": visibility_all}]},
                    {"label": "Surface only", "method": "update", "args": [{"visible": visibility_surface}]},
                ],
            }
        ] if camera_trace_count else [],
        annotations=[
            {
                "text": f"{len(meshes)} keyframes | {sum(mesh['num_triangles'] for mesh in meshes):,} triangles",
                "xref": "paper",
                "yref": "paper",
                "x": 0.99,
                "y": 1.04,
                "xanchor": "right",
                "showarrow": False,
                "font": {"size": 12, "color": "#555c66"},
            }
        ],
    )
    return figure


def _camera_frustum_coordinates(
    poses: Any,
    intrinsics: Any | None,
    frame_indices: list[int],
    image_shape: tuple[int, int],
    scale: float,
    np: Any,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    height, width = image_shape
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for frame_index in frame_indices:
        if frame_index >= poses.shape[0]:
            continue
        pose = poses[frame_index]
        rotation = pose[:3, :3]
        center = -rotation.T @ pose[:3, 3]
        if intrinsics is not None:
            intrinsic = np.asarray(intrinsics[min(frame_index, len(intrinsics) - 1)])
            fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
            cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
            pixels = [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]
            rays = [np.asarray([(u - cx) / max(fx, 1e-8), (v - cy) / max(fy, 1e-8), 1.0]) for u, v in pixels]
        else:
            rays = [np.asarray([-0.65, -0.45, 1.0]), np.asarray([0.65, -0.45, 1.0]), np.asarray([0.65, 0.45, 1.0]), np.asarray([-0.65, 0.45, 1.0])]
        corners = []
        for ray in rays:
            ray = ray / np.linalg.norm(ray) * scale
            corners.append(center + rotation.T @ ray)
        edges = [(center, corner) for corner in corners]
        edges.extend((corners[index], corners[(index + 1) % 4]) for index in range(4))
        for start, end in edges:
            xs.extend([float(start[0]), float(end[0]), None])
            ys.extend([float(start[1]), float(end[1]), None])
            zs.extend([float(start[2]), float(end[2]), None])
    return xs, ys, zs


def _robust_scene_scale(points: Any, np: Any) -> float:
    lower = np.percentile(points, 2.0, axis=0)
    upper = np.percentile(points, 98.0, axis=0)
    return max(float(np.linalg.norm(upper - lower)), 1e-3)


def _normalize_camera_pose_array(camera_pose: Any, np: Any) -> Any:
    poses = np.asarray(camera_pose)
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3:
        raise ValueError(f"camera_pose must have shape [N,3,4] or [N,4,4], got {poses.shape}")
    if poses.shape[1:] == (4, 4):
        return poses.astype(float)
    if poses.shape[1:] == (3, 4):
        normalized = np.tile(np.eye(4), (poses.shape[0], 1, 1)).astype(float)
        normalized[:, :3, :4] = poses
        return normalized
    raise ValueError(f"camera_pose must have shape [N,3,4] or [N,4,4], got {poses.shape}")


def _camera_centers(poses: Any, np: Any) -> Any:
    return np.asarray([-pose[:3, :3].T @ pose[:3, 3] for pose in poses], dtype=float)


def _select_keyframes(num_frames: int, max_keyframes: int, np: Any) -> list[int]:
    count = min(num_frames, max_keyframes)
    return sorted(set(int(value) for value in np.linspace(0, num_frames - 1, count)))


def _collect_frame_paths(frame_dir: Path) -> list[Path]:
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")
    paths = sorted(path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No frame images found in {frame_dir}")
    return paths


def _load_frame_rgb(frame_path: Path | None, width: int, height: int, np: Any) -> Any:
    if frame_path is None:
        return np.full((height, width, 3), 180, dtype=np.uint8)
    from PIL import Image

    with Image.open(frame_path) as image:
        return np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def _load_optional_array(path: Path, np: Any) -> Any | None:
    return np.load(path) if path.exists() else None


def _validate_geometry(point_map: Any, point_conf: Any) -> None:
    if point_map.ndim != 4 or point_map.shape[-1] != 3:
        raise ValueError(f"point_map must have shape [N,H,W,3], got {point_map.shape}")
    if point_conf.ndim != 3 or point_map.shape[:3] != point_conf.shape:
        raise ValueError(f"point_conf must match point_map [N,H,W], got {point_conf.shape}")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def _load_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("G3T surface visualization requires numpy and plotly.") from exc
    return np, go


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize G3T point maps as continuous RGB surface meshes.")
    parser.add_argument("--g3t_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--frame_dir", type=Path, default=None)
    parser.add_argument("--output_name", default="g3t_surface")
    parser.add_argument("--max_keyframes", type=int, default=6)
    parser.add_argument("--pixel_stride", type=int, default=5)
    parser.add_argument("--min_point_conf", type=float, default=1.0)
    parser.add_argument("--max_edge_factor", type=float, default=3.5)
    parser.add_argument("--mesh_opacity", type=float, default=0.96)
    parser.add_argument("--hide_cameras", action="store_true")
    parser.add_argument("--objects_json", type=Path, default=None)
    parser.add_argument("--show_object_bboxes", action="store_true")
    parser.add_argument("--show_object_orientations", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = visualize_g3t_surface(
        g3t_dir=args.g3t_dir,
        output_dir=args.output_dir,
        frame_dir=args.frame_dir,
        output_name=args.output_name,
        max_keyframes=args.max_keyframes,
        pixel_stride=args.pixel_stride,
        min_point_conf=args.min_point_conf,
        max_edge_factor=args.max_edge_factor,
        mesh_opacity=args.mesh_opacity,
        show_cameras=not args.hide_cameras,
        objects_json_path=args.objects_json,
        show_object_bboxes=args.show_object_bboxes,
        show_object_orientations=args.show_object_orientations,
    )
    print(json.dumps({"html_path": str(result["html_path"]), "report_path": str(result["report_path"])}, indent=2))


if __name__ == "__main__":
    main()
