#!/usr/bin/env python3
"""Build a portable visualization of final metrics and scene-level predictions."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "src" / "evaluation_showcase_template"
VERIFIED_METRICS = PROJECT_ROOT / "temp" / "final_model_metrics_verified.json"
SCANNET_ROOT = Path("/data/wy/data/scannetpp_audited/pipeline_processed")
EGO4D_ROOT = Path("/data/wy/data/ego4d_audited/pipeline_processed")
SCANNET_GS_ROOT = Path("/data/wy/data/scannet++/scannetpp_gs")

DEFAULT_SCENES = (
    {
        "id": "e8e81396b6",
        "source": "scannetpp",
        "title": "Office and storage room",
        "note": "Official ScanNet++ 3DGS with clear object scale and occlusion cues.",
    },
    {
        "id": "ego4d_a1cf9215_73b91ff616",
        "source": "ego4d",
        "title": "Kitchen workspace",
        "note": "Continuous RGB surfel surface reconstructed from G3T-Long geometry.",
    },
    {
        "id": "ego4d_b6582e60_79f17d9a07",
        "source": "ego4d",
        "title": "Living-room conversation",
        "note": "A wider scene for comparing position, size, and heading.",
    },
)

MODEL_KEYS = {
    "Qwen3-VL-8B": ("qwen3_vl_8b", "open-weight"),
    "InternVL3.5-8B": ("internvl35_8b", "open-weight"),
    "Gemma4-12B": ("gemma4_12b", "open-weight"),
    "SenseNova-SI-8B": ("sensenova_si_8b", "open-weight"),
    "SpatialStack-4B": ("spatialstack_4b", "open-weight"),
    "Cambrian-S-7B": ("cambrian_s_7b", "open-weight"),
    "Claude Sonnet 5": ("claude_sonnet_5", "api"),
    "Gemini 3.1 Pro": ("gemini_31_pro", "api"),
    "GPT-5.6 Sol": ("gpt_56_sol", "api"),
    "Qwen3.7 Plus": ("qwen37_plus", "api"),
}

METRIC_ORDER = (
    "match_rate",
    "visibility",
    "cross_view_identity",
    "position",
    "size",
    "heading_within_30",
)

POSITION_PROXY_FACTORS = {-2: -0.95, -1: -0.425, 0: 0.0, 1: 0.425, 2: 0.95}
SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "temp" / "evaluation_showcase_v2",
    )
    parser.add_argument("--verified-metrics", type=Path, default=VERIFIED_METRICS)
    parser.add_argument("--max-ego-surfels", type=int, default=120_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def extract_explorer_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const\s+DATA\s*=\s*", text)
    if not match:
        raise ValueError(f"Cannot find embedded DATA payload in {path}")
    payload, _ = json.JSONDecoder().raw_decode(text[match.end() :])
    return payload


def parse_rgb(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(max(0, min(255, int(round(float(channel))))) for channel in value[:3])
    if isinstance(value, str):
        channels = re.findall(r"[-+]?\d*\.?\d+", value)
        if len(channels) >= 3:
            return tuple(max(0, min(255, int(round(float(channel))))) for channel in channels[:3])
    return (170, 178, 175)


def point_cloud_bounds(cloud: dict[str, Any]) -> dict[str, Any]:
    np = _load_numpy()
    points = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("Explorer point cloud has no finite points")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm((maximum - minimum) * 0.5))
    return {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "center": center.tolist(),
        "radius": radius,
    }


def normalized_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def match_predictions(
    queries: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, dict[str, Any] | None]:
    matches: dict[str, dict[str, Any] | None] = {query["id"]: None for query in queries}
    used: set[int] = set()
    for query in queries:
        for index, prediction in enumerate(predictions):
            if index not in used and prediction.get("id") == query["id"]:
                matches[query["id"]] = prediction
                used.add(index)
                break
    for query in queries:
        if matches[query["id"]] is not None:
            continue
        expected = normalized_label(query.get("label"))
        for index, prediction in enumerate(predictions):
            if index not in used and normalized_label(prediction.get("label")) == expected:
                matches[query["id"]] = prediction
                used.add(index)
                break
    return matches


def observed_iou(expected: Any, predicted: Any) -> float:
    expected_set = {int(value) for value in expected or []}
    predicted_set = {int(value) for value in predicted or []}
    union = expected_set | predicted_set
    return len(expected_set & predicted_set) / len(union) if union else 1.0


def axis_accuracy(expected: Any, predicted: Any, axes: tuple[str, ...]) -> float:
    if not isinstance(expected, dict) or not isinstance(predicted, dict):
        return 0.0
    return sum(predicted.get(axis) == expected.get(axis) for axis in axes) / len(axes)


def horizontal_heading_angle(expected: Any, predicted: Any) -> float | None:
    if not isinstance(expected, dict):
        return None
    if not isinstance(predicted, dict):
        return 180.0
    expected_vector = (float(expected.get("x", 0)), float(expected.get("z", 0)))
    predicted_vector = (float(predicted.get("x", 0)), float(predicted.get("z", 0)))
    expected_norm = math.hypot(*expected_vector)
    predicted_norm = math.hypot(*predicted_vector)
    if expected_norm <= 1e-9 or predicted_norm <= 1e-9:
        return 180.0
    cosine = sum(a * b for a, b in zip(expected_vector, predicted_vector)) / (
        expected_norm * predicted_norm
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def score_prediction(
    ground_truth: dict[str, Any], prediction: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    matched = prediction is not None
    visible_expected = ground_truth.get("observed_in") or []
    visibility = observed_iou(
        visible_expected, prediction.get("observed_in") if matched else []
    )
    metrics: dict[str, dict[str, Any]] = {
        "visibility": {"value": visibility, "note": "可见视图集合 IoU"},
        "cross_view_identity": {
            "value": float(
                matched
                and set(prediction.get("observed_in") or []) == set(visible_expected)
            )
            if len(visible_expected) >= 2
            else None,
            "note": "跨视角集合完全一致" if len(visible_expected) >= 2 else "单视角目标",
        },
        "position": {
            "value": axis_accuracy(
                ground_truth.get("position_code"),
                prediction.get("position_code") if matched else None,
                ("x", "y", "z"),
            ),
            "note": "位置三轴准确率",
        },
        "size": {
            "value": axis_accuracy(
                ground_truth.get("size_code"),
                prediction.get("size_code") if matched else None,
                ("w", "h", "d"),
            ),
            "note": "尺寸三轴准确率",
        },
    }
    angle = horizontal_heading_angle(
        ground_truth.get("heading_code"),
        prediction.get("heading_code") if matched else None,
    )
    metrics["heading"] = {
        "value": float(angle <= 30.0) if angle is not None else None,
        "note": f"水平夹角 {angle:.1f}°" if angle is not None else "GT 朝向不可评测",
        "angle_degrees": angle,
    }
    return metrics


def object_index(objects_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in objects_payload.get("objects", []):
        for key in ("track_id", "object_id", "annotation_id"):
            value = item.get(key)
            if value is not None:
                index.setdefault(str(value), item)
    return index


def transform_geometry(
    instance: dict[str, Any], scene_objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    source_id = instance.get("source_object_id")
    obj = scene_objects.get(str(source_id), {})
    aabb = (obj.get("bbox_3d") or {}).get("aabb") or {}
    center_source = aabb.get("center") or obj.get("center_3d")
    size_source = aabb.get("size")
    orientation = obj.get("orientation_3d") or {}
    heading_source = orientation.get("forward")
    center = None if not center_source else [float(value) for value in center_source]
    size = None if not size_source else [float(value) for value in size_source]
    heading = None if not heading_source else [float(value) for value in heading_source]
    return {
        "source_object_id": source_id,
        "center": center,
        "size": size,
        "heading": heading,
        "heading_available": bool(orientation.get("available") and heading),
        "heading_evaluable": bool(instance.get("heading_evaluable")),
        "orientation_confidence": orientation.get("confidence"),
    }


def decode_prediction_geometry(
    prediction: dict[str, Any] | None, ground_truth_payload: dict[str, Any]
) -> dict[str, Any] | None:
    if not prediction:
        return None
    position = prediction.get("position_code")
    size = prediction.get("size_code")
    if not isinstance(position, dict) or not isinstance(size, dict):
        return None
    try:
        scales = ground_truth_payload["coding_policy"]["position"]["scale"]
        origin = ground_truth_payload["coordinate_frame"]["origin_before_centering"]
        center = [
            float(origin[index])
            + POSITION_PROXY_FACTORS[int(position[axis])] * float(scales[index])
            for index, axis in enumerate(("x", "y", "z"))
        ]
        thresholds = ground_truth_payload["coding_policy"]["size"]["thresholds_per_axis"]
        dimensions = []
        for axis, code in zip(("w", "h", "d"), (size["w"], size["h"], size["d"])):
            if code is None:
                return None
            low, high = (float(value) for value in thresholds[axis])
            code = int(code)
            if code == 1:
                dimensions.append(max(low * 0.70, 1e-3))
            elif code == 2:
                dimensions.append((low + high) * 0.5)
            elif code == 3:
                dimensions.append(high + max((high - low) * 0.5, high * 0.25))
            else:
                return None
    except (KeyError, TypeError, ValueError):
        return None
    heading_code = prediction.get("heading_code")
    heading = None
    if isinstance(heading_code, dict):
        vector = [float(heading_code.get(axis, 0)) for axis in ("x", "y", "z")]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 1e-9:
            heading = [value / norm for value in vector]
    return {
        "center": center,
        "size": dimensions,
        "heading": heading,
        "is_code_proxy": True,
    }


def find_scene_metric(
    metrics_payload: dict[str, Any], scene_id: str
) -> dict[str, Any] | None:
    return next(
        (scene for scene in metrics_payload.get("scenes", []) if scene.get("scene_id") == scene_id),
        None,
    )


def representative_scene_score(query_results: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for result in query_results:
        for metric in result.get("metrics", {}).values():
            value = metric.get("value")
            if value is not None:
                values.append(float(value))
    return sum(values) / len(values) if values else None


def matrix_to_quaternion(rotation: list[list[float]]) -> list[float]:
    # Return Three.js order [x, y, z, w] for a proper 3x3 rotation matrix.
    m = rotation
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return [x / norm, y / norm, z / norm, w / norm]


def export_ego_gaussian_ply(
    cloud: dict[str, Any], output_path: Path, max_surfels: int, seed: int = 0
) -> dict[str, Any]:
    np = _load_numpy()
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise ImportError("Ego4D RGB surfel export requires scipy") from error

    points = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float32)
    colors = np.asarray([parse_rgb(value) for value in cloud["rgb"]], dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if points.shape[0] > max_surfels:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(points.shape[0], max_surfels, replace=False))
        points = points[keep]
        colors = colors[keep]
    if points.shape[0] < 32:
        raise ValueError("Not enough finite Ego4D points for surfel reconstruction")

    neighbor_count = min(10, points.shape[0])
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=neighbor_count, workers=-1)
    neighborhoods = points[neighbors[:, 1:]]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(neighbor_count - 2, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].astype(np.float32)
    toward_camera = -points
    flip = np.einsum("ij,ij->i", normals, toward_camera) < 0
    normals[flip] *= -1.0

    radii = np.median(distances[:, 1 : min(5, neighbor_count)], axis=1).astype(np.float32)
    valid_radii = radii[np.isfinite(radii) & (radii > 0)]
    low, high = np.quantile(valid_radii, [0.02, 0.98])
    radii = np.clip(radii, max(float(low), 1e-5), max(float(high), float(low) * 1.1))
    tangent_scale = radii * 1.55
    normal_scale = radii * 0.24

    # Quaternion rotating local +Z onto each estimated surface normal; PLY stores w,x,y,z.
    nz = np.clip(normals[:, 2], -1.0, 1.0)
    quaternion = np.zeros((len(points), 4), dtype=np.float32)
    regular = nz > -0.9999
    quaternion[regular, 0] = np.sqrt((1.0 + nz[regular]) * 0.5)
    denominator = np.maximum(2.0 * quaternion[regular, 0], 1e-8)
    quaternion[regular, 1] = -normals[regular, 1] / denominator
    quaternion[regular, 2] = normals[regular, 0] / denominator
    quaternion[~regular, 2] = 1.0
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-8)

    dtype = np.dtype(
        [(name, "<f4") for name in (
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        )]
    )
    records = np.zeros(len(points), dtype=dtype)
    for index, axis in enumerate(("x", "y", "z")):
        records[axis] = points[:, index]
    rgb = colors.astype(np.float32) / 255.0
    for index, field in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        records[field] = (rgb[:, index] - 0.5) / SH_C0
    records["opacity"] = 2.35
    records["scale_0"] = np.log(np.maximum(tangent_scale, 1e-7))
    records["scale_1"] = np.log(np.maximum(tangent_scale, 1e-7))
    records["scale_2"] = np.log(np.maximum(normal_scale, 1e-7))
    for index, field in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        records[field] = quaternion[:, index]

    fields = [name for name, _ in dtype.descr]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(records)}"]
    header.extend(f"property float {field}" for field in fields)
    header.append("end_header")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        records.tofile(handle)
    return {
        "count": int(len(records)),
        "radius_median": float(np.median(radii)),
        "radius_range": [float(np.min(radii)), float(np.max(radii))],
    }


def prepare_render_asset(
    source: str,
    scene_id: str,
    root: Path,
    point_payload: dict[str, Any],
    output_dir: Path,
    max_ego_surfels: int,
) -> dict[str, Any]:
    asset_dir = output_dir / "assets" / "scenes" / scene_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    bounds = point_cloud_bounds(point_payload["point_cloud"])
    if source == "scannetpp":
        source_asset = SCANNET_GS_ROOT / scene_id / "point_cloud_30000_lod550k_l1_sh1.ksplat"
        if not source_asset.is_file():
            raise FileNotFoundError(
                f"Missing official ScanNet++GS asset: {source_asset}. "
                "Download point_cloud_30000.ply and convert it to KSplat first."
            )
        destination = asset_dir / "scene.ksplat"
        shutil.copy2(source_asset, destination)
        metadata = load_json(root / "scannetpp_metadata.json")
        transform = metadata["coordinate_alignment"]["source_to_canonical"]
        rotation = [[float(transform[row][column]) for column in range(3)] for row in range(3)]
        position = [float(transform[row][3]) for row in range(3)]
        return {
            "kind": "gaussian_splat",
            "asset": f"assets/scenes/{scene_id}/scene.ksplat",
            "format": "ksplat",
            "renderer_label": "ScanNet++ official 3DGS",
            "count": 550_000,
            "spherical_harmonics_degree": 1,
            "splat_scale": 1.55,
            "transform": {
                "position": position,
                "rotation": matrix_to_quaternion(rotation),
                "scale": [1.0, 1.0, 1.0],
            },
            "bounds": bounds,
            "camera_up": [0.0, -1.0, 0.0],
            "anchor_camera": {
                "position": [0.0, 0.0, 0.0],
                "look_at": [0.0, 0.0, 2.0],
                "fov_y": 53.6,
            },
        }

    destination = asset_dir / "scene_rgb_surfels.ply"
    diagnostics = export_ego_gaussian_ply(
        point_payload["point_cloud"], destination, max_surfels=max_ego_surfels
    )
    return {
        "kind": "rgb_surfel_splat",
        "asset": f"assets/scenes/{scene_id}/scene_rgb_surfels.ply",
        "format": "ply",
        "renderer_label": "G3T-Long RGB surface",
        "count": diagnostics["count"],
        "spherical_harmonics_degree": 0,
        "splat_scale": 1.0,
        "transform": {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": bounds,
        "camera_up": [0.0, -1.0, 0.0],
        "anchor_camera": {
            "position": [0.0, 0.0, 0.0],
            "look_at": [0.0, 0.0, max(1.0, min(2.5, float(bounds["radius"]) * 0.55))],
            "fov_y": 60.0,
        },
        "surfel_diagnostics": diagnostics,
    }


def prepare_scene(
    scene_spec: dict[str, str],
    model_records: list[dict[str, Any]],
    metrics_cache: dict[Path, dict[str, Any]],
    output_dir: Path,
    max_ego_surfels: int,
) -> dict[str, Any]:
    scene_id = scene_spec["id"]
    source = scene_spec["source"]
    root = (SCANNET_ROOT if source == "scannetpp" else EGO4D_ROOT) / scene_id
    explorer = root / (
        "scannetpp_explorer.html" if source == "scannetpp" else "g3t_long_explorer.html"
    )
    oriented_path = root / "objects_oriented.json"
    if not explorer.is_file() or not oriented_path.is_file():
        raise FileNotFoundError(f"Incomplete scene assets: {root}")

    first_metrics_path = Path(model_records[0]["sources"][source]["metrics_path"])
    first_metrics = metrics_cache.setdefault(first_metrics_path, load_json(first_metrics_path))
    scene_metric = find_scene_metric(first_metrics, scene_id)
    if not scene_metric:
        raise KeyError(f"Scene {scene_id} is missing from {first_metrics_path}")
    ground_truth_path = (
        first_metrics_path.parent
        / ".."
        / "shared"
        / "ground_truth"
        / source
        / f"{scene_id}.json"
    )
    ground_truth = load_json(ground_truth_path.resolve())

    point_payload = extract_explorer_payload(explorer)
    render_asset = prepare_render_asset(
        source, scene_id, root, point_payload, output_dir, max_ego_surfels
    )

    image_dir = output_dir / "assets" / "images" / scene_id
    image_dir.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []
    for index, name in enumerate(ground_truth.get("images", [])):
        source_image = root / "multiview_images" / name
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        destination = image_dir / name
        shutil.copy2(source_image, destination)
        views.append(
            {
                "index": index + 1,
                "label": "View 1 · coordinate reference" if index == 0 else f"View {index + 1}",
                "image": f"assets/images/{scene_id}/{name}",
                "name": name,
            }
        )

    oriented = load_json(oriented_path)
    scene_objects = object_index(oriented)
    target_objects = {item["id"]: item for item in ground_truth["target"]["objects"]}
    instances = {item["id"]: item for item in ground_truth.get("instances", [])}
    query_specs = scene_metric["query"]["objects"]
    queries: list[dict[str, Any]] = []
    for query_spec in query_specs:
        query_id = query_spec["id"]
        instance = instances.get(query_id, {})
        queries.append(
            {
                "id": query_id,
                "label": query_spec["label"],
                "label_frequency": query_spec.get("label_frequency", 1),
                "ground_truth": target_objects[query_id],
                "geometry": transform_geometry(instance, scene_objects),
                "predictions": {},
            }
        )

    model_results: dict[str, Any] = {}
    for model in model_records:
        model_source = model["sources"][source]
        metrics_path = Path(model_source["metrics_path"])
        metrics_payload = metrics_cache.setdefault(metrics_path, load_json(metrics_path))
        per_scene = find_scene_metric(metrics_payload, scene_id)
        response_path = metrics_path.parent / "responses" / source / f"{scene_id}.json"
        response = load_json(response_path) if response_path.is_file() else {}
        predictions = (response.get("parsed_output") or {}).get("objects") or []
        matched = match_predictions(queries, predictions)
        query_results: list[dict[str, Any]] = []
        for query in queries:
            prediction = matched[query["id"]]
            result = {
                "matched": prediction is not None,
                "object": prediction,
                "proxy_geometry": decode_prediction_geometry(prediction, ground_truth),
                "metrics": score_prediction(query["ground_truth"], prediction),
            }
            query["predictions"][model["key"]] = result
            query_results.append(result)
        model_results[model["key"]] = {
            "evaluation_status": per_scene.get("evaluation_status") if per_scene else "missing",
            "representative_score": representative_scene_score(query_results),
            "latency_seconds": response.get("latency_seconds"),
        }

    scene_output = {
        "id": scene_id,
        "source": source,
        "title": scene_spec["title"],
        "note": scene_spec["note"],
        "coordinate_system": "+X right, +Y gravity/down, +Z View 1 forward",
        "views": views,
        "render": render_asset,
        "queries": queries,
        "model_results": model_results,
    }
    scene_path = output_dir / "data" / "scenes" / f"{scene_id}.json"
    dump_json(scene_path, scene_output)
    return {
        "id": scene_id,
        "title": scene_spec["title"],
        "data": f"data/scenes/{scene_id}.json",
        "renderer_label": render_asset["renderer_label"],
        "render_count": render_asset["count"],
        "query_count": len(queries),
    }


def prepare_models(verified: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reference_counts: tuple[int, int] | None = None
    for name, verified_record in verified["models"].items():
        key, kind = MODEL_KEYS[name]
        scan_count = int(verified_record["scannetpp"]["support"]["scenes"])
        ego_count = int(verified_record["ego4d"]["support"]["scenes"])
        if reference_counts is None:
            reference_counts = (scan_count, ego_count)
        elif reference_counts != (scan_count, ego_count):
            raise ValueError(f"Inconsistent dataset scene support for {name}")
        total = scan_count + ego_count
        combined = {
            metric: (
                scan_count * float(verified_record["scannetpp"]["metrics"][metric])
                + ego_count * float(verified_record["ego4d"]["metrics"][metric])
            )
            / total
            for metric in METRIC_ORDER
        }
        records.append(
            {
                "key": key,
                "name": name,
                "kind": kind,
                "metrics": {"combined": combined},
                "sources": {
                    "scannetpp": {
                        "metrics_path": verified_record["scannetpp"]["metrics_path"],
                        "support": verified_record["scannetpp"]["support"],
                    },
                    "ego4d": {
                        "metrics_path": verified_record["ego4d"]["metrics_path"],
                        "support": verified_record["ego4d"]["support"],
                    },
                },
            }
        )
    return records


def copy_template(output_dir: Path) -> None:
    for name in ("index.html", "styles.css", "app.js", "README.md"):
        shutil.copy2(TEMPLATE_DIR / name, output_dir / name)
    shutil.copytree(TEMPLATE_DIR / "vendor", output_dir / "vendor", dirs_exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "VISUALIZATION_DESIGN.md", output_dir / "VISUALIZATION_DESIGN.md")


def build(
    output_dir: Path, verified_metrics: Path, overwrite: bool, max_ego_surfels: int
) -> dict[str, Any]:
    if max_ego_surfels < 10_000:
        raise ValueError("--max-ego-surfels must be at least 10000")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_template(output_dir)

    verified = load_json(verified_metrics)
    models = prepare_models(verified)
    metrics_cache: dict[Path, dict[str, Any]] = {}
    scenes = [
        prepare_scene(spec, models, metrics_cache, output_dir, max_ego_surfels)
        for spec in DEFAULT_SCENES
    ]

    scan_support = models[0]["sources"]["scannetpp"]["support"]
    ego_support = models[0]["sources"]["ego4d"]["support"]
    summary = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_order": list(METRIC_ORDER),
        "benchmark": {
            "scenes": int(scan_support["scenes"]) + int(ego_support["scenes"]),
            "queries": int(scan_support["queried_objects"]) + int(ego_support["queried_objects"]),
        },
        "models": [
            {key: value for key, value in model.items() if key != "sources"}
            for model in models
        ],
        "scenes": scenes,
    }
    dump_json(output_dir / "data" / "summary.json", summary)
    return summary


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise ImportError("Evaluation showcase generation requires numpy") from error
    return np


def main() -> None:
    args = parse_args()
    summary = build(
        args.output_dir.resolve(),
        args.verified_metrics.resolve(),
        args.overwrite,
        args.max_ego_surfels,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "models": len(summary["models"]),
                "scenes": len(summary["scenes"]),
                "render_primitives": sum(scene["render_count"] for scene in summary["scenes"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
