"""Ego4D/EgoTracks subset planning and output finalization helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator


EGO4D_PLAN_VERSION = 3


def build_egotracks_subset_plan(
    annotations_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    limit: int = 24,
    clip_duration_sec: float = 20.0,
    min_track_boxes: int = 12,
    min_median_box_area_ratio: float = 0.003,
    max_source_duration_sec: float = 1800.0,
    max_per_participant: int = 1,
    max_per_scenario: int = 2,
) -> list[dict[str, Any]]:
    """Select a compact, diverse, annotation-rich EgoTracks validation subset."""

    if limit < 1:
        raise ValueError("limit must be at least 1")
    metadata = _load_video_metadata(Path(metadata_path))
    candidates: list[dict[str, Any]] = []
    for video in _iter_annotation_videos(Path(annotations_path)):
        video_uid = str(video.get("video_uid", ""))
        video_meta = metadata.get(video_uid)
        if not video_uid or video_meta is None:
            continue
        best = _best_video_candidate(
            video,
            video_meta,
            clip_duration_sec=clip_duration_sec,
            min_track_boxes=min_track_boxes,
            min_median_box_area_ratio=min_median_box_area_ratio,
            max_source_duration_sec=max_source_duration_sec,
        )
        if best is not None:
            candidates.append(best)

    candidates.sort(key=lambda item: (-item["quality_score"], item["video_uid"]))
    selected: list[dict[str, Any]] = []
    participants: dict[str, int] = {}
    scenarios: dict[str, int] = {}
    for candidate in candidates:
        participant_key = candidate["diversity"]["participant_key"]
        scenario_key = candidate["diversity"]["scenario_key"]
        if participants.get(participant_key, 0) >= max_per_participant:
            continue
        if scenarios.get(scenario_key, 0) >= max_per_scenario:
            continue
        candidate["selection_rank"] = len(selected) + 1
        selected.append(candidate)
        participants[participant_key] = participants.get(participant_key, 0) + 1
        scenarios[scenario_key] = scenarios.get(scenario_key, 0) + 1
        if len(selected) >= limit:
            break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(output_path, selected)
    summary = {
        "version": EGO4D_PLAN_VERSION,
        "annotations_path": str(Path(annotations_path).resolve()),
        "metadata_path": str(Path(metadata_path).resolve()),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "parameters": {
            "limit": limit,
            "clip_duration_sec": clip_duration_sec,
            "min_track_boxes": min_track_boxes,
            "min_median_box_area_ratio": min_median_box_area_ratio,
            "max_source_duration_sec": max_source_duration_sec,
            "max_per_participant": max_per_participant,
            "max_per_scenario": max_per_scenario,
        },
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return selected


def write_scene_inputs(scene: dict[str, Any], staging_dir: str | Path) -> dict[str, Path]:
    """Write official EgoTracks metadata and a SAM-compatible first-frame box prompt."""

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = staging_dir / "ego4d_annotation.json"
    metadata_path.write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")

    label = scene["official_label"]
    first_box = scene["official_track"]["first_box_normalized_xywh"]
    prompt_payload = {
        "source": "EgoTracks official long-term track",
        "prompts": {
            label: {
                "frame_index": 0,
                "bounding_boxes": [first_box],
                "bounding_box_labels": [1],
            }
        },
    }
    prompt_path = staging_dir / "sam31_prompt_boxes.json"
    prompt_path.write_text(json.dumps(prompt_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"metadata": metadata_path, "prompt_boxes": prompt_path}


def finalize_ego4d_scene(
    scene_plan_path: str | Path,
    output_dir: str | Path,
    *,
    multiview_count: int = 3,
) -> dict[str, Any]:
    """Attach dataset provenance and choose camera-aware representative views."""

    scene = json.loads(Path(scene_plan_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    geometry_dir = output_dir / "g3t_long"
    frame_dir = output_dir / "frames"
    selection = select_multiview_frames(
        frame_dir=frame_dir,
        camera_pose_path=geometry_dir / "camera_pose.npy",
        output_dir=output_dir / "multiview_images",
        count=multiview_count,
    )
    metadata = {
        "version": EGO4D_PLAN_VERSION,
        "dataset": "Ego4D",
        "annotation_source": "EgoTracks " + str(scene.get("annotation_split", "val")),
        "scene_id": scene["scene_id"],
        "video_uid": scene["video_uid"],
        "clip_uid": scene["clip_uid"],
        "clip_start_sec": scene["clip_start_sec"],
        "clip_duration_sec": scene["clip_duration_sec"],
        "official_label": scene["official_label"],
        "official_track": scene["official_track"],
        "coordinate_system": "g3t_long_gravity_first_view_forward",
        "coordinate_origin": "first sampled clip frame camera center",
        "multiview_selection": selection,
    }
    metadata_path = output_dir / "ego4d_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def select_multiview_frames(
    frame_dir: str | Path,
    camera_pose_path: str | Path,
    output_dir: str | Path,
    *,
    count: int = 3,
    max_position_distance: float = 1.5,
) -> list[dict[str, Any]]:
    """Select the first view plus nearby views with different camera headings."""

    import numpy as np

    frame_paths = sorted(
        path for path in Path(frame_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    poses = np.load(camera_pose_path)
    usable = min(len(frame_paths), int(poses.shape[0]))
    if usable == 0:
        raise ValueError("No frame/camera-pose pairs are available for multiview selection.")
    count = min(max(1, count), usable)
    centers = []
    forwards = []
    for pose in poses[:usable]:
        rotation = np.asarray(pose[:3, :3], dtype=np.float64)
        translation = np.asarray(pose[:3, 3], dtype=np.float64)
        centers.append(-rotation.T @ translation)
        forward = rotation.T @ np.array([0.0, 0.0, 1.0])
        forwards.append(forward / max(float(np.linalg.norm(forward)), 1e-12))
    centers = np.asarray(centers)
    forwards = np.asarray(forwards)

    selected = [0]
    while len(selected) < count:
        best_index = None
        best_score = -math.inf
        for index in range(1, usable):
            if index in selected:
                continue
            distance = float(np.linalg.norm(centers[index] - centers[0]))
            if distance > max_position_distance:
                continue
            angular_diversity = min(
                math.acos(float(np.clip(np.dot(forwards[index], forwards[other]), -1.0, 1.0)))
                for other in selected
            )
            temporal_diversity = min(abs(index - other) for other in selected) / max(usable - 1, 1)
            score = angular_diversity + 0.15 * temporal_diversity - 0.05 * distance
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            remaining = [index for index in range(usable) if index not in selected]
            if not remaining:
                break
            best_index = max(remaining, key=lambda index: min(abs(index - other) for other in selected))
        selected.append(best_index)

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for order, index in enumerate(selected):
        source = frame_paths[index]
        target = output_dir / f"view_{order:02d}_frame_{index:06d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        records.append(
            {
                "order": order,
                "frame_array_index": index,
                "source_frame": source.name,
                "output_file": target.name,
                "camera_center": centers[index].tolist(),
                "camera_forward": forwards[index].tolist(),
                "distance_from_first": float(np.linalg.norm(centers[index] - centers[0])),
                "angle_from_first_deg": math.degrees(
                    math.acos(float(np.clip(np.dot(forwards[index], forwards[0]), -1.0, 1.0)))
                ),
            }
        )
    (output_dir / "selection.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return records


def _best_video_candidate(
    video: dict[str, Any],
    metadata: dict[str, Any],
    *,
    clip_duration_sec: float,
    min_track_boxes: int,
    min_median_box_area_ratio: float,
    max_source_duration_sec: float,
) -> dict[str, Any] | None:
    source_duration = float(metadata.get("duration_sec") or 0.0)
    if source_duration <= 0 or source_duration > max_source_duration_sec:
        return None
    best = None
    for clip in video.get("clips") or []:
        if clip.get("annotation_complete") is False:
            continue
        clip_fps = float(clip.get("clip_fps") or metadata.get("video_metadata", {}).get("fps") or 30.0)
        for annotation in clip.get("annotations") or []:
            for query_set_id, query in _iter_query_sets(annotation):
                if not query.get("is_valid", True) or query.get("errors"):
                    continue
                track = query.get("lt_track") or query.get("response_track") or []
                run = _longest_contiguous_track(track, fps=clip_fps)
                if len(run) < min_track_boxes:
                    continue
                area_ratios = [_box_area_ratio(box) for box in run]
                median_area = _median(area_ratios)
                if median_area < min_median_box_area_ratio:
                    continue
                label = _normalize_label(query.get("object_title"))
                if not label:
                    continue
                first = run[0]
                width = float(first.get("original_width") or metadata.get("video_metadata", {}).get("display_resolution_width") or 0)
                height = float(first.get("original_height") or metadata.get("video_metadata", {}).get("display_resolution_height") or 0)
                if width <= 0 or height <= 0:
                    continue
                anchor_frame = int(first.get("video_frame_number", first.get("frame_number", 0)))
                source_fps = float(metadata.get("video_metadata", {}).get("fps") or clip_fps)
                clip_start = max(0.0, anchor_frame / source_fps)
                duration = min(clip_duration_sec, max(0.0, source_duration - clip_start))
                if duration < min(5.0, clip_duration_sec):
                    continue
                normalized_box = _normalized_xywh(first)
                if normalized_box is None:
                    continue
                quality = (
                    math.log1p(min(len(run), int(round(clip_duration_sec * clip_fps))))
                    + 4.0 * math.sqrt(median_area)
                    + min(duration / clip_duration_sec, 1.0)
                    - 0.15 * math.log1p(source_duration / 60.0)
                )
                clip_uid = str(clip.get("clip_uid", ""))
                scene_id = _scene_id(str(video["video_uid"]), clip_uid, str(query_set_id))
                scenario = _first_text(metadata.get("scenarios")) or "unknown"
                participant = metadata.get("fb_participant_id")
                participant_key = str(participant) if participant is not None else f"{metadata.get('video_source')}:{metadata.get('origin_video_id')}"
                candidate = {
                    "version": EGO4D_PLAN_VERSION,
                    "scene_id": scene_id,
                    "video_uid": str(video["video_uid"]),
                    "clip_uid": clip_uid,
                    "query_set_id": str(query_set_id),
                    "official_label": label,
                    "clip_start_sec": round(clip_start, 6),
                    "clip_duration_sec": round(duration, 6),
                    "source_fps": source_fps,
                    "source_duration_sec": source_duration,
                    "source_s3_path": metadata.get("s3_path"),
                    "source": {
                        "video_source": metadata.get("video_source"),
                        "origin_video_id": metadata.get("origin_video_id"),
                        "device": metadata.get("device"),
                        "participant_id": participant,
                        "physical_setting_name": metadata.get("physical_setting_name"),
                        "scenarios": metadata.get("scenarios") or [],
                    },
                    "official_track": {
                        "num_boxes_in_contiguous_run": len(run),
                        "median_box_area_ratio": median_area,
                        "anchor_video_frame": anchor_frame,
                        "first_box_normalized_xywh": normalized_box,
                        "boxes": [_compact_box(box) for box in run],
                    },
                    "diversity": {
                        "participant_key": participant_key,
                        "scenario_key": scenario.casefold(),
                    },
                    "quality_score": quality,
                }
                if best is None or quality > best["quality_score"]:
                    best = candidate
    return best


def _iter_query_sets(annotation: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    query_sets = annotation.get("query_sets")
    if isinstance(query_sets, dict):
        yield from ((str(key), value) for key, value in query_sets.items() if isinstance(value, dict))
    elif isinstance(query_sets, list):
        for index, value in enumerate(query_sets):
            if isinstance(value, dict):
                yield str(value.get("query_set_id", index)), value


def _longest_contiguous_track(track: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    ordered = sorted(
        (box for box in track if isinstance(box, dict)),
        key=lambda box: int(box.get("frame_number", box.get("video_frame_number", 0))),
    )
    if not ordered:
        return []
    # frame_number is the 5 Hz annotation index; source frames advance faster.
    max_gap = max(2, int(round(fps * 0.4)))
    runs: list[list[dict[str, Any]]] = [[ordered[0]]]
    for box in ordered[1:]:
        current_frame = int(box.get("frame_number", box.get("video_frame_number", 0)))
        previous_frame = int(runs[-1][-1].get("frame_number", runs[-1][-1].get("video_frame_number", 0)))
        if 0 < current_frame - previous_frame <= max_gap:
            runs[-1].append(box)
        elif current_frame != previous_frame:
            runs.append([box])
    return max(runs, key=len)


def _box_area_ratio(box: dict[str, Any]) -> float:
    normalized = _normalized_xywh(box)
    return 0.0 if normalized is None else normalized[2] * normalized[3]


def _normalized_xywh(box: dict[str, Any]) -> list[float] | None:
    width = float(box.get("original_width") or 0.0)
    height = float(box.get("original_height") or 0.0)
    if width <= 0 or height <= 0:
        return None
    x1 = min(max(float(box.get("x") or 0.0) / width, 0.0), 1.0)
    y1 = min(max(float(box.get("y") or 0.0) / height, 0.0), 1.0)
    x2 = min(max((float(box.get("x") or 0.0) + float(box.get("width") or 0.0)) / width, 0.0), 1.0)
    y2 = min(max((float(box.get("y") or 0.0) + float(box.get("height") or 0.0)) / height, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(value, 12) for value in (x1, y1, x2 - x1, y2 - y1)]


def _compact_box(box: dict[str, Any]) -> dict[str, Any]:
    return {
        key: box[key]
        for key in (
            "video_frame_number", "frame_number", "exported_clip_frame_number",
            "x", "y", "width", "height", "original_width", "original_height",
        )
        if key in box
    }


def _normalize_label(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return " ".join(str(value or "").strip().casefold().replace("_", " ").split())


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _scene_id(video_uid: str, clip_uid: str, query_set_id: str) -> str:
    digest = hashlib.sha1(f"{video_uid}:{clip_uid}:{query_set_id}".encode()).hexdigest()[:10]
    return f"ego4d_{video_uid[:8]}_{digest}"


def _iter_annotation_videos(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson
    except ImportError:
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from payload.get("videos", [])
        return
    with path.open("rb") as stream:
        yield from ijson.items(stream, "videos.item")


def _load_video_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload.get("videos") or []
    if isinstance(videos, dict):
        videos = videos.values()
    return {str(video["video_uid"]): video for video in videos if isinstance(video, dict) and video.get("video_uid")}


def _write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
    temporary.replace(path)


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--annotations", type=Path, required=True)
    plan.add_argument("--metadata", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--limit", type=int, default=24)
    plan.add_argument("--clip-duration", type=float, default=20.0)
    plan.add_argument("--min-track-boxes", type=int, default=12)
    plan.add_argument("--min-box-area", type=float, default=0.003)
    plan.add_argument("--max-source-duration", type=float, default=1800.0)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--scene-plan", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--multiview-count", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "plan":
        selected = build_egotracks_subset_plan(
            args.annotations,
            args.metadata,
            args.output,
            limit=args.limit,
            clip_duration_sec=args.clip_duration,
            min_track_boxes=args.min_track_boxes,
            min_median_box_area_ratio=args.min_box_area,
            max_source_duration_sec=args.max_source_duration,
        )
        print(json.dumps({"selected": len(selected), "plan": str(args.output)}, indent=2))
    else:
        result = finalize_ego4d_scene(
            args.scene_plan,
            args.output_dir,
            multiview_count=args.multiview_count,
        )
        print(json.dumps({
            "scene_id": result["scene_id"],
            "output_dir": str(args.output_dir),
            "multiview_count": len(result["multiview_selection"]),
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
