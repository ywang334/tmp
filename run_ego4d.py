#!/usr/bin/env python3
"""Download and process EgoTracks scenes through the video pipeline."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from src.ego4d_dataset import EGO4D_PLAN_VERSION, build_egotracks_subset_plan, write_scene_inputs


ANNOTATION_BUCKET = "ego4d-consortium-sharing"
ANNOTATION_KEYS = {
    "train": "public/v2/egotracks/egotracks_train.json",
    "val": "public/v2/egotracks/egotracks_val.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego4d-root", type=Path, default=Path("/data/wy/data/ego4d"))
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--project-python", type=Path, default=Path("/data/wy/miniconda3/envs/video_object_graph/bin/python"))
    parser.add_argument("--subset-size", type=int, default=24)
    parser.add_argument("--all-scenes", action="store_true", help="Process every usable train/val EgoTracks video.")
    parser.add_argument("--clip-duration", type=float, default=20.0)
    parser.add_argument("--min-track-boxes", type=int, default=12)
    parser.add_argument("--min-box-area", type=float, default=0.003)
    parser.add_argument("--max-source-duration", type=float, default=1800.0)
    parser.add_argument("--cuda-visible-devices", default="0,2,3,4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Concurrent source download and clip preparation workers.",
    )
    parser.add_argument(
        "--prefetch-clips",
        type=int,
        default=8,
        help="Maximum prepared clips waiting for an available GPU worker.",
    )
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--multiview-count", type=int, default=3)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 processes the whole selected subset.")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--keep-clips", action="store_true")
    parser.add_argument("--keep-source-videos", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    root = args.ego4d_root.resolve()
    processed_root = (args.processed_root or root / "pipeline_processed").resolve()
    state_dir = processed_root / "_state"
    staging_root = root / ".pipeline_staging"
    metadata_path = root / "ego4d.json"
    processed_log = state_dir / "processed_scenes.jsonl"
    failed_log = state_dir / "failed_attempts.jsonl"
    for path in (processed_root, state_dir, staging_root):
        path.mkdir(parents=True, exist_ok=True)

    lock_stream = (state_dir / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another run_ego4d.py process already holds the pipeline lock.", file=sys.stderr)
        return 3

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Ego4D metadata is missing: {metadata_path}")

    if args.all_scenes:
        annotation_paths = {
            split: root / "v2/egotracks" / f"egotracks_{split}.json"
            for split in ("train", "val")
        }
        for split, annotation_path in annotation_paths.items():
            _ensure_annotation(annotation_path, ANNOTATION_KEYS[split], minimum_size=100_000_000)
        plan_path = state_dir / "egotracks_all_train_val_plan.jsonl"
        plan_parameters = {
            "mode": "all_train_val_videos",
            "splits": ["train", "val"],
            "clip_duration_sec": args.clip_duration,
            "one_scene_per_video": True,
        }
        if not _plan_is_current(plan_path, plan_parameters):
            print("[plan] selecting every usable EgoTracks train/val video")
            _build_all_scenes_plan(
                annotation_paths,
                metadata_path,
                plan_path,
                clip_duration_sec=args.clip_duration,
                parameters=plan_parameters,
            )
    else:
        annotation_path = root / "v2/egotracks/egotracks_val.json"
        _ensure_annotation(annotation_path, ANNOTATION_KEYS["val"], minimum_size=400_000_000)
        plan_path = state_dir / "egotracks_subset_plan.jsonl"
        plan_parameters = {
            "limit": args.subset_size,
            "clip_duration_sec": args.clip_duration,
            "min_track_boxes": args.min_track_boxes,
            "min_median_box_area_ratio": args.min_box_area,
            "max_source_duration_sec": args.max_source_duration,
            "max_per_participant": 1,
            "max_per_scenario": 2,
        }
        if not _plan_is_current(plan_path, plan_parameters):
            print(f"[plan] selecting {args.subset_size} diverse EgoTracks scenes")
            build_egotracks_subset_plan(
                annotation_path,
                metadata_path,
                plan_path,
                limit=args.subset_size,
                clip_duration_sec=args.clip_duration,
                min_track_boxes=args.min_track_boxes,
                min_median_box_area_ratio=args.min_box_area,
                max_source_duration_sec=args.max_source_duration,
            )
    plan = _read_jsonl(plan_path)
    print(f"[plan] {len(plan)} scenes: {plan_path}")
    if args.plan_only:
        return 0

    completed = {
        item.get("scene_id") for item in _read_jsonl(processed_log)
        if item.get("status") == "processed"
    }
    pending = [scene for scene in plan if scene["scene_id"] not in completed]
    if args.max_scenes > 0:
        pending = pending[: args.max_scenes]
    if not pending:
        print("[done] no pending Ego4D scenes")
        return 0

    gpu_devices = [device.strip() for device in args.cuda_visible_devices.split(",") if device.strip()]
    if not gpu_devices:
        raise ValueError("--cuda-visible-devices must contain at least one GPU index")
    worker_count = min(max(1, args.workers), len(gpu_devices), len(pending))
    download_worker_count = min(max(1, args.download_workers), len(pending))
    prefetch_clips = max(1, args.prefetch_clips)
    print(
        f"[workers] {worker_count} parallel pipelines on GPUs "
        + ",".join(gpu_devices[:worker_count])
    )
    print(
        f"[downloads] {download_worker_count} parallel clip workers; "
        f"ready queue capacity={prefetch_clips}"
    )
    _run_shared_queue_pipeline(
        pending,
        gpu_devices=gpu_devices[:worker_count],
        download_worker_count=download_worker_count,
        prefetch_clips=prefetch_clips,
        args=args,
        root=root,
        staging_root=staging_root,
        project_dir=project_dir,
        processed_root=processed_root,
        processed_log=processed_log,
        failed_log=failed_log,
    )
    return 0


PreparedScene = tuple[dict[str, object], Path]


def _run_shared_queue_pipeline(
    scenes: list[dict[str, object]],
    *,
    gpu_devices: list[str],
    download_worker_count: int,
    prefetch_clips: int,
    args: argparse.Namespace,
    root: Path,
    staging_root: Path,
    project_dir: Path,
    processed_root: Path,
    processed_log: Path,
    failed_log: Path,
) -> None:
    ready_queue: queue.Queue[PreparedScene | None] = queue.Queue(maxsize=prefetch_clips)
    stop_event = threading.Event()
    fatal_errors: list[BaseException] = []
    fatal_error_lock = threading.Lock()

    def record_fatal_error(exc: BaseException) -> None:
        with fatal_error_lock:
            if not fatal_errors:
                fatal_errors.append(exc)
        stop_event.set()

    with ThreadPoolExecutor(
        max_workers=len(gpu_devices),
        thread_name_prefix="ego4d-gpu",
    ) as gpu_pool:
        gpu_futures = [
            gpu_pool.submit(
                _run_gpu_worker,
                ready_queue,
                gpu_device=gpu_device,
                stop_event=stop_event,
                record_fatal_error=record_fatal_error,
                args=args,
                root=root,
                project_dir=project_dir,
                processed_root=processed_root,
                processed_log=processed_log,
                failed_log=failed_log,
            )
            for gpu_device in gpu_devices
        ]

        scene_iterator = iter(scenes)
        with ThreadPoolExecutor(
            max_workers=download_worker_count,
            thread_name_prefix="ego4d-download",
        ) as download_pool:
            downloads: dict[Future[Path], dict[str, object]] = {}

            def submit_download() -> bool:
                try:
                    scene = next(scene_iterator)
                except StopIteration:
                    return False
                future = download_pool.submit(_prepare_clip, scene, root, staging_root)
                downloads[future] = scene
                return True

            for _ in range(download_worker_count):
                if not submit_download():
                    break

            while downloads and not stop_event.is_set():
                completed, _ = wait(downloads, return_when=FIRST_COMPLETED)
                for future in completed:
                    scene = downloads.pop(future)
                    try:
                        clip_path = future.result()
                    except Exception as exc:
                        _append_jsonl(failed_log, _failure_record(scene, "clip preparation", exc))
                        print(f"[failed download] {scene['scene_id']}: {exc}", file=sys.stderr)
                        if args.stop_on_error:
                            record_fatal_error(exc)
                            break
                    else:
                        if not _put_ready_scene(
                            ready_queue,
                            (scene, clip_path),
                            stop_event=stop_event,
                        ):
                            break
                        print(
                            f"[queued clip] {scene['scene_id']} "
                            f"ready={ready_queue.qsize()}/{ready_queue.maxsize}"
                        )
                    if not stop_event.is_set():
                        submit_download()

            if stop_event.is_set():
                for future in downloads:
                    future.cancel()

        if not stop_event.is_set():
            for _ in gpu_devices:
                ready_queue.put(None)
        for future in gpu_futures:
            future.result()

    if fatal_errors:
        raise fatal_errors[0]


def _put_ready_scene(
    ready_queue: queue.Queue[PreparedScene | None],
    prepared: PreparedScene,
    *,
    stop_event: threading.Event,
) -> bool:
    while not stop_event.is_set():
        try:
            ready_queue.put(prepared, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _run_gpu_worker(
    ready_queue: queue.Queue[PreparedScene | None],
    *,
    gpu_device: str,
    stop_event: threading.Event,
    record_fatal_error: Callable[[BaseException], None],
    args: argparse.Namespace,
    root: Path,
    project_dir: Path,
    processed_root: Path,
    processed_log: Path,
    failed_log: Path,
) -> None:
    while not stop_event.is_set():
        try:
            prepared = ready_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if prepared is None:
            ready_queue.task_done()
            return
        scene, clip_path = prepared
        try:
            print(
                f"[worker gpu={gpu_device}] {scene['scene_id']} "
                f"ready={ready_queue.qsize()}/{ready_queue.maxsize}"
            )
            try:
                _process_scene(
                    scene,
                    clip_path=clip_path,
                    project_dir=project_dir,
                    project_python=args.project_python,
                    output_dir=processed_root / str(scene["scene_id"]),
                    cuda_visible_devices=gpu_device,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                    multiview_count=args.multiview_count,
                    keep_intermediates=args.keep_intermediates,
                )
                source_path = root / "v2/full_scale" / f"{scene['video_uid']}.mp4"
                source_deleted = False
                if not args.keep_source_videos and source_path.is_file():
                    source_path.unlink()
                    source_deleted = True
                    print(f"[deleted source gpu={gpu_device}] {source_path}")
                if not args.keep_clips:
                    clip_path.unlink(missing_ok=True)
                _append_jsonl(
                    processed_log,
                    {
                        "scene_id": scene["scene_id"],
                        "video_uid": scene["video_uid"],
                        "status": "processed",
                        "processed_at": _now(),
                        "output_dir": str(processed_root / str(scene["scene_id"])),
                        "source_deleted": source_deleted,
                        "gpu_device": gpu_device,
                    },
                )
                print(f"[processed gpu={gpu_device}] {scene['scene_id']}")
            except Exception as exc:
                _append_jsonl(failed_log, _failure_record(scene, "pipeline", exc))
                print(f"[failed pipeline gpu={gpu_device}] {scene['scene_id']}: {exc}", file=sys.stderr)
                if args.stop_on_error:
                    record_fatal_error(exc)
                    return
        finally:
            ready_queue.task_done()


def _build_all_scenes_plan(
    annotation_paths: dict[str, Path],
    metadata_path: Path,
    output_path: Path,
    *,
    clip_duration_sec: float,
    parameters: dict[str, object],
) -> list[dict[str, object]]:
    scenes_by_video: dict[str, dict[str, object]] = {}
    temporary_plans: list[Path] = []
    try:
        for split, annotation_path in annotation_paths.items():
            temporary_plan = output_path.with_name(f".{output_path.stem}_{split}.jsonl")
            temporary_plans.append(temporary_plan)
            split_scenes = build_egotracks_subset_plan(
                annotation_path,
                metadata_path,
                temporary_plan,
                limit=1_000_000,
                clip_duration_sec=clip_duration_sec,
                min_track_boxes=1,
                min_median_box_area_ratio=0.0,
                max_source_duration_sec=1_000_000_000.0,
                max_per_participant=1_000_000,
                max_per_scenario=1_000_000,
            )
            for scene in split_scenes:
                scene["annotation_split"] = split
                video_uid = str(scene["video_uid"])
                previous = scenes_by_video.get(video_uid)
                if previous is None or float(scene["quality_score"]) > float(previous["quality_score"]):
                    scenes_by_video[video_uid] = scene
    finally:
        for temporary_plan in temporary_plans:
            temporary_plan.unlink(missing_ok=True)
            temporary_plan.with_suffix(".summary.json").unlink(missing_ok=True)

    selected = sorted(
        scenes_by_video.values(),
        key=lambda item: (-float(item["quality_score"]), str(item["video_uid"])),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for scene in selected:
            stream.write(json.dumps(scene, ensure_ascii=False, default=float) + "\n")
    temporary.replace(output_path)
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "version": EGO4D_PLAN_VERSION,
                "selected_count": len(selected),
                "parameters": parameters,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return selected


def _plan_is_current(plan_path: Path, expected_parameters: dict[str, object]) -> bool:
    summary_path = plan_path.with_suffix(".summary.json")
    if not plan_path.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("version") == EGO4D_PLAN_VERSION
        and summary.get("parameters") == expected_parameters
        and summary.get("selected_count", 0) > 0
    )


def _ensure_annotation(path: Path, key: str, *, minimum_size: int) -> None:
    if path.is_file() and path.stat().st_size > minimum_size:
        return
    path.unlink(missing_ok=True)
    print(f"[download annotations] s3://{ANNOTATION_BUCKET}/{key}")
    _download_s3_object(
        f"s3://{ANNOTATION_BUCKET}/{key}",
        path,
        retries=8,
    )
    if path.stat().st_size <= minimum_size:
        raise RuntimeError(f"downloaded annotation is incomplete: {path}")


def _prepare_clip(scene: dict[str, object], root: Path, staging_root: Path) -> Path:
    scene_dir = staging_root / str(scene["scene_id"])
    scene_dir.mkdir(parents=True, exist_ok=True)
    clip_path = scene_dir / "clip.mp4"
    if clip_path.is_file() and clip_path.stat().st_size > 1_000_000:
        return clip_path
    local_source = root / "v2/full_scale" / f"{scene['video_uid']}.mp4"
    downloaded_source = False
    if local_source.is_file():
        input_source = local_source
    else:
        input_source = scene_dir / "source.mp4"
        _download_s3_object(str(scene["source_s3_path"]), input_source)
        downloaded_source = True
    temporary = clip_path.with_suffix(".tmp.mp4")
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).with_name("ffmpeg"))
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError(f"ffmpeg was not found at {ffmpeg}")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(scene["clip_start_sec"]), "-i", str(input_source),
        "-t", str(scene["clip_duration_sec"]), "-an",
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    print(f"[prefetch clip] {scene['scene_id']} @ {scene['clip_start_sec']}s")
    subprocess.run(command, check=True)
    if temporary.stat().st_size <= 1_000_000:
        raise RuntimeError(f"ffmpeg produced an incomplete clip: {temporary}")
    temporary.replace(clip_path)
    if downloaded_source:
        input_source.unlink(missing_ok=True)
    return clip_path


def _download_s3_object(s3_path: str, target: Path, retries: int = 4) -> None:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    parsed = urlparse(s3_path)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid source_s3_path: {s3_path!r}")
    if target.is_file() and target.stat().st_size > 1_000_000:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".multipart")
    key = parsed.path.lstrip("/")
    client = boto3.Session(profile_name="default").client(
        "s3",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    transfer = TransferConfig(
        multipart_threshold=16 * 1024 * 1024,
        multipart_chunksize=16 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    for attempt in range(1, retries + 1):
        _remove_transfer_temporary_files(temporary)
        try:
            print(f"[download source {attempt}/{retries}] {parsed.netloc}/{key}")
            total_bytes = int(
                client.head_object(Bucket=parsed.netloc, Key=key)["ContentLength"]
            )
            progress = _DownloadProgress(
                label=f"{target.parent.name}/{target.name}",
                total_bytes=total_bytes,
            )
            client.download_file(
                parsed.netloc,
                key,
                str(temporary),
                Config=transfer,
                Callback=progress,
            )
            if temporary.stat().st_size <= 1_000_000:
                raise RuntimeError(f"downloaded source is incomplete: {temporary}")
            temporary.replace(target)
            return
        except Exception:
            _remove_transfer_temporary_files(temporary)
            if attempt >= retries:
                raise
            time.sleep(min(30, 2 ** attempt))


class _DownloadProgress:
    def __init__(self, *, label: str, total_bytes: int) -> None:
        self.label = label
        self.total_bytes = max(1, total_bytes)
        self.downloaded_bytes = 0
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        self.report_step_bytes = max(1, self.total_bytes // 20)
        self.next_report_bytes = self.report_step_bytes
        self.lock = threading.Lock()

    def __call__(self, bytes_amount: int) -> None:
        with self.lock:
            self.downloaded_bytes = min(
                self.total_bytes,
                self.downloaded_bytes + int(bytes_amount),
            )
            now = time.monotonic()
            if (
                self.downloaded_bytes < self.next_report_bytes
                and self.downloaded_bytes < self.total_bytes
                and now - self.last_report_at < 30.0
            ):
                return
            elapsed = max(now - self.started_at, 1e-6)
            speed = self.downloaded_bytes / elapsed
            remaining = max(0, self.total_bytes - self.downloaded_bytes)
            eta_seconds = remaining / speed if speed > 0 else float("inf")
            print(
                f"[download progress] {self.label} "
                f"{100.0 * self.downloaded_bytes / self.total_bytes:.1f}% "
                f"{_format_bytes(self.downloaded_bytes)}/{_format_bytes(self.total_bytes)} "
                f"{_format_bytes(speed)}/s ETA={_format_duration(eta_seconds)}",
                flush=True,
            )
            self.last_report_at = now
            while self.next_report_bytes <= self.downloaded_bytes:
                self.next_report_bytes += self.report_step_bytes


def _remove_transfer_temporary_files(temporary: Path) -> None:
    temporary.unlink(missing_ok=True)
    for path in temporary.parent.glob(temporary.name + ".*"):
        path.unlink(missing_ok=True)


def _format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units[:-1]:
        if abs(amount) < 1024.0:
            return f"{amount:.1f}{unit}"
        amount /= 1024.0
    return f"{amount:.1f}{units[-1]}"


def _format_duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _process_scene(
    scene: dict[str, object],
    *,
    clip_path: Path,
    project_dir: Path,
    project_python: Path,
    output_dir: Path,
    cuda_visible_devices: str,
    frame_stride: int,
    max_frames: int,
    multiview_count: int,
    keep_intermediates: bool,
) -> None:
    if _final_output_complete(output_dir, str(scene["scene_id"])):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = clip_path.parent
    inputs = write_scene_inputs(scene, staging_dir)
    command = [
        str(project_python), str(project_dir / "main.py"),
        "--video_path", str(clip_path),
        "--output_dir", str(output_dir),
        "--resume",
        "--frame_stride", str(frame_stride),
        "--max_frames", str(max_frames),
        "--geometry_backend", "g3t_long",
        "--g3t_long_chunk_size", "50",
        "--g3t_long_overlap", "5",
        "--g3t_long_loop_chunk_size", "3",
        "--g3t_long_loop_enable",
        "--text_prompt", str(scene["official_label"]),
        "--auto_text_prompt",
        "--tagger_model", str(project_dir / "checkpoints/RAM++/ram_plus_swin_large_14m.pth"),
        "--tagger_repo", str(project_dir / "third_party/recognize-anything"),
        "--tagger_max_tags", "8",
        "--tagger_min_count", "1",
        "--tagger_min_confidence", "0.8",
        "--tagger_num_frames", "4",
        "--sam31_prompt_boxes", str(inputs["prompt_boxes"]),
        "--sam31_tracking_mode", "sam3",
        "--sam31_detection_frame_stride", "1",
        "--sam31_native_frame_stride", "2",
        "--min_track_frames", "5",
        "--short_track_min_score", "0.75",
        "--short_track_min_points", "1000",
        "--estimate_orientation",
        "--orient_anything_model", str(project_dir / "checkpoints/Orient-Anything-V2"),
        "--orient_anything_repo", str(project_dir / "third_party/Orient-Anything-V2"),
        "--orientation_max_frames", "1",
        "--orientation_crop_padding", "0.25",
        "--orientation_front_axis", "+x",
        "--visualize", "--geometry_visualization", "--visualize_object_bboxes", "--visualize_object_orientations",
        "--cuda_visible_devices", cuda_visible_devices,
    ]
    environment = dict(os.environ)
    environment.setdefault("SAM31_GROUNDING_BATCH_SIZE", "1")
    environment.setdefault("SAM31_POSTPROCESS_BATCH_SIZE", "1")
    environment.setdefault("SAM31_RECOVERY_IOU_THRESHOLD", "0.2")
    environment.setdefault("SAM31_RECOVERY_MAX_FRAME_GAP", "8")
    environment.setdefault("SAM31_MAX_RECOVERY_ATTEMPTS", "8")
    environment.setdefault("SAM31_MAX_SAME_ANCHOR_ATTEMPTS", "1")
    print(f"[pipeline] {scene['scene_id']} -> {output_dir}")
    subprocess.run(command, cwd=project_dir, env=environment, check=True)
    subprocess.run(
        [
            str(project_python), "-m", "src.ego4d_dataset", "finalize",
            "--scene-plan", str(inputs["metadata"]),
            "--output-dir", str(output_dir),
            "--multiview-count", str(multiview_count),
        ],
        cwd=project_dir,
        check=True,
    )
    shutil.copy2(inputs["prompt_boxes"], output_dir / "sam31_prompt_boxes.json")
    if not keep_intermediates:
        _prune_output(output_dir)
    _write_manifest(output_dir, scene)
    if not _final_output_complete(output_dir, str(scene["scene_id"])):
        raise RuntimeError("final output validation failed")


def _prune_output(output_dir: Path) -> None:
    geometry_metadata = output_dir / "g3t_long/g3t_long_metadata.json"
    if geometry_metadata.is_file():
        shutil.copy2(geometry_metadata, output_dir / "g3t_long_metadata.json")
    for name in ("frames", "g3t_long", "tagger", "sam31", "orientation"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    for name in (
        "frame_meta.json", "object_center_debug.json", "orientation_debug.json",
        "precluster_detections.html", "precluster_detections.png", "precluster_detections.csv",
    ):
        (output_dir / name).unlink(missing_ok=True)


def _write_manifest(output_dir: Path, scene: dict[str, object]) -> None:
    payload = {
        "version": 1,
        "status": "complete",
        "scene_id": scene["scene_id"],
        "video_uid": scene["video_uid"],
        "completed_at": _now(),
        "geometry_backend": "g3t_long",
        "instance_backend": "EgoTracks seed + RAM++ + SAM3.1",
        "orientation_backend": "Orient-Anything-V2",
        "retained_files": sorted(path.name for path in output_dir.iterdir() if path.name != "pipeline_manifest.json"),
    }
    temporary = output_dir / "pipeline_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_dir / "pipeline_manifest.json")


def _final_output_complete(output_dir: Path, scene_id: str) -> bool:
    required = (
        "objects.json", "objects_oriented.json", "object_graph.html", "object_graph.csv",
        "g3t_long_scene.html", "g3t_long_explorer.html", "ego4d_metadata.json",
        "sam31_prompt_boxes.json", "multiview_images/selection.json", "pipeline_manifest.json",
    )
    if not all((output_dir / relative).is_file() and (output_dir / relative).stat().st_size > 0 for relative in required):
        return False
    try:
        manifest = json.loads((output_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
        objects = json.loads((output_dir / "objects_oriented.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("status") == "complete" and manifest.get("scene_id") == scene_id and isinstance(objects.get("objects"), list)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _failure_record(scene: dict[str, object], stage: str, exc: Exception) -> dict[str, object]:
    return {
        "scene_id": scene["scene_id"],
        "video_uid": scene["video_uid"],
        "status": "failed",
        "stage": stage,
        "failed_at": _now(),
        "reason": f"{type(exc).__name__}: {exc}",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
