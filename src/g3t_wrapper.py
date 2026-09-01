"""Wrapper for running G3T feed-forward geometry prediction.

This module keeps the public output contract compatible with ``src.vggt_wrapper``:
``camera_pose.npy``, ``depth.npy``, ``point_map.npy``, and ``point_conf.npy``.
It imports the official G3T code at runtime and does not modify the checkout.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

DEFAULT_G3T_MODEL = "checkpoints/g3t"
DEFAULT_G3T_REPO = "third_party/g3t"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class G3TConfig:
    """Runtime configuration for the G3T wrapper."""

    model_path: str = DEFAULT_G3T_MODEL
    repo_path: str = DEFAULT_G3T_REPO
    points_source: str = "point_head"
    img_load_resolution: int = 1024
    inference_resolution: int = 518
    seed: int = 42
    max_feed_forward_tokens: int = 12_000


@dataclass(frozen=True)
class G3TLongConfig:
    """Runtime configuration for the G3T-Long wrapper."""

    model_path: str = DEFAULT_G3T_MODEL
    repo_path: str = DEFAULT_G3T_REPO
    config_path: str = "third_party/g3t/vggt_long/configs/g3t_long.yaml"
    points_source: str = "point_head"
    chunk_size: int = 8
    overlap: int = 2
    loop_chunk_size: int = 3
    loop_enable: bool = False
    conf_thresh_multiplier_for_alignment: float = 0.1
    conf_thresh_multiplier_for_viz: float = 0.75
    keep_cache: bool = True


@dataclass(frozen=True)
class G3TDependencies:
    """Runtime dependencies imported from the official G3T stack."""

    np: Any
    torch: Any
    functional: Any
    g3t_cls: Any
    load_and_preprocess_images_square: Any
    pose_encoding_to_extri_intri: Any
    make_4x4: Any
    unproject_depth_map_to_point_map: Any


def run_g3t(
    frame_dir: str | Path,
    output_dir: str | Path,
    model_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    points_source: str | None = None,
    img_load_resolution: int | None = None,
    inference_resolution: int | None = None,
    seed: int | None = None,
) -> dict[str, Path]:
    """Run official G3T feed-forward inference and save VGGT-compatible outputs.

    Args:
        frame_dir: Directory containing ordered image frames.
        output_dir: Directory where geometry arrays and ``g3t.log`` are written.
        model_path: Local G3T checkpoint directory or checkpoint file. Defaults
            to ``G3T_MODEL``, then ``checkpoints/g3t``.
        repo_path: Local official G3T checkout. Defaults to ``G3T_REPO``, then
            ``third_party/g3t``.
        points_source: ``point_head`` for direct G3T point maps or ``depth_head``
            to unproject the depth head with predicted camera parameters.
        img_load_resolution: Square image load resolution before interpolation.
        inference_resolution: Final square inference resolution. Official G3T
            weights use 518.
        seed: Random seed for deterministic torch/numpy setup.

    Returns:
        dict[str, Path]: Paths keyed like ``run_vggt`` plus G3T-specific extras.
    """

    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir / "g3t.log")

    config = _load_config_from_environment(
        model_path=model_path,
        repo_path=repo_path,
        points_source=points_source,
        img_load_resolution=img_load_resolution,
        inference_resolution=inference_resolution,
        seed=seed,
    )
    _validate_config(config)
    frame_paths = _collect_frame_paths(frame_dir)
    _validate_feed_forward_size(config=config, num_frames=len(frame_paths))

    LOGGER.info("Starting G3T wrapper")
    LOGGER.info("Frame directory: %s", frame_dir)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Model: %s", config.model_path)
    LOGGER.info("Official repo: %s", config.repo_path)
    LOGGER.info("Points source: %s", config.points_source)
    LOGGER.info("Discovered %d input frames", len(frame_paths))
    LOGGER.info("Estimated feed-forward tokens: %d", _estimate_feed_forward_tokens(config, len(frame_paths)))

    deps = _load_g3t_dependencies(config.repo_path)
    _set_seeds(deps=deps, seed=config.seed)
    device = "cuda" if deps.torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    try:
        predictions = _run_inference(
            deps=deps,
            config=config,
            frame_paths=frame_paths,
            device=device,
        )
        saved_paths = _save_predictions(
            predictions=predictions,
            output_dir=output_dir,
            config=config,
            deps=deps,
        )
        del predictions
        _release_cuda_memory(deps=deps, device=device)
    except Exception:
        LOGGER.exception("G3T wrapper failed")
        raise

    LOGGER.info("Saved G3T outputs to %s", output_dir)
    return saved_paths



def run_g3t_long(
    frame_dir: str | Path,
    output_dir: str | Path,
    model_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    config_path: str | Path | None = None,
    points_source: str | None = None,
    chunk_size: int = 8,
    overlap: int = 2,
    loop_chunk_size: int = 3,
    loop_enable: bool = False,
    conf_thresh_multiplier_for_alignment: float = 0.1,
    conf_thresh_multiplier_for_viz: float = 0.75,
    keep_cache: bool = True,
) -> dict[str, Path]:
    """Run official G3T-Long and save VGGT-compatible dense geometry outputs.

    G3T-Long processes long videos in overlapping feed-forward chunks, aligns the
    chunks, then this wrapper stitches the aligned chunk point maps back into one
    dense ``point_map.npy`` / ``point_conf.npy`` sequence for Stage 3/4.
    """

    frame_dir = Path(frame_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir / "g3t_long.log")

    selected_model_path = Path(model_path or os.environ.get("G3T_MODEL", DEFAULT_G3T_MODEL))
    selected_repo_path = Path(repo_path or os.environ.get("G3T_REPO", DEFAULT_G3T_REPO)).resolve()
    selected_config_path = Path(
        config_path
        or os.environ.get(
            "G3T_LONG_CONFIG",
            Path(DEFAULT_G3T_REPO) / "vggt_long/configs/g3t_long.yaml",
        )
    ).resolve()
    if selected_model_path.exists():
        selected_model_path = selected_model_path.resolve()

    config = G3TLongConfig(
        model_path=str(selected_model_path),
        repo_path=str(selected_repo_path),
        config_path=str(selected_config_path),
        points_source=str(points_source or os.environ.get("G3T_POINTS_SOURCE", "point_head")),
        chunk_size=int(chunk_size),
        overlap=int(overlap),
        loop_chunk_size=int(loop_chunk_size),
        loop_enable=bool(loop_enable),
        conf_thresh_multiplier_for_alignment=float(conf_thresh_multiplier_for_alignment),
        conf_thresh_multiplier_for_viz=float(conf_thresh_multiplier_for_viz),
        keep_cache=bool(keep_cache),
    )
    _validate_g3t_long_config(config)
    frame_paths = _collect_frame_paths(frame_dir)

    LOGGER.info("Starting G3T-Long wrapper")
    LOGGER.info("Frame directory: %s", frame_dir)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Model: %s", config.model_path)
    LOGGER.info("Official repo: %s", config.repo_path)
    LOGGER.info("Chunk size / overlap: %d / %d", config.chunk_size, config.overlap)
    LOGGER.info("Discovered %d input frames", len(frame_paths))

    deps = _load_g3t_long_dependencies(config.repo_path)
    official_config = _load_official_g3t_long_config(
        deps=deps,
        repo_path=Path(config.repo_path),
        config_path=Path(config.config_path),
    )
    _configure_official_g3t_long_config(
        official_config=official_config,
        config=config,
        repo_path=Path(config.repo_path),
    )
    _validate_g3t_long_loop_runtime(official_config=official_config)
    _patch_g3t_long_adapter_load(deps=deps)

    previous_cwd = Path.cwd()
    g3t_long_obj = None
    try:
        # Official loop modules resolve the bundled DINO repository relative to cwd.
        os.chdir(config.repo_path)
        g3t_long_obj = deps.g3t_long_cls(
            str(frame_dir),
            str(output_dir),
            official_config,
            conf_thresh_multiplier=config.conf_thresh_multiplier_for_alignment,
        )
        intrinsic, camera_pose, w2g, c2g = g3t_long_obj.run()
        point_map, point_conf = _collect_g3t_long_dense_outputs(
            g3t_long_obj=g3t_long_obj,
            num_frames=len(frame_paths),
            np=deps.np,
        )
        saved_paths = _save_g3t_long_outputs(
            output_dir=output_dir,
            config=config,
            point_map=point_map,
            point_conf=point_conf,
            intrinsic=intrinsic,
            camera_pose=camera_pose,
            w2g=w2g,
            c2g=c2g,
            g3t_long_obj=g3t_long_obj,
            deps=deps,
        )
    finally:
        if g3t_long_obj is not None and not config.keep_cache:
            g3t_long_obj.close()
        os.chdir(previous_cwd)
        _release_cuda_memory(deps=deps, device="cuda" if deps.torch.cuda.is_available() else "cpu")

    LOGGER.info("Saved G3T-Long outputs to %s", output_dir)
    return saved_paths

def _run_inference(
    deps: G3TDependencies,
    config: G3TConfig,
    frame_paths: list[Path],
    device: str,
) -> dict[str, Any]:
    """Run official G3T feed-forward inference on ordered frames."""

    model = _load_model(deps=deps, config=config).to(device)
    if hasattr(model, "eval"):
        model.eval()

    images, original_coords = deps.load_and_preprocess_images_square(
        [str(path) for path in frame_paths],
        config.img_load_resolution,
    )
    images = images.to(device)
    if len(images.shape) != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected images with shape [N,3,H,W], got {tuple(images.shape)}")

    images = deps.functional.interpolate(
        images,
        size=(config.inference_resolution, config.inference_resolution),
        mode="bilinear",
        align_corners=False,
    )

    LOGGER.info("Running G3T feed-forward inference")
    with deps.torch.no_grad():
        if device == "cuda":
            with deps.torch.amp.autocast("cuda", enabled=True, dtype=deps.torch.bfloat16):
                raw = model(images=images[None])
        else:
            raw = model(images=images[None])

    if not isinstance(raw, dict):
        raise RuntimeError("G3T model output must be a dictionary.")
    return _decode_predictions(
        raw=raw,
        images=images,
        original_coords=original_coords,
        points_source=config.points_source,
        deps=deps,
    )


def _load_model(deps: G3TDependencies, config: G3TConfig) -> Any:
    """Load G3T from a local HF directory, safetensors file, or torch checkpoint."""

    model_path = Path(config.model_path)
    if model_path.is_dir():
        try:
            return deps.g3t_cls.from_pretrained(str(model_path))
        except Exception as exc:
            LOGGER.warning("G3T.from_pretrained(%s) failed, falling back to manual load: %s", model_path, exc)
            safetensors_path = model_path / "model.safetensors"
            torch_path = model_path / "pytorch_model.bin"
            if safetensors_path.exists():
                return _load_safetensors_checkpoint(deps=deps, checkpoint_path=safetensors_path)
            if torch_path.exists():
                return _load_torch_checkpoint(deps=deps, checkpoint_path=torch_path)
            raise

    if model_path.suffix == ".safetensors":
        return _load_safetensors_checkpoint(deps=deps, checkpoint_path=model_path)
    return _load_torch_checkpoint(deps=deps, checkpoint_path=model_path)


def _load_safetensors_checkpoint(deps: G3TDependencies, checkpoint_path: Path) -> Any:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Loading G3T safetensors checkpoints requires safetensors.") from exc

    model = deps.g3t_cls(
        enable_point=True,
        enable_depth=True,
        enable_gravity_camera_heads=True,
    )
    state = load_file(str(checkpoint_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    LOGGER.info("Loaded G3T safetensors checkpoint with %d missing and %d unexpected keys", len(missing), len(unexpected))
    return model


def _load_torch_checkpoint(deps: G3TDependencies, checkpoint_path: Path) -> Any:
    model = deps.g3t_cls(
        enable_point=True,
        enable_depth=True,
        enable_gravity_camera_heads=True,
    )
    with checkpoint_path.open("rb") as file:
        checkpoint = deps.torch.load(file, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    LOGGER.info("Loaded G3T torch checkpoint with %d missing and %d unexpected keys", len(missing), len(unexpected))
    return model


def _decode_predictions(
    raw: dict[str, Any],
    images: Any,
    original_coords: Any,
    points_source: str,
    deps: G3TDependencies,
) -> dict[str, Any]:
    """Decode official G3T heads to arrays compatible with the current pipeline."""

    local_pose_enc = _pick_prediction(raw, "local_pose_enc")
    global_pose_enc = _pick_prediction(raw, "global_pose_enc")

    g2c, intrinsic = deps.pose_encoding_to_extri_intri(
        local_pose_enc,
        images.shape[-2:],
        pose_encoding_type="noT_quaR_FoV",
    )
    w2g, _ = deps.pose_encoding_to_extri_intri(
        global_pose_enc,
        images.shape[-2:],
        pose_encoding_type="absT_quaRy_noFoV",
    )
    w2c = deps.torch.matmul(deps.make_4x4(g2c), deps.make_4x4(w2g))[..., :3, :]
    g2c = g2c.squeeze(0)
    w2g = w2g.squeeze(0)
    w2c = w2c.squeeze(0)
    intrinsic = intrinsic.squeeze(0)

    c2g = deps.torch.zeros_like(g2c)
    c2g[:, :3, :3] = g2c[:, :3, :3].transpose(1, 2)

    if points_source == "point_head":
        point_map = _pick_prediction(raw, "world_points").squeeze(0)
        point_conf = _pick_prediction(raw, "world_points_conf").squeeze(0)
    elif points_source == "depth_head":
        depth_for_points = _pick_prediction(raw, "depth").squeeze(0)
        point_conf = _pick_prediction(raw, "depth_conf").squeeze(0)
        point_map_np = deps.unproject_depth_map_to_point_map(depth_for_points, w2c, intrinsic)
        point_map = deps.torch.from_numpy(point_map_np).to(depth_for_points.device).to(deps.torch.float32)
    else:
        raise ValueError(f"Unsupported points_source: {points_source}")

    depth = _pick_prediction(raw, "depth").squeeze(0)
    depth_conf = _pick_prediction(raw, "depth_conf").squeeze(0)
    colors = images.detach().permute(0, 2, 3, 1).cpu().numpy()

    return {
        "camera_pose": w2c,
        "depth": depth,
        "depth_conf": depth_conf,
        "point_map": point_map,
        "point_conf": point_conf,
        "intrinsic": intrinsic,
        "w2g": w2g,
        "g2c": g2c,
        "c2g": c2g,
        "image_colors": colors,
        "original_coords": original_coords,
    }


def _save_predictions(
    predictions: dict[str, Any],
    output_dir: Path,
    config: G3TConfig,
    deps: G3TDependencies,
) -> dict[str, Path]:
    """Save G3T predictions using the current geometry output contract."""

    arrays = {key: _to_numpy_array(value) for key, value in predictions.items()}
    point_map = arrays["point_map"]
    point_conf = arrays["point_conf"]
    depth = arrays["depth"]
    point_map, camera_pose, arrays["w2g"], frame_alignment = _canonicalize_g3t_first_view_forward(
        point_map=point_map,
        camera_pose=arrays["camera_pose"],
        w2g=arrays["w2g"],
        np=deps.np,
    )
    _validate_point_outputs(point_map=point_map, point_conf=point_conf)

    paths = {
        "camera_pose": output_dir / "camera_pose.npy",
        "depth": output_dir / "depth.npy",
        "point_map": output_dir / "point_map.npy",
        "point_conf": output_dir / "point_conf.npy",
        "intrinsic": output_dir / "intrinsic.npy",
        "w2g": output_dir / "w2g.npy",
        "g2c": output_dir / "g2c.npy",
        "c2g": output_dir / "c2g.npy",
        "depth_conf": output_dir / "depth_conf.npy",
        "metadata": output_dir / "g3t_metadata.json",
    }
    deps.np.save(paths["camera_pose"], camera_pose)
    deps.np.save(paths["depth"], depth)
    deps.np.save(paths["point_map"], point_map)
    deps.np.save(paths["point_conf"], point_conf)
    deps.np.save(paths["intrinsic"], arrays["intrinsic"])
    deps.np.save(paths["w2g"], arrays["w2g"])
    deps.np.save(paths["g2c"], arrays["g2c"])
    deps.np.save(paths["c2g"], arrays["c2g"])
    deps.np.save(paths["depth_conf"], arrays["depth_conf"])

    metadata = {
        "geometry_backend": "g3t",
        "coordinate_system": "g3t_gravity_first_view_forward",
        "model_path": config.model_path,
        "repo_path": config.repo_path,
        "points_source": config.points_source,
        "img_load_resolution": config.img_load_resolution,
        "inference_resolution": config.inference_resolution,
        "seed": config.seed,
        "first_view_alignment": frame_alignment,
        "axes": {"+x": "first-frame right", "+y": "gravity/down", "+z": "first-frame forward"},
        "shapes": {
            "camera_pose": list(camera_pose.shape),
            "depth": list(depth.shape),
            "point_map": list(point_map.shape),
            "point_conf": list(point_conf.shape),
            "intrinsic": list(arrays["intrinsic"].shape),
            "w2g": list(arrays["w2g"].shape),
            "c2g": list(arrays["c2g"].shape),
        },
        "notes": (
            "point_map.npy stores gravity-aligned G3T world_points after a yaw-only "
            "canonicalization that maps the first camera's horizontal viewing direction to +Z."
        ),
    }
    with paths["metadata"].open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    LOGGER.info("Saved camera_pose with shape %s", getattr(camera_pose, "shape", None))
    LOGGER.info("Saved depth with shape %s", getattr(depth, "shape", None))
    LOGGER.info("Saved point_map with shape %s", getattr(point_map, "shape", None))
    LOGGER.info("Saved point_conf with shape %s", getattr(point_conf, "shape", None))
    return paths



def _validate_g3t_long_config(config: G3TLongConfig) -> None:
    if config.points_source not in {"point_head", "depth_head"}:
        raise ValueError("points_source must be 'point_head' or 'depth_head'.")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be greater than or equal to 1.")
    if config.overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0.")
    if config.overlap >= config.chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
    if config.loop_chunk_size < 1:
        raise ValueError("loop_chunk_size must be greater than or equal to 1.")


@dataclass(frozen=True)
class G3TLongDependencies:
    np: Any
    torch: Any
    g3t_cls: Any
    g3t_adapter_cls: Any
    g3t_long_cls: Any
    load_config: Any
    sim3utils: Any


def _load_g3t_long_dependencies(repo_path: str) -> G3TLongDependencies:
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        import numpy as np
        import torch
        from vggt.models.g3t import G3T
        from vggt_long.base_models.base_model import G3TAdapter
        from vggt_long.g3t_long import G3T_Long
        from vggt_long.loop_utils.config_utils import load_config
        from vggt_long.loop_utils import sim3utils
    except ImportError as exc:
        raise ImportError(
            "G3T-Long dependencies are unavailable. Install the official G3T-Long "
            "dependencies and set G3T_REPO to the local checkout path."
        ) from exc
    return G3TLongDependencies(
        np=np,
        torch=torch,
        g3t_cls=G3T,
        g3t_adapter_cls=G3TAdapter,
        g3t_long_cls=G3T_Long,
        load_config=load_config,
        sim3utils=sim3utils,
    )


def _load_official_g3t_long_config(deps: G3TLongDependencies, repo_path: Path, config_path: Path) -> dict[str, Any]:
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"G3T-Long config does not exist: {config_path}")
    cwd = Path.cwd()
    try:
        os.chdir(repo_path)
        return deps.load_config(str(config_path))
    finally:
        os.chdir(cwd)


def _configure_official_g3t_long_config(
    official_config: dict[str, Any],
    config: G3TLongConfig,
    repo_path: Path,
) -> None:
    official_config["Weights"]["model"] = "G3T"
    official_config["Weights"]["G3T"] = config.model_path
    for weight_name in ("SALAD", "DNIO", "DBoW"):
        weight_path = Path(official_config["Weights"][weight_name])
        if not weight_path.is_absolute():
            official_config["Weights"][weight_name] = str((repo_path / weight_path).resolve())
    official_config["Model"]["chunk_size"] = config.chunk_size
    official_config["Model"]["overlap"] = config.overlap
    official_config["Model"]["loop_chunk_size"] = config.loop_chunk_size
    official_config["Model"]["loop_enable"] = bool(config.loop_enable)
    official_config["Model"]["model_type"] = f"g3t_{config.points_source}"
    official_config["Model"]["delete_temp_files"] = not config.keep_cache
    official_config["Model"]["calib"] = False
    official_config["Model"].setdefault("Pointcloud_Save", {})["conf_threshold_coef"] = config.conf_thresh_multiplier_for_viz


def _validate_g3t_long_loop_runtime(official_config: dict[str, Any]) -> None:
    if not official_config["Model"]["loop_enable"]:
        return

    use_dbow = bool(official_config["Model"].get("useDBoW", False))
    required_weights = ("DBoW",) if use_dbow else ("SALAD", "DNIO")
    missing = [
        f"{name}={official_config['Weights'].get(name)}"
        for name in required_weights
        if not Path(official_config["Weights"].get(name, "")).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "G3T-Long loop closure weights are missing: " + ", ".join(missing)
        )

    if not use_dbow:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "G3T-Long SALAD loop closure requires faiss. Install a faiss-cpu "
                "or faiss-gpu build compatible with the current NumPy environment."
            ) from exc


def _patch_g3t_long_adapter_load(deps: G3TLongDependencies) -> None:
    def load(self: Any) -> None:
        path = self.config["Weights"]["G3T"]
        if path is None:
            model = deps.g3t_cls.from_pretrained("thatbrguy/g3t")
        else:
            model_path = Path(path)
            if model_path.is_dir():
                model = deps.g3t_cls.from_pretrained(str(model_path))
            elif model_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                model = deps.g3t_cls(enable_point=True, enable_depth=True, enable_gravity_camera_heads=True)
                model.load_state_dict(load_file(str(model_path), device="cpu"), strict=False)
            else:
                model = deps.g3t_cls(enable_point=True, enable_depth=True, enable_gravity_camera_heads=True)
                checkpoint = deps.torch.load(str(model_path), map_location="cpu")
                state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state, strict=False)
        model.eval()
        self.model = model.to(self.device)
    deps.g3t_adapter_cls.load = load


def _collect_g3t_long_dense_outputs(g3t_long_obj: Any, num_frames: int, np: Any) -> tuple[Any, Any]:
    point_maps: list[Any | None] = [None] * num_frames
    point_confs: list[Any | None] = [None] * num_frames
    for chunk_idx, (start, end) in enumerate(g3t_long_obj.chunk_indices):
        chunk_path = Path(g3t_long_obj.result_aligned_dir) / f"chunk_{chunk_idx}.npy"
        if not chunk_path.exists():
            chunk_path = Path(g3t_long_obj.result_unaligned_dir) / f"chunk_{chunk_idx}.npy"
        chunk = np.load(chunk_path, allow_pickle=True).item()
        points = chunk["world_points"]
        conf = chunk["world_points_conf"]
        offset = 0 if chunk_idx == 0 else min(g3t_long_obj.overlap, end - start)
        for local_idx in range(offset, end - start):
            global_idx = start + local_idx
            if global_idx >= num_frames:
                continue
            point_maps[global_idx] = points[local_idx]
            point_confs[global_idx] = conf[local_idx]
    missing = [idx for idx, item in enumerate(point_maps) if item is None]
    if missing:
        raise RuntimeError(f"G3T-Long did not produce dense point maps for frame indices: {missing[:10]}")
    return np.stack(point_maps, axis=0), np.stack(point_confs, axis=0)


def _save_g3t_long_outputs(
    output_dir: Path,
    config: G3TLongConfig,
    point_map: Any,
    point_conf: Any,
    intrinsic: Any,
    camera_pose: Any,
    w2g: Any,
    c2g: Any,
    g3t_long_obj: Any,
    deps: G3TLongDependencies,
) -> dict[str, Path]:
    point_map, camera_pose, w2g, frame_alignment = _canonicalize_g3t_first_view_forward(
        point_map=point_map,
        camera_pose=camera_pose,
        w2g=w2g,
        np=deps.np,
    )
    _validate_point_outputs(point_map=point_map, point_conf=point_conf)
    depth = deps.np.full(point_conf.shape + (1,), deps.np.nan, dtype=point_map.dtype)
    paths = {
        "camera_pose": output_dir / "camera_pose.npy",
        "depth": output_dir / "depth.npy",
        "point_map": output_dir / "point_map.npy",
        "point_conf": output_dir / "point_conf.npy",
        "intrinsic": output_dir / "intrinsic.npy",
        "w2g": output_dir / "w2g.npy",
        "c2g": output_dir / "c2g.npy",
        "metadata": output_dir / "g3t_long_metadata.json",
        "pointcloud": output_dir / "pointcloud.ply",
    }
    deps.np.save(paths["camera_pose"], camera_pose)
    deps.np.save(paths["depth"], depth)
    deps.np.save(paths["point_map"], point_map)
    deps.np.save(paths["point_conf"], point_conf)
    deps.np.save(paths["intrinsic"], intrinsic)
    deps.np.save(paths["w2g"], w2g)
    deps.np.save(paths["c2g"], c2g)
    pointcloud_files = sorted(Path(g3t_long_obj.pcd_dir).glob("*.ply"))
    if pointcloud_files:
        deps.sim3utils.merge_ply_files(g3t_long_obj.pcd_dir, str(paths["pointcloud"]))
        _transform_ply_coordinate_frame(
            pointcloud_path=paths["pointcloud"],
            old_from_new_rotation=deps.np.asarray(frame_alignment["old_from_new_rotation"]),
            np=deps.np,
        )
    else:
        LOGGER.info("No G3T-Long pointcloud shards found under %s; skipping pointcloud.ply export", g3t_long_obj.pcd_dir)
    metadata = {
        "geometry_backend": "g3t_long",
        "coordinate_system": "g3t_long_gravity_first_view_forward",
        "model_path": config.model_path,
        "repo_path": config.repo_path,
        "config_path": config.config_path,
        "points_source": config.points_source,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "loop_enable": config.loop_enable,
        "first_view_alignment": frame_alignment,
        "axes": {"+x": "first-frame right", "+y": "gravity/down", "+z": "first-frame forward"},
        "chunk_indices": [list(item) for item in g3t_long_obj.chunk_indices],
        "cache_dir": str(g3t_long_obj.cache_dir),
        "shapes": {
            "camera_pose": list(camera_pose.shape),
            "point_map": list(point_map.shape),
            "point_conf": list(point_conf.shape),
            "intrinsic": list(intrinsic.shape),
            "w2g": list(w2g.shape),
            "c2g": list(c2g.shape),
        },
        "notes": (
            "point_map.npy is reconstructed from aligned G3T-Long chunks and yaw-canonicalized "
            "so the first camera's horizontal viewing direction is +Z."
        ),
    }
    with paths["metadata"].open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return paths

def _load_config_from_environment(
    model_path: str | Path | None,
    repo_path: str | Path | None,
    points_source: str | None,
    img_load_resolution: int | None,
    inference_resolution: int | None,
    seed: int | None,
) -> G3TConfig:
    return G3TConfig(
        model_path=str(model_path or os.environ.get("G3T_MODEL", DEFAULT_G3T_MODEL)),
        repo_path=str(repo_path or os.environ.get("G3T_REPO", DEFAULT_G3T_REPO)),
        points_source=str(points_source or os.environ.get("G3T_POINTS_SOURCE", "point_head")),
        img_load_resolution=int(img_load_resolution or os.environ.get("G3T_IMG_LOAD_RESOLUTION", 1024)),
        inference_resolution=int(inference_resolution or os.environ.get("G3T_INFERENCE_RESOLUTION", 518)),
        seed=int(seed if seed is not None else os.environ.get("G3T_SEED", 42)),
        max_feed_forward_tokens=int(os.environ.get("G3T_MAX_FEED_FORWARD_TOKENS", 12_000)),
    )


def _estimate_feed_forward_tokens(config: G3TConfig, num_frames: int) -> int:
    patches_per_side = (config.inference_resolution + 13) // 14
    return int(num_frames * patches_per_side * patches_per_side)


def _validate_feed_forward_size(config: G3TConfig, num_frames: int) -> None:
    token_count = _estimate_feed_forward_tokens(config=config, num_frames=num_frames)
    if token_count <= config.max_feed_forward_tokens:
        return
    patches_per_side = (config.inference_resolution + 13) // 14
    suggested_frames = max(1, config.max_feed_forward_tokens // (patches_per_side * patches_per_side))
    raise ValueError(
        "G3T feed-forward inference is likely to run out of CUDA memory: "
        f"{num_frames} frames at {config.inference_resolution}x{config.inference_resolution} "
        f"produce about {token_count} attention tokens. "
        f"Reduce --max_frames to {suggested_frames} or lower, increase --frame_stride, "
        "or explicitly raise G3T_MAX_FEED_FORWARD_TOKENS if you know the GPU has enough memory. "
        "For long videos, G3T-Long or chunked reconstruction is needed; plain feed-forward is not suitable for dozens of frames."
    )


def _validate_config(config: G3TConfig) -> None:
    if config.points_source not in {"point_head", "depth_head"}:
        raise ValueError("points_source must be 'point_head' or 'depth_head'.")
    if config.img_load_resolution < 1:
        raise ValueError("img_load_resolution must be greater than or equal to 1.")
    if config.inference_resolution < 1:
        raise ValueError("inference_resolution must be greater than or equal to 1.")
    if config.max_feed_forward_tokens < 1:
        raise ValueError("max_feed_forward_tokens must be greater than or equal to 1.")


def _load_g3t_dependencies(repo_path: str) -> G3TDependencies:
    """Import official G3T feed-forward dependencies at runtime."""

    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from vggt.models.g3t import G3T
        from vggt.utils.geometry import make_4x4, unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as exc:
        raise ImportError(
            "G3T feed-forward dependencies are unavailable. Install the official "
            "G3T environment and set G3T_REPO to the local checkout path."
        ) from exc

    return G3TDependencies(
        np=np,
        torch=torch,
        functional=F,
        g3t_cls=G3T,
        load_and_preprocess_images_square=load_and_preprocess_images_square,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        make_4x4=make_4x4,
        unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
    )


def _set_seeds(deps: G3TDependencies, seed: int) -> None:
    deps.np.random.seed(seed)
    deps.torch.manual_seed(seed)
    if deps.torch.cuda.is_available():
        deps.torch.cuda.manual_seed(seed)
        deps.torch.cuda.manual_seed_all(seed)


def _release_cuda_memory(deps: G3TDependencies, device: str) -> None:
    gc.collect()
    if device == "cuda" and deps.torch.cuda.is_available():
        deps.torch.cuda.empty_cache()
        deps.torch.cuda.ipc_collect()
        LOGGER.info("Released G3T CUDA cache")


def _collect_frame_paths(frame_dir: Path) -> list[Path]:
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame path is not a directory: {frame_dir}")

    frame_paths = [path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    frame_paths.sort(key=_extract_frame_index)
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in: {frame_dir}")
    return frame_paths


def _extract_frame_index(frame_path: Path) -> int:
    stem = frame_path.stem
    if stem.isdigit():
        return int(stem)
    if stem.startswith("frame_") and stem.removeprefix("frame_").isdigit():
        return int(stem.removeprefix("frame_"))
    raise ValueError(
        "Frame names must be numeric, such as 0.jpg, or use this project's "
        f"frame_000000.jpg format: {frame_path.name}"
    )


def _pick_prediction(predictions: dict[str, Any], *candidate_keys: str) -> Any:
    for key in candidate_keys:
        if key in predictions:
            return predictions[key]
    raise KeyError(f"Missing G3T prediction key. Tried: {', '.join(candidate_keys)}")


def _to_numpy_array(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _canonicalize_g3t_first_view_forward(
    point_map: Any,
    camera_pose: Any,
    w2g: Any,
    np: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Keep G3T's +Y gravity axis and rotate yaw so first-camera forward is +Z."""

    point_map = np.asarray(point_map)
    camera_pose = np.asarray(camera_pose)
    w2g = np.asarray(w2g)
    if camera_pose.ndim != 3 or camera_pose.shape[0] < 1 or camera_pose.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"camera_pose must have shape [N,3,4] or [N,4,4], got {camera_pose.shape}")

    first_rotation = camera_pose[0, :3, :3]
    camera_forward = first_rotation.T @ np.asarray([0.0, 0.0, 1.0])
    horizontal_forward = camera_forward.copy()
    horizontal_forward[1] = 0.0
    norm = float(np.linalg.norm(horizontal_forward))
    if norm < 1e-8:
        camera_right = first_rotation.T @ np.asarray([1.0, 0.0, 0.0])
        camera_right[1] = 0.0
        right_norm = float(np.linalg.norm(camera_right))
        if right_norm < 1e-8:
            raise ValueError("Cannot determine first-camera horizontal viewing direction.")
        camera_right /= right_norm
        horizontal_forward = np.cross(camera_right, np.asarray([0.0, 1.0, 0.0]))
    else:
        horizontal_forward /= norm

    gravity_down = np.asarray([0.0, 1.0, 0.0])
    first_view_right = np.cross(gravity_down, horizontal_forward)
    first_view_right /= np.linalg.norm(first_view_right)
    old_from_new = np.stack([first_view_right, gravity_down, horizontal_forward], axis=1)

    transformed_points = np.matmul(point_map, old_from_new).astype(point_map.dtype, copy=False)
    homogeneous = np.eye(4, dtype=camera_pose.dtype)
    homogeneous[:3, :3] = old_from_new
    transformed_camera_pose = np.matmul(camera_pose, homogeneous).astype(camera_pose.dtype, copy=False)
    transformed_w2g = np.matmul(w2g, homogeneous).astype(w2g.dtype, copy=False)

    report = {
        "method": "first_camera_opencv_forward_projected_to_xz",
        "camera_forward_before": [float(value) for value in camera_forward],
        "horizontal_forward_before": [float(value) for value in horizontal_forward],
        "old_from_new_rotation": old_from_new.tolist(),
        "target_forward_axis": "+z",
        "gravity_axis": "+y",
    }
    return transformed_points, transformed_camera_pose, transformed_w2g, report


def _transform_ply_coordinate_frame(pointcloud_path: Path, old_from_new_rotation: Any, np: Any) -> None:
    """Apply the same world-frame rotation to the optional merged PLY export."""

    try:
        import trimesh

        geometry = trimesh.load(str(pointcloud_path), process=False)
        transform = np.eye(4)
        transform[:3, :3] = old_from_new_rotation.T
        geometry.apply_transform(transform)
        geometry.export(str(pointcloud_path))
    except Exception:
        LOGGER.exception("Failed to canonicalize optional G3T-Long pointcloud.ply")
        raise


def _validate_point_outputs(point_map: Any, point_conf: Any) -> None:
    point_map_shape = getattr(point_map, "shape", None)
    point_conf_shape = getattr(point_conf, "shape", None)
    if point_map_shape is None:
        raise ValueError("point_map must expose a shape.")
    if point_conf_shape is None:
        raise ValueError("point_conf must expose a shape.")
    if len(point_map_shape) != 4 or point_map_shape[-1] != 3:
        raise ValueError(f"point_map must have shape [N,H,W,3], got {tuple(point_map_shape)}")
    if len(point_conf_shape) != 3:
        raise ValueError(f"point_conf must have shape [N,H,W], got {tuple(point_conf_shape)}")
    if tuple(point_map_shape[:3]) != tuple(point_conf_shape):
        raise ValueError(
            "point_map and point_conf must share [N,H,W] dimensions, got "
            f"{tuple(point_map_shape)} and {tuple(point_conf_shape)}"
        )


def _configure_file_logging(log_path: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path = str(log_path.resolve())

    for handler in LOGGER.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved_log_path:
            return

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    LOGGER.addHandler(file_handler)
