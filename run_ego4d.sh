#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

source /data/wy/tools/vpn/use_proxy.sh
export PYTHONUNBUFFERED=1

EGO4D_PYTHON="${EGO4D_PYTHON:-/data/wy/miniconda3/envs/ego4d/bin/python}"
EGO4D_ROOT="${EGO4D_ROOT:-/data/wy/data/ego4d}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${EGO4D_ROOT}/pipeline_processed}"
ALL_SCENES="${ALL_SCENES:-1}"
SUBSET_SIZE="${SUBSET_SIZE:-24}"
CLIP_DURATION="${CLIP_DURATION:-20}"
MIN_TRACK_BOXES="${MIN_TRACK_BOXES:-12}"
MIN_BOX_AREA="${MIN_BOX_AREA:-0.003}"
MAX_SOURCE_DURATION="${MAX_SOURCE_DURATION:-1800}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
WORKERS="${WORKERS:-4}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-4}"
PREFETCH_CLIPS="${PREFETCH_CLIPS:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-5}"
MAX_FRAMES="${MAX_FRAMES:-96}"
MULTIVIEW_COUNT="${MULTIVIEW_COUNT:-3}"
MAX_SCENES="${MAX_SCENES:-0}"

ARGS=(
  --ego4d-root "${EGO4D_ROOT}"
  --processed-root "${PROCESSED_ROOT}"
  --subset-size "${SUBSET_SIZE}"
  --clip-duration "${CLIP_DURATION}"
  --min-track-boxes "${MIN_TRACK_BOXES}"
  --min-box-area "${MIN_BOX_AREA}"
  --max-source-duration "${MAX_SOURCE_DURATION}"
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}"
  --workers "${WORKERS}"
  --download-workers "${DOWNLOAD_WORKERS}"
  --prefetch-clips "${PREFETCH_CLIPS}"
  --frame-stride "${FRAME_STRIDE}"
  --max-frames "${MAX_FRAMES}"
  --multiview-count "${MULTIVIEW_COUNT}"
  --max-scenes "${MAX_SCENES}"
)

[[ "${ALL_SCENES}" == "1" ]] && ARGS+=(--all-scenes)
[[ "${PLAN_ONLY:-0}" == "1" ]] && ARGS+=(--plan-only)
[[ "${KEEP_CLIPS:-0}" == "1" ]] && ARGS+=(--keep-clips)
[[ "${KEEP_SOURCE_VIDEOS:-0}" == "1" ]] && ARGS+=(--keep-source-videos)
[[ "${KEEP_INTERMEDIATES:-0}" == "1" ]] && ARGS+=(--keep-intermediates)
[[ "${STOP_ON_ERROR:-0}" == "1" ]] && ARGS+=(--stop-on-error)

exec "${EGO4D_PYTHON}" run_ego4d.py "${ARGS[@]}"
