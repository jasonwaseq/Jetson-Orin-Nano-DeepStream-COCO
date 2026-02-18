#!/usr/bin/env bash
set -euo pipefail

# Resolve project root relative to this script (works regardless of CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_CFG="${PROJECT_ROOT}/configs/deepstream_app_usb_yolo.txt"
CAM_DEV="${CAM_DEV:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS_N="${FPS_N:-80}"
FPS_D="${FPS_D:-1}"
RTSP_PORT="${RTSP_PORT:-8554}"
UDP_PORT="${UDP_PORT:-5400}"
DS_YOLO_LOG="${DS_YOLO_LOG:-1}"

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [[ ! -e "${CAM_DEV}" ]]; then
  echo "ERROR: Camera device not found: ${CAM_DEV}" >&2
  exit 1
fi

if [[ ! -f "${APP_CFG}" ]]; then
  echo "ERROR: DeepStream config not found: ${APP_CFG}" >&2
  exit 1
fi

CAM_NODE="${CAM_DEV#/dev/video}"
if [[ ! "${CAM_NODE}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Unsupported camera device name: ${CAM_DEV}" >&2
  exit 1
fi

if ! command -v v4l2-ctl &>/dev/null; then
  echo "WARNING: v4l2-ctl not found — skipping camera probe." >&2
else
  echo "[1/4] Probing camera caps on ${CAM_DEV} ..."
  v4l2-ctl -d "${CAM_DEV}" --list-formats-ext >/dev/null || \
    echo "WARNING: v4l2-ctl probe failed (camera may still work)." >&2
fi

if ! command -v deepstream-app &>/dev/null; then
  echo "ERROR: deepstream-app not found in PATH. Is DeepStream installed?" >&2
  exit 1
fi

# ── Create a temp config so we never mutate the source file ─────────────────

TMP_CFG="$(mktemp /tmp/deepstream_app_XXXXXX.txt)"
trap 'rm -f "${TMP_CFG}"' EXIT INT TERM

cp "${APP_CFG}" "${TMP_CFG}"

echo "[2/4] Patching DeepStream config (temp copy: ${TMP_CFG}) ..."
perl -0777 -i -pe "s/camera-v4l2-dev-node=\d+/camera-v4l2-dev-node=${CAM_NODE}/g" "${TMP_CFG}"
perl -0777 -i -pe "s/camera-width=\d+/camera-width=${WIDTH}/g; s/camera-height=\d+/camera-height=${HEIGHT}/g" "${TMP_CFG}"
perl -0777 -i -pe "s/camera-fps-n=\d+/camera-fps-n=${FPS_N}/g; s/camera-fps-d=\d+/camera-fps-d=${FPS_D}/g" "${TMP_CFG}"
perl -0777 -i -pe "s/rtsp-port=\d+/rtsp-port=${RTSP_PORT}/g; s/udp-port=\d+/udp-port=${UDP_PORT}/g" "${TMP_CFG}"

echo "[3/4] Launching DeepStream..."
echo "RTSP:   rtsp://<JETSON_IP>:${RTSP_PORT}/ds-test"
echo "Camera: ${CAM_DEV} (${WIDTH}x${HEIGHT} @ ${FPS_N}/${FPS_D})"
echo "YOLO log: ${DS_YOLO_LOG}"

cd "${PROJECT_ROOT}"
echo "[4/4] Running: deepstream-app -c ${TMP_CFG}"
export DS_YOLO_LOG
exec deepstream-app -c "${TMP_CFG}"
