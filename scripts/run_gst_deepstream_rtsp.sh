#!/usr/bin/env bash
set -euo pipefail

# Resolve project root relative to this script (works regardless of CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CFG="${PROJECT_ROOT}/configs/config_infer_primary_yolov8n.txt"
PORT="${RTSP_PORT:-8554}"
UDP_PORT="${UDP_PORT:-5400}"
CAM_DEV="${CAM_DEV:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
# FPS used for GStreamer caps negotiation (must match what the camera delivers)
FPS="${FPS:-30}"

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [[ ! -e "${CAM_DEV}" ]]; then
  echo "ERROR: Camera device not found: ${CAM_DEV}" >&2
  exit 1
fi

if [[ ! -f "${CFG}" ]]; then
  echo "ERROR: Inference config not found: ${CFG}" >&2
  exit 1
fi

# Force camera to MJPG at the requested resolution (reliable over USB)
if command -v v4l2-ctl &>/dev/null; then
  sudo v4l2-ctl -d "${CAM_DEV}" \
    --set-fmt-video=width=${WIDTH},height=${HEIGHT},pixelformat=MJPG \
    >/dev/null 2>&1 || true
else
  echo "WARNING: v4l2-ctl not found — skipping camera format setup." >&2
fi

echo "RTSP will be available at: rtsp://<JETSON_IP>:${PORT}/ds-test"

# ── Encoder selection ────────────────────────────────────────────────────────
# Notes:
#   - nvv4l2decoder mjpeg=1 does HW MJPEG decode
#   - nvstreammux requires NVMM NV12 buffers
#   - x264enc fallback is used if nvv4l2h264enc is unavailable

if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
  ENC="nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 idrinterval=30 preset-level=1"
else
  ENC="x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30"
fi

# ── Launch GStreamer pipeline in the background ──────────────────────────────

gst-launch-1.0 -e \
  v4l2src device="${CAM_DEV}" io-mode=2 ! \
    "image/jpeg,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1" ! \
    nvv4l2decoder mjpeg=1 ! \
    nvvidconv ! "video/x-raw(memory:NVMM),format=NV12,width=${WIDTH},height=${HEIGHT}" ! \
    queue ! mux.sink_0 \
  nvstreammux name=mux batch-size=1 width="${WIDTH}" height="${HEIGHT}" \
    live-source=1 batched-push-timeout=40000 ! \
    nvinfer config-file-path="${CFG}" ! \
    nvtracker \
      ll-lib-file=/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so \
      ll-config-file=/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
      tracker-width=640 tracker-height=384 ! \
    nvdsosd ! \
    nvvideoconvert ! "video/x-raw,format=I420" ! \
    ${ENC} ! h264parse ! rtph264pay pt=96 config-interval=1 ! \
    udpsink host=127.0.0.1 port="${UDP_PORT}" sync=false async=false &

PIPE_PID=$!

# Kill the pipeline on any exit (normal, Ctrl+C, or error)
trap 'echo "Stopping pipeline (PID ${PIPE_PID})..."; kill "${PIPE_PID}" 2>/dev/null || true' EXIT INT TERM

# ── Start RTSP server ─────────────────────────────────────────────────────────
# Prefer the standalone binary if available; fall back to Python GstRtspServer
# (GstRtspServer Python bindings ship with DeepStream / Jetson GStreamer stack)

LAUNCH_CAPS="application/x-rtp,media=video,encoding-name=H264,payload=96"
LAUNCH_PIPELINE="( udpsrc name=pay0 port=${UDP_PORT} caps=\"${LAUNCH_CAPS}\" )"

if command -v gst-rtsp-server &>/dev/null; then
  gst-rtsp-server --port="${PORT}" "${LAUNCH_PIPELINE}"
else
  echo "INFO: gst-rtsp-server binary not found — using Python GstRtspServer." >&2
  python3 - <<PYEOF
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)
loop   = GLib.MainLoop()
server = GstRtspServer.RTSPServer.new()
server.set_service("${PORT}")
mounts = server.get_mount_points()
factory = GstRtspServer.RTSPMediaFactory.new()
factory.set_launch("${LAUNCH_PIPELINE}")
factory.set_shared(True)
mounts.add_factory("/ds-test", factory)
server.attach(None)
print("RTSP server listening on rtsp://0.0.0.0:${PORT}/ds-test")
try:
    loop.run()
except KeyboardInterrupt:
    pass
PYEOF
fi
