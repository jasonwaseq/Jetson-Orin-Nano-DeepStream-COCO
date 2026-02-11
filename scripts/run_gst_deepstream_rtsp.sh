#!/usr/bin/env bash
set -euo pipefail

CFG="/home/group7/jetson-deepstream-coco/configs/config_infer_primary_yolov8n.txt"
PORT=8554

# Force camera to MJPG 1280x720 (reliable over USB)
sudo v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG >/dev/null 2>&1 || true

echo "RTSP will be available at: rtsp://<JETSON_IP>:${PORT}/ds-test"

# Notes:
# - nvv4l2decoder mjpeg=1 does HW MJPEG decode
# - nvstreammux requires NVMM NV12 buffers
# - x264enc fallback is used if nvv4l2h264enc is unavailable

if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
  ENC="nvv4l2h264enc bitrate=4000000 insert-sps-pps=1 idrinterval=30 preset-level=1"
else
  ENC="x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30"
fi

gst-launch-1.0 -e \
  v4l2src device=/dev/video0 io-mode=2 ! \
    image/jpeg,width=1280,height=720,framerate=30/1 ! \
    nvv4l2decoder mjpeg=1 ! \
    nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=1280,height=720 ! \
    queue ! mux.sink_0 \
  nvstreammux name=mux batch-size=1 width=1280 height=720 live-source=1 batched-push-timeout=40000 ! \
    nvinfer config-file-path=${CFG} ! \
    nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so \
              ll-config-file=/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
              tracker-width=640 tracker-height=384 ! \
    nvdsosd ! \
    nvvideoconvert ! video/x-raw,format=I420 ! \
    ${ENC} ! h264parse ! rtph264pay pt=96 config-interval=1 ! \
    udpsink host=127.0.0.1 port=5400 sync=false async=false &

PIPE_PID=$!

# In-process RTSP server (GStreamer)
gst-rtsp-server \
  --port=${PORT} \
  "( udpsrc name=pay0 port=5400 caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" )"

kill ${PIPE_PID} >/dev/null 2>&1 || true
