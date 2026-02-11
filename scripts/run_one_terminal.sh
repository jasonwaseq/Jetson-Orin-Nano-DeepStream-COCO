#!/usr/bin/env bash
set -euo pipefail

APP_CFG="/home/group7/jetson-deepstream-coco/configs/deepstream_app_usb_yolo.txt"
CAM_DEV="${CAM_DEV:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS_N="${FPS_N:-80}"
FPS_D="${FPS_D:-1}"
RTSP_PORT="${RTSP_PORT:-8554}"
UDP_PORT="${UDP_PORT:-5400}"

if [[ ! -e "${CAM_DEV}" ]]; then
  echo "Camera device not found: ${CAM_DEV}"
  exit 1
fi

if [[ ! -f "${APP_CFG}" ]]; then
  echo "DeepStream config not found: ${APP_CFG}"
  exit 1
fi

CAM_NODE="${CAM_DEV#/dev/video}"
if [[ ! "${CAM_NODE}" =~ ^[0-9]+$ ]]; then
  echo "Unsupported camera device name: ${CAM_DEV}"
  exit 1
fi

echo "[1/4] Probing camera caps on ${CAM_DEV} ..."
v4l2-ctl -d "${CAM_DEV}" --list-formats-ext >/dev/null

echo "[2/4] Setting DeepStream source/sink in ${APP_CFG} ..."
perl -0777 -i -pe "s/camera-v4l2-dev-node=\\d+/camera-v4l2-dev-node=${CAM_NODE}/g" "${APP_CFG}"
perl -0777 -i -pe "s/camera-width=\\d+/camera-width=${WIDTH}/g; s/camera-height=\\d+/camera-height=${HEIGHT}/g" "${APP_CFG}"
perl -0777 -i -pe "s/camera-fps-n=\\d+/camera-fps-n=${FPS_N}/g; s/camera-fps-d=\\d+/camera-fps-d=${FPS_D}/g" "${APP_CFG}"
perl -0777 -i -pe "s/rtsp-port=\\d+/rtsp-port=${RTSP_PORT}/g; s/udp-port=\\d+/udp-port=${UDP_PORT}/g" "${APP_CFG}"

echo "[3/4] Launching DeepStream..."
echo "RTSP: rtsp://<JETSON_IP>:${RTSP_PORT}/ds-test"
echo "Camera: ${CAM_DEV} (${WIDTH}x${HEIGHT} @ ${FPS_N}/${FPS_D})"

cd /home/group7/jetson-deepstream-coco
echo "[4/4] Running: deepstream-app -c configs/deepstream_app_usb_yolo.txt"
exec deepstream-app -c configs/deepstream_app_usb_yolo.txt
