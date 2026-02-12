# Jetson DeepStream COCO (USB camera + RTSP)

Run YOLOv8 COCO detection on a Jetson Orin Nano with a See3CAM_CU27 USB camera and stream annotated video over RTSP.

## Requirements
- Jetson with DeepStream 7.1
- See3CAM_CU27 connected as /dev/video0
- GStreamer (with RTSP plugins) on the viewer machine
- Tailnet/VPN connectivity if viewing remotely

## Quick start (on Jetson)
```bash
# Start DeepStream + RTSP
CAM_DEV=/dev/video0 WIDTH=1280 HEIGHT=720 FPS_N=80 FPS_D=1 \
  RTSP_PORT=8554 UDP_PORT=5400 \
  /home/group7/jetson-deepstream-coco/scripts/run_one_terminal.sh
```

If you see “Device /dev/video0 is busy”, stop any existing DeepStream process and retry.
sudo pkill -f deepstream-app

## View the stream
**Remote (Windows/macOS/Linux with GUI):**
- Open VLC → Media → Open Network Stream
- URL: `rtsp://<JETSON_IP>:8554/ds-test`
- Click **Play** (no transcoding)

**Headless on Jetson (validation):**
```bash
gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/ds-test protocols=tcp latency=200 \
  ! rtph264depay ! h264parse ! nvv4l2decoder ! fakesink sync=false
```

## Key configs
- App config: `configs/deepstream_app_usb_yolo.txt`
- Model config: `configs/config_infer_primary_yolov8n.txt`
- Models: `models/`

## Troubleshooting
- **No boxes:** ensure `display-bbox=1` is set in the OSD section and the engine matches the ONNX.
- **Latency/glitching in VLC:** reduce network caching to 100–200 ms.
- **RTSP timeout from WSL:** WSL must have Tailnet access; use Windows VLC if needed.

## Notes
The TensorRT engine can take several minutes to build on first run.
