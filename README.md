# Jetson DeepStream COCO (USB camera + RTSP)

Run YOLOv8 COCO detection on a Jetson Orin Nano with a See3CAM_CU27 USB camera and stream annotated video over RTSP.

## Requirements
- Jetson with DeepStream 7.1
- See3CAM_CU27 connected as /dev/video0
- GStreamer (with RTSP plugins) on the viewer machine
- Tailnet/VPN connectivity if viewing remotely

## Quick start (on Jetson)

```bash
# Start DeepStream + RTSP (all parameters are optional; shown with defaults)
CAM_DEV=/dev/video0 WIDTH=1280 HEIGHT=720 FPS_N=80 FPS_D=1 \
  RTSP_PORT=8554 UDP_PORT=5400 \
  /home/group7/jetson-deepstream-coco/scripts/run_one_terminal.sh
```

If you see "Device /dev/video0 is busy", stop any existing DeepStream process and retry:
```bash
sudo pkill -f deepstream-app
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAM_DEV` | `/dev/video0` | Camera device path |
| `WIDTH` | `1280` | Capture width (pixels) |
| `HEIGHT` | `720` | Capture height (pixels) |
| `FPS_N` | `80` | FPS numerator |
| `FPS_D` | `1` | FPS denominator |
| `RTSP_PORT` | `8554` | RTSP server port |
| `UDP_PORT` | `5400` | Internal UDP relay port |
| `DS_YOLO_LOG` | `1` | Enable YOLO detection logging |

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
- **Camera caps negotiation failure (Option B):** ensure `FPS` matches the camera's actual delivery rate.

## Notes
- The TensorRT engine can take several minutes to build on first run.
- `run_one_terminal.sh` never modifies the config files — it patches a temporary copy.
- To stop tracking `run.log` in git (it is already in `.gitignore`):
  ```bash
  git rm --cached run.log
  git commit -m "chore: untrack run.log"
  ```
