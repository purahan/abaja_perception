# abaja_perception

ROS2 package for the aBAJA SAEINDIA 2026 Object Detection and Classification
event. Target platform: Jetson Orin Nano/NX.

## Workflow

1. **Data**: build a YOLO-format dataset covering all 15 PART C classes.
   - Common classes (car, pedestrian, bicyclist, two-wheeler, cow, cone):
     start from IDD / BDD100K / COCO, fine-tune.
   - Event-specific classes (5 speed-limit signs, 3 light states, steel
     barricade): synthesize + augment, and capture real footage once you
     know what soft-targets the organizers use.

2. **Train**: standard Ultralytics YOLO fine-tuning (`yolo detect train
   data=... model=yolov8s.pt`). Not included here — this repo starts from
   a trained `best.pt`.

3. **Export**: `scripts/export_tensorrt.py` converts `best.pt` -> TensorRT
   engine (FP16 or INT8) and prints an FPS benchmark. Run both precisions,
   keep the numbers for your technical presentation.

4. **Tune thresholds against the actual rubric**: `scripts/threshold_sweep.py`
   sweeps per-class confidence thresholds to maximize expected event score
   (not mAP) using your validation set + the +5/+2/-2/-3 scoring rules.
   Merge the output into `config/classes.yaml`.

5. **Set `engine_path`** in `config/classes.yaml` to your exported engine.

6. **Wire up hardware-specific bits** (marked `TODO` in
   `abaja_perception/perception_node.py`):
   - Camera topic name/type (currently assumes `sensor_msgs/Image` on
     `/camera/image_raw` — change if your driver differs).
   - RADAR message type/topic, and the `radar_confirms()` gating logic —
     currently a no-op placeholder.

7. **Run**:
   ```
   colcon build --packages-select abaja_perception
   source install/setup.bash
   ros2 launch abaja_perception perception.launch.py
   ```
   This displays the live overlay window and records to
   `/mnt/user-data/outputs/operational_run.mp4` — submit that file after
   the operational evaluation as required by B.3.

## Testing with a laptop/USB webcam (before real vehicle hardware is available)

You don't need the actual vehicle's camera or RADAR to prove the pipeline
works. `webcam_publisher.py` publishes frames from any USB/laptop webcam
on the same topic `perception_node.py` expects, so the rest of the stack
(inference -> tracking -> overlay -> recording) runs unmodified.

```bash
colcon build --packages-select abaja_perception
source install/setup.bash
ros2 launch abaja_perception test_webcam.launch.py device_id:=0
```

`device_id:=0` is usually the default webcam; if you have multiple
cameras try `1`, `2`, etc., or check `ls /dev/video*` on Linux to see
what's available.

This brings up both the webcam publisher and the perception node
together, so you get the live overlay window and the recorded `.mp4`
exactly like the real run — swapping in the actual vehicle camera later
is just changing the topic/driver in `perception_node.py`'s subscription,
nothing else in the pipeline changes.

## Design notes (useful for the technical presentation)

- **Tracker/smoothing** (`tracker.py`): a plain greedy IoU tracker with a
  majority-vote class over the last N frames. Exists specifically because
  the rulebook penalizes flickering detections and forbids offline
  post-processing — smoothing has to happen live, in-loop.
- **Score-aware thresholds**: confidence thresholds are tuned per class
  against the actual point rubric, not generic mAP — pedestrian/cow/
  bicyclist get lower thresholds (favor recall, since missing them costs
  -3) while other classes are tuned more toward precision.
- **Camera+RADAR gating**: `radar_confirms()` is a stub for suppressing
  camera false positives (e.g. shadows) using RADAR range/bearing
  correlation — implement once you have camera-radar extrinsic
  calibration done.
