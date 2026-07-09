# abaja_perception

ROS2 package for the aBAJA SAEINDIA 2026 Object Detection and Classification
event. Target platform: Jetson Orin Nano/NX.

## Requirements

- **ROS2 Humble** (prerequisite, installed via apt, not pip): https://docs.ros.org/en/humble/Installation.html
- Runtime Python deps: `pip3 install -r requirements.txt --break-system-packages`
- Training/dataset Python deps (separate venv, see below): `pip install -r requirements-training.txt`

```bash
python3 -m venv ~/abaja_training_env
source ~/abaja_training_env/bin/activate
pip install -r requirements-training.txt
```
Keeping these two dependency sets in separate environments avoids a
known numpy/opencv/matplotlib version conflict between `fiftyone` and
ROS2's system Python (see git history / commit messages if curious why).

## Trained models

Trained weights live in `models/`, not `runs/` (which is gitignored since
full training runs are large and regenerable). After training:
```bash
mkdir -p models
cp runs/detect/train/weights/best.pt models/abaja_v1.pt
cp runs/detect/train/results.png models/abaja_v1_results.png
cp runs/detect/train/results.csv models/abaja_v1_results.csv
```
Then set `torch_weights` in `config/classes.yaml` to point at
`models/abaja_v1.pt` and `use_custom_classes: true`.

Current model status (update this as you retrain):
- Classes with real training data: 13 of 15 (missing: traffic_cone)
- Overall mAP50: 0.84 (see `models/abaja_v1_results.png` for full breakdown)
- Known weak spots: bicyclist (COCO's bicycle class doesn't guarantee a
  rider), car, cow (only 51 training images)
- Sign/light/barricade classes (~0.99 mAP50) are validated only against
  synthetic data so far - needs a real-photo sanity check before trusting
  those numbers in the field

## Getting the code

```bash
git clone <your-repo-url>
cd abaja_perception
```

## Workflow

1. **Data**: build a YOLO-format dataset covering all 15 PART C classes.
   - Common classes (car, pedestrian, bicyclist, two-wheeler, cow, cone):
     start from IDD / BDD100K / COCO, fine-tune.
   - Event-specific classes (5 speed-limit signs, 3 light states, steel
     barricade): synthesize + augment, and capture real footage once you
     know what soft-targets the organizers use.

2. **Train**: standard Ultralytics YOLO fine-tuning (`yolo detect train
   data=... model=yolov8s.pt`). A trained checkpoint is included in
   `models/` (see the "Trained models" section above) — re-run this step
   yourself once you add more data (e.g. traffic_cone, more cow images).

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
