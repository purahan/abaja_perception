"""
Manual sensor_msgs/Image <-> numpy BGR8 conversion.

Why not cv_bridge: on ROS2 Humble, cv_bridge's compiled extension is
built against a specific OpenCV build. If a pip-installed opencv-python
is also present (as it usually is, since Ultralytics depends on it),
the two disagree on internal type constants and cv_bridge raises
'KeyError: 16' or similar from its own lookup tables — a known, common
conflict, not something wrong with your setup specifically. Since we
only ever need bgr8 <-> numpy, it's simpler and more robust to just do
the conversion by hand and drop the cv_bridge dependency entirely.
"""

import numpy as np
from sensor_msgs.msg import Image


def imgmsg_to_bgr8(msg: Image) -> np.ndarray:
    if msg.encoding != 'bgr8':
        raise ValueError(f"Expected encoding 'bgr8', got '{msg.encoding}'")
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape(msg.height, msg.width, 3)
    return arr


def bgr8_to_imgmsg(frame: np.ndarray, header=None) -> Image:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f'Expected HxWx3 BGR array, got shape {frame.shape}')
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height, msg.width = frame.shape[0], frame.shape[1]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg
