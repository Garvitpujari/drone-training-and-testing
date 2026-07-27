from ultralytics import YOLO
import numpy as np


class DroneDetector:
    """
    YOLO Detector Only
    ------------------
    No ByteTrack
    No model.track()
    No LAP

    Returns:
        detections = [
            {
                "bbox": [x1,y1,x2,y2],
                "conf": confidence,
                "class": class_id
            },
            ...
        ]
    """

    def __init__(self, model_path, conf=0.25):

        print("[INFO] Loading YOLO Detector...")

        self.model = YOLO(model_path)
        self.conf = conf

        print("[INFO] Detector Ready.")

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            verbose=False
        )

        detections = []

        if len(results) == 0:
            return detections

        boxes = results[0].boxes

        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()

        confs = boxes.conf.cpu().numpy()

        classes = boxes.cls.cpu().numpy().astype(int)

#         xyxy =
# [
#  [100,150,180,220],
#  [420,80,480,140]
# ]

# confs =
# [
# 0.94,
# 0.89
# ]

# classes =
# [
# 0,
# 0
# ]

        for box, score, cls in zip(xyxy, confs, classes):

            detections.append({

                "bbox": np.array(box, dtype=float),

                "conf": float(score),

                "class": int(cls)

            })

        return detections

#     [
#     {
#         "bbox": np.array([100.,150.,180.,220.]),
#         "conf": 0.94,
#         "class": 0
#     },
#     {
#         "bbox": np.array([420.,80.,480.,140.]),
#         "conf": 0.89,
#         "class": 0
#     }
# ]