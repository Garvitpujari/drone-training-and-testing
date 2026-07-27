import numpy as np
from scipy.optimize import linear_sum_assignment
# imports the Hungarian Algorithm, which solves the assignment problem


# [
#     {
#         "bbox": [100,150,180,220],
#         "conf": 0.94,
#         "class": 0
#     },
#     {
#         "bbox": [420,80,480,140],
#         "conf": 0.89,
#         "class": 0
#     }
# ]

class Track:
    def __init__(self, track_id, bbox, confidence, cls):

        self.id = track_id

        self.bbox = np.array(bbox, dtype=np.float32)

        self.confidence = confidence

        self.cls = cls

        self.age = 0   # number of frames for which drone hs not been detected 

        self.hits = 1  # no.oif frame for which the drone has got detected 


class CustomTracker:

    def __init__(
        self,
        iou_threshold=0.3,  #minimum IoU required to consider a detection as the same object
        max_age=20          
    ):

        self.iou_threshold = iou_threshold
 
        self.max_age = max_age    # delete track after max_age frames 

        self.next_id = 0          # next free id 

        self.tracks = []    

#    self.tracks =
# [
#     Track(id=0),
#     Track(id=1)
# ]

    # --------------------------------------------------------IoU (Intersection over Union) is a metric that tells us how much two bounding boxes overlap------

    def iou(self, a, b):  # bounding boxes of existing and new track 

        x1 = max(a[0], b[0]) 
        y1 = max(a[1], b[1])

        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        if inter == 0:
            return 0

        area1 = (a[2]-a[0]) * (a[3]-a[1])
        area2 = (b[2]-b[0]) * (b[3]-b[1])

        return inter / (area1 + area2 - inter)

    # --------------------------------------------------------

    def update(self, detections):

        if len(self.tracks) == 0:


#             detections = [
#     {"bbox":[50,80,170,210],"conf":0.95,"class":0},

#     {"bbox":[300,120,380,220],"conf":0.90,"class":0}
# ]


    for det in detections:

                self.tracks.append(

                    Track(

                        self.next_id,

                        det["bbox"],

                        det["conf"],

                        det["class"]

                    )

                )

                self.next_id += 1



           return self.export_tracks()


# self.tracks =
# [
#     Track(id=0, bbox=[50,80,170,210]),
#     Track(id=1, bbox=[300,100,380,200])
# ]

# detections =
# [
#     {"bbox":[55,85,175,215]},      # Drone moved slightly
#     {"bbox":[305,105,385,205]}     # Second drone moved slightly
# ]

    cost = np.ones((len(self.tracks), len(detections)))

    for i, tr in enumerate(self.tracks):

            for j, det in enumerate(detections):

                cost[i, j] = 1 - self.iou(

                    tr.bbox,

                    det["bbox"]

                )
                # filling the  cost matrix

    rows, cols = linear_sum_assignment(cost)

    matched_tracks = set()

    matched_dets = set()

    for r, c in zip(rows, cols):

            if cost[r, c] <= (1 - self.iou_threshold):

                self.tracks[r].bbox = detections[c]["bbox"]

                self.tracks[r].confidence = detections[c]["conf"]

                self.tracks[r].cls = detections[c]["class"]

                self.tracks[r].age = 0

                self.tracks[r].hits += 1

                matched_tracks.add(r)

                matched_dets.add(c)

    for i, tr in enumerate(self.tracks):

            if i not in matched_tracks:

                tr.age += 1

    self.tracks = [

            t for t in self.tracks

            if t.age <= self.max_age

    ]

    for i, det in enumerate(detections):

            if i not in matched_dets:

                self.tracks.append(

                    Track(

                        self.next_id,

                        det["bbox"],

                        det["conf"],

                        det["class"]

                    )

                )

                self.next_id += 1

    return self.export_tracks()

    # --------------------------------------------------------

    def export_tracks(self):

        output = []

        for t in self.tracks:

            output.append({

                "id": t.id,

                "bbox": t.bbox,

                "conf": t.confidence,

                "class": t.cls

            })

        return output