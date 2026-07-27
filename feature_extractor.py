# ============================================================
# Feature Extractor
# ============================================================

import math


class FeatureExtractor:  
    # bounding box and to 9 features 

    def __init__(self):

        # Stores previous information for every track
        self.previous = {}
#   # updated as video is processed
#     0:{
#         "cx":120,
#         "cy":200,
#         "speed":15
#     },

#     1:{
#         "cx":350,
#         "cy":90,
#         "speed":12
#     }

# }

    def extract(self, track_id, box):

        """
        Parameters
        ----------
        track_id : int

        box : (x1,y1,x2,y2)

        Returns
        -------
        Dictionary containing

        cx
        cy
        width
        height
        vx
        vy
        speed
        heading
        acceleration
        """

        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # First observation of this object
        if track_id not in self.previous:


            self.previous[track_id] = {

                "cx": cx,
                "cy": cy,
                "speed":0

            }

            return {

                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "vx": 0,
                "vy": 0,
                "speed": 0,
                "heading": 0,
                "acceleration": 0

            }

        # -----------------------------
        # Previous values
        # -----------------------------

        prev = self.previous[track_id]

        vx = cx - prev["cx"]
        vy = cy - prev["cy"]

        speed = math.sqrt(vx**2 + vy**2)

        heading = math.atan2(vy, vx)

        acceleration = speed - prev["speed"]

        # -----------------------------
        # Update history
        # -----------------------------

        self.previous[track_id] = {

            "cx": cx,
            "cy": cy,
            "speed": speed

        }

        return {

            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "heading": heading,
            "acceleration": acceleration

        }