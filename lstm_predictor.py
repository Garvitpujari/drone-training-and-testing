# ============================================================
# LSTM Predictor
# ============================================================

# This file is responsible for taking the features from feature_extractor.py and predicting the drone's future trajectory using the trained LSTM model.

import torch
import torch.nn as nn
import numpy as np
import joblib
from collections import defaultdict, deque


# ------------------------------------------------------------
# LSTM MODEL
# ------------------------------------------------------------

class DroneTrajectoryLSTM(nn.Module):
    #  send in a tensor (batch_size, 20, 9)
    def __init__(self):

        super().__init__()
     
        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        # 128 128 two hlayers  64 10 fully connected layers

        self.norm = nn.LayerNorm(128)

        self.head = nn.Sequential(

            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64,10)

        )

    def forward(self,x):

        _,(hidden,_) = self.lstm(x)

        x = self.norm(hidden[-1])

        return self.head(x)


# ============================================================
# PREDICTOR
# ============================================================

class LSTMPredictor:

    def __init__(self,
                 model_path,
                 input_scaler_path,
                 target_scaler_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("[INFO] Loading LSTM...")

        self.model = DroneTrajectoryLSTM().to(self.device)

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.eval()

        self.input_scaler = joblib.load(input_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        # Buffer for each tracked drone
        self.history = defaultdict(
            lambda: deque(maxlen=20)
        )

        print("[INFO] LSTM Ready.")

    # --------------------------------------------------------

    def update(self, track_id, feature_dict):

        feature = [

            feature_dict["cx"],
            feature_dict["cy"],
            feature_dict["width"],
            feature_dict["height"],
            feature_dict["vx"],
            feature_dict["vy"],
            feature_dict["speed"],
            feature_dict["heading"],
            feature_dict["acceleration"]

        ]

        self.history[track_id].append(feature)

    # --------------------------------------------------------

    def ready(self, track_id):  # do we have enough frames to start prediction

        return len(self.history[track_id]) == 20 

    # --------------------------------------------------------

    def predict(self, track_id):

        if not self.ready(track_id):
            return None

        sequence = np.array(
            self.history[track_id],
            dtype=np.float32
        )
#        [
#  [cx1, cy1, w1, h1, vx1, vy1, speed1, heading1, acc1],
#  [cx2, cy2, w2, h2, vx2, vy2, speed2, heading2, acc2],
#  ...
#  [cx20,cy20,...]
# ]
        sequence = self.input_scaler.transform(
            sequence
        )

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        #(1,20,9)

        with torch.no_grad():

            pred = self.model(tensor)

        pred = pred.cpu().numpy()

        pred = self.target_scaler.inverse_transform(
            pred
        )[0]

        return pred.reshape(5,2)  #(1,10)