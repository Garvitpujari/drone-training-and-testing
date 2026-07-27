import math

class ThreatAssessment:
    def assess(self, avg_speed, max_speed, avg_acceleration,
               max_acceleration, prediction_count, total_frames):
        score=0
        reasons=[]

        if avg_speed>=6:
            score+=20; reasons.append("High average drone speed.")
        elif avg_speed>=3:
            score+=10; reasons.append("Moderate average drone speed.")

        if max_speed>=80:
            score+=25; reasons.append("Very high peak speed.")
        elif max_speed>=30:
            score+=15; reasons.append("High peak speed.")
        elif max_speed>=10:
            score+=5; reasons.append("Moderate peak speed.")

        if abs(avg_acceleration)>=2:
            score+=10; reasons.append("Continuous acceleration detected.")

        if abs(max_acceleration)>=80:
            score+=20; reasons.append("Aggressive maneuver detected.")
        elif abs(max_acceleration)>=20:
            score+=10; reasons.append("Rapid maneuver detected.")

        ratio=prediction_count/max(total_frames,1)
        if ratio>0.90:
            score+=15; reasons.append("Drone tracked during most of the video.")
        elif ratio>0.60:
            score+=8; reasons.append("Drone tracked for a significant duration.")

        score=min(score,100)

        if score>=70:
            level="HIGH"
        elif score>=45:
            level="MEDIUM"
        elif score>=20:
            level="LOW"
        else:
            level="NO THREAT"

        return {"score":score,"level":level,"reasons":reasons}
