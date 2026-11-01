import math
import numpy as np


class LidarSafetyFilter:
    """Hardware-agnostic LiDAR safety velocity scaler and repulsive steering assist."""

    def __init__(
        self,
        stop_dist: float = 0.28,
        caution_dist: float = 0.85,
        repulse_dist: float = 0.65,
        repulse_gain: float = 0.35,
    ):
        self.stop_dist = stop_dist
        self.caution_dist = caution_dist
        self.repulse_dist = repulse_dist
        self.repulse_gain = repulse_gain

    def filter_command(
        self,
        v_cmd: float,
        steer_cmd: float,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ):
        """
        Adjusts (v_cmd, steer_cmd) based on real-time LaserScan ranges.
        Returns: (v_safe, steer_safe, min_front_dist, status_str)
        """
        if ranges is None or len(ranges) == 0:
            return v_cmd, steer_cmd, float('inf'), 'NO_SCAN'

        n = len(ranges)
        angles = angle_min + np.arange(n) * angle_increment

        # Replace NaNs/Infs and ignore self-hits < 0.08m
        clean_r = np.where(np.isfinite(ranges) & (ranges > 0.08), ranges, 10.0)

        # 1. Forward corridor centered slightly toward current steering angle
        corridor_center = 0.5 * steer_cmd
        front_mask = np.abs(angles - corridor_center) <= math.radians(32.0)
        min_front = float(np.min(clean_r[front_mask])) if np.any(front_mask) else 10.0

        # 2. Lateral repulsive field (left: [10 deg, 75 deg], right: [-75 deg, -10 deg])
        left_mask = (angles >= math.radians(10.0)) & (angles <= math.radians(75.0))
        right_mask = (angles <= -math.radians(10.0)) & (angles >= -math.radians(75.0))

        min_left = float(np.min(clean_r[left_mask])) if np.any(left_mask) else 10.0
        min_right = float(np.min(clean_r[right_mask])) if np.any(right_mask) else 10.0

        steer_bias = 0.0
        if min_left < self.repulse_dist:
            # Obstacle on left -> push steering negative (right)
            steer_bias -= self.repulse_gain * (1.0 / max(min_left, 0.15) - 1.0 / self.repulse_dist)
        if min_right < self.repulse_dist:
            # Obstacle on right -> push steering positive (left)
            steer_bias += self.repulse_gain * (1.0 / max(min_right, 0.15) - 1.0 / self.repulse_dist)

        steer_safe = steer_cmd + steer_bias

        # 3. Velocity scaling
        if min_front <= self.stop_dist:
            return 0.0, steer_safe, min_front, 'EMERGENCY_STOP'
        elif min_front < self.caution_dist:
            scale = (min_front - self.stop_dist) / (self.caution_dist - self.stop_dist)
            v_safe = v_cmd * max(0.15, scale)
            return v_safe, steer_safe, min_front, 'CAUTION'

        return v_cmd, steer_safe, min_front, 'CLEAR'
