import math
import numpy as np


class TrajectoryGenerator:
    """Analytical reference trajectory generator and Frenet error calculator."""

    def __init__(self, profile: str = 'figure8', v_nominal: float = 0.40, num_points: int = 1200):
        self.profile = profile
        self.v_nominal = v_nominal
        self.num_points = num_points
        self.waypoints = self._generate(profile, v_nominal, num_points)
        self.last_idx = 0

    def _generate(self, profile: str, v_nom: float, n: int) -> np.ndarray:
        """Returns array of shape (N, 5): [x, y, yaw, v_ref, curvature]."""
        t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)

        if profile == 'circle':
            r = 1.5
            x = r * np.sin(t)
            y = r * (1.0 - np.cos(t))
            dx = r * np.cos(t)
            dy = r * np.sin(t)
            ddx = -r * np.sin(t)
            ddy = r * np.cos(t)

        elif profile == 'oval_slalom':
            a, b = 3.5, 2.8
            x = a * np.sin(t)
            y = b * (1.0 - np.cos(t)) - b
            dx = a * np.cos(t)
            dy = b * np.sin(t)
            ddx = -a * np.sin(t)
            ddy = b * np.cos(t)

        else:  # 'figure8' cleanly inside the (+/-2.5, +/-1.5) pillars
            a = 2.05
            b = 0.90
            x = a * np.sin(t)
            y = b * np.sin(2.0 * t)
            dx = a * np.cos(t)
            dy = 2.0 * b * np.cos(2.0 * t)
            ddx = -a * np.sin(t)
            ddy = -4.0 * b * np.sin(2.0 * t)

        yaw = np.arctan2(dy, dx)
        denom = np.power(dx * dx + dy * dy, 1.5) + 1e-9
        curvature = (dx * ddy - dy * ddx) / denom

        # Smooth curvature speed regulation so steering never saturates
        v_ref = v_nom / (1.0 + 0.75 * np.abs(curvature))
        return np.column_stack((x, y, yaw, v_ref, curvature))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def compute_tracking_errors(self, x: float, y: float, yaw: float, lookahead_pts: int = 8):
        n = len(self.waypoints)
        search_window = 160
        indices = (np.arange(self.last_idx - 25, self.last_idx + search_window)) % n
        pts = self.waypoints[indices, :2]
        dists_sq = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        best_local = int(np.argmin(dists_sq))
        nearest_idx = int(indices[best_local])

        if dists_sq[best_local] > 2.25:
            all_dists = (self.waypoints[:, 0] - x) ** 2 + (self.waypoints[:, 1] - y) ** 2
            nearest_idx = int(np.argmin(all_dists))

        self.last_idx = nearest_idx
        target_idx = (nearest_idx + lookahead_pts) % n

        x_r, y_r, yaw_r, v_r, kappa_r = self.waypoints[nearest_idx]
        x_lh, y_lh, yaw_lh, v_lh, kappa_lh = self.waypoints[target_idx]

        dx = x_r - x
        dy = y_r - y

        cte = -math.sin(yaw_r) * dx + math.cos(yaw_r) * dy
        heading_err = self.normalize_angle(yaw_lh - yaw)

        return cte, heading_err, (x_lh, y_lh, yaw_lh, v_lh, kappa_lh), nearest_idx
