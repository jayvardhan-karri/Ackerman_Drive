import csv
import math
import os
import time
import numpy as np
import scipy.linalg

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, Imu

from limo_gz_sim.trajectory_generator import TrajectoryGenerator
from limo_gz_sim.lidar_safety import LidarSafetyFilter


class LqrTrackerNode(Node):
    """Hardware-transferable Kinematic LQR Trajectory Tracker (Algebraic Riccati Equation)."""

    def __init__(self):
        super().__init__('lqr_tracker_node')

        self.declare_parameter('profile', 'figure8')
        self.declare_parameter('v_nominal', 0.40)
        self.declare_parameter('wheelbase', 0.20)
        self.declare_parameter('max_steer', 0.48869)
        self.declare_parameter('q_cte', 450.0)
        self.declare_parameter('q_heading', 35.0)
        self.declare_parameter('r_steer', 0.5)
        self.declare_parameter('lookahead_pts', 8)
        self.declare_parameter('log_dir', 'evaluation')
        self.declare_parameter('run_duration', 25.0)

        profile = self.get_parameter('profile').value
        v_nom = float(self.get_parameter('v_nominal').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.q_cte = float(self.get_parameter('q_cte').value)
        self.q_heading = float(self.get_parameter('q_heading').value)
        self.r_steer = float(self.get_parameter('r_steer').value)
        self.lookahead_pts = int(self.get_parameter('lookahead_pts').value)
        self.run_duration = float(self.get_parameter('run_duration').value)

        self.traj = TrajectoryGenerator(profile=profile, v_nominal=v_nom)
        self.safety = LidarSafetyFilter()

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.x0 = None
        self.y0 = None
        self.yaw0 = None
        self.v_actual = 0.0
        self.odom_received = False

        self.scan_ranges = None
        self.scan_angle_min = -2.094395
        self.scan_angle_inc = 0.005825

        self.prev_time = None
        self.start_time = None
        self.cte_history = []
        self.heading_err_history = []

        log_dir = self.get_parameter('log_dir').value
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f'lqr_{profile}_telemetry.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't', 'x', 'y', 'yaw', 'v_actual', 'v_cmd', 'omega_cmd', 'steer_rad',
            'x_ref', 'y_ref', 'yaw_ref', 'cte', 'heading_err', 'min_lidar_dist', 'safety_status'
        ])

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/reference_path', 1)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Imu, '/imu', self.imu_cb, 20)

        self.control_timer = self.create_timer(0.02, self.control_loop)
        self.log_timer = self.create_timer(2.0, self.print_status)

        self.get_logger().info(
            f'LQR Tracker initialized | profile={profile} | logging to {self.csv_path}'
        )

    @staticmethod
    def quat_to_yaw(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def odom_cb(self, msg: Odometry):
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_yaw = self.quat_to_yaw(msg.pose.pose.orientation)
        if self.x0 is None:
            self.x0, self.y0, self.yaw0 = raw_x, raw_y, raw_yaw
        dx = raw_x - self.x0
        dy = raw_y - self.y0
        c0, s0 = math.cos(self.yaw0), math.sin(self.yaw0)
        self.x = c0 * dx + s0 * dy
        self.y = -s0 * dx + c0 * dy
        self.yaw = math.atan2(math.sin(raw_yaw - self.yaw0), math.cos(raw_yaw - self.yaw0))
        self.v_actual = msg.twist.twist.linear.x
        self.odom_received = True

    def scan_cb(self, msg: LaserScan):
        self.scan_ranges = np.array(msg.ranges, dtype=np.float32)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment

    def imu_cb(self, msg: Imu):
        pass

    def _solve_lqr_gain(self, v: float, kappa: float):
        """
        Computes LQR gain K = [k_cte, k_heading] using continuous ARE for error state:
          e_dot = A e + B u
        """
        L = self.wheelbase
        sec2 = 1.0 + (L * kappa) ** 2

        # State: [e_cte, e_heading]^T, Input: steering delta
        A = np.array([
            [0.0, v],
            [0.0, 0.0]
        ], dtype=np.float64)
        B = np.array([
            [0.0],
            [v * sec2 / L]
        ], dtype=np.float64)

        Q = np.diag([self.q_cte, self.q_heading])
        R = np.array([[self.r_steer]], dtype=np.float64)

        try:
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ (B.T @ P)
            return K[0]  # [k_cte, k_heading]
        except Exception:
            return np.array([2.5, 1.2])

    def control_loop(self):
        if not self.odom_received:
            return

        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
            self.prev_time = now
            self.publish_ref_path()
            return

        elapsed = now - self.start_time
        self.prev_time = now

        if self.run_duration > 0.0 and elapsed >= self.run_duration:
            self.stop_robot()
            rms_cte = math.sqrt(np.mean(np.square(self.cte_history))) if self.cte_history else 0.0
            max_cte = float(np.max(np.abs(self.cte_history))) if self.cte_history else 0.0
            rms_head = math.sqrt(np.mean(np.square(self.heading_err_history))) if self.heading_err_history else 0.0
            self.get_logger().info(
                f'=== LQR RUN COMPLETE ({elapsed:.1f}s) | RMS CTE: {rms_cte:.4f}m | Max CTE: {max_cte:.4f}m | RMS Heading Err: {rms_head:.4f}rad ==='
            )
            raise SystemExit

        cte, heading_err, ref_pt, nearest_idx = self.traj.compute_tracking_errors(
            self.x, self.y, self.yaw, lookahead_pts=self.lookahead_pts
        )
        x_ref, y_ref, yaw_ref, v_ref, kappa_ref = ref_pt

        # Solve LQR feedback gain
        K = self.solve_lqr_gain_cached(v_ref, kappa_ref)
        ff_steer = math.atan(self.wheelbase * kappa_ref)
        lqr_steer = ff_steer + (K[0] * cte + K[1] * heading_err)
        steer_cmd = float(np.clip(lqr_steer, -self.max_steer, self.max_steer))

        v_cmd = v_ref * max(0.4, 1.0 - 0.5 * abs(steer_cmd) / self.max_steer)

        v_safe, steer_safe, min_dist, status = self.safety.filter_command(
            v_cmd, steer_cmd, self.scan_ranges, self.scan_angle_min, self.scan_angle_inc
        )
        steer_safe = float(np.clip(steer_safe, -self.max_steer, self.max_steer))

        omega_cmd = (v_safe / self.wheelbase) * math.tan(steer_safe) if abs(v_safe) > 1e-4 else 0.0

        twist = Twist()
        twist.linear.x = float(v_safe)
        twist.angular.z = float(omega_cmd)
        self.cmd_pub.publish(twist)

        self.cte_history.append(cte)
        self.heading_err_history.append(heading_err)
        self.csv_writer.writerow([
            f'{elapsed:.3f}', f'{self.x:.4f}', f'{self.y:.4f}', f'{self.yaw:.4f}',
            f'{self.v_actual:.4f}', f'{v_safe:.4f}', f'{omega_cmd:.4f}', f'{steer_safe:.4f}',
            f'{x_ref:.4f}', f'{y_ref:.4f}', f'{yaw_ref:.4f}',
            f'{cte:.4f}', f'{heading_err:.4f}', f'{min_dist:.3f}', status
        ])

    def solve_lqr_gain_cached(self, v, kappa):
        return self._solve_lqr_gain(v, kappa)

    def print_status(self):
        if not self.odom_received or not self.cte_history:
            return
        elapsed = time.monotonic() - self.start_time if self.start_time else 0.0
        cte = self.cte_history[-1]
        herr = self.heading_err_history[-1]
        rms_cte = math.sqrt(np.mean(np.square(self.cte_history)))
        self.get_logger().info(
            f't={elapsed:4.1f}s | pos=({self.x:5.2f}, {self.y:5.2f}) | CTE={cte:+6.3f}m (RMS={rms_cte:.3f}m) | HeadErr={herr:+5.2f}rad'
        )

    def publish_ref_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for pt in self.traj.waypoints[::5]:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(pt[0])
            ps.pose.position.y = float(pt[1])
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = LqrTrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
