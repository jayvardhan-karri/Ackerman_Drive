import csv
import math
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan, Imu

from limo_gz_sim.trajectory_generator import TrajectoryGenerator
from limo_gz_sim.lidar_safety import LidarSafetyFilter


class PidTrackerNode(Node):
    """Hardware-transferable Kinematic Ackermann PID Trajectory Tracker."""

    def __init__(self):
        super().__init__('pid_tracker_node')

        # Declare configurable parameters (identical for sim and physical LIMO)
        self.declare_parameter('profile', 'figure8')
        self.declare_parameter('v_nominal', 0.40)
        self.declare_parameter('kp_cte', 1.8)
        self.declare_parameter('ki_cte', 0.05)
        self.declare_parameter('kd_cte', 0.45)
        self.declare_parameter('kp_heading', 1.15)
        self.declare_parameter('wheelbase', 0.20)
        self.declare_parameter('max_steer', 0.48869)  # 28 deg from limo_driver.h
        self.declare_parameter('lookahead_pts', 10)
        self.declare_parameter('log_dir', 'evaluation')
        self.declare_parameter('run_duration', 45.0)  # seconds (0.0 for infinite)

        profile = self.get_parameter('profile').value
        v_nom = float(self.get_parameter('v_nominal').value)
        self.kp_cte = float(self.get_parameter('kp_cte').value)
        self.ki_cte = float(self.get_parameter('ki_cte').value)
        self.kd_cte = float(self.get_parameter('kd_cte').value)
        self.kp_heading = float(self.get_parameter('kp_heading').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.lookahead_pts = int(self.get_parameter('lookahead_pts').value)
        self.run_duration = float(self.get_parameter('run_duration').value)

        self.traj = TrajectoryGenerator(profile=profile, v_nominal=v_nom)
        self.safety = LidarSafetyFilter()

        # State variables
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

        # PID memory
        self.integral_cte = 0.0
        self.prev_cte = 0.0
        self.prev_time = None
        self.start_time = None
        self.cte_history = []
        self.heading_err_history = []

        # Telemetry CSV setup
        log_dir = self.get_parameter('log_dir').value
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f'pid_{profile}_telemetry.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't', 'x', 'y', 'yaw', 'v_actual', 'v_cmd', 'omega_cmd', 'steer_rad',
            'x_ref', 'y_ref', 'yaw_ref', 'cte', 'heading_err', 'min_lidar_dist', 'safety_status'
        ])

        # Publishers & Subscribers (matched to limo_base hardware topics)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/reference_path', 1)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Imu, '/imu', self.imu_cb, 20)

        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz (matches limo_base)
        self.log_timer = self.create_timer(2.0, self.print_status)

        self.get_logger().info(
            f'PID Tracker initialized | profile={profile} | v_nom={v_nom} m/s | logging to {self.csv_path}'
        )

    @staticmethod
    def quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

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
        pass  # Available for dynamic yaw-rate damping if needed

    def control_loop(self):
        if not self.odom_received:
            return

        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
            self.prev_time = now
            self.publish_ref_path()
            return

        dt = max(now - self.prev_time, 1e-3)
        elapsed = now - self.start_time
        self.prev_time = now

        if self.run_duration > 0.0 and elapsed >= self.run_duration:
            self.stop_robot()
            rms_cte = math.sqrt(np.mean(np.square(self.cte_history))) if self.cte_history else 0.0
            max_cte = float(np.max(np.abs(self.cte_history))) if self.cte_history else 0.0
            rms_head = math.sqrt(np.mean(np.square(self.heading_err_history))) if self.heading_err_history else 0.0
            self.get_logger().info(
                f'=== RUN COMPLETE ({elapsed:.1f}s) | RMS CTE: {rms_cte:.4f}m | Max CTE: {max_cte:.4f}m | RMS Heading Err: {rms_head:.4f}rad ==='
            )
            raise SystemExit

        # 1. Compute trajectory errors
        cte, heading_err, ref_pt, idx = self.traj.compute_tracking_errors(
            self.x, self.y, self.yaw, lookahead_pts=self.lookahead_pts
        )
        x_ref, y_ref, yaw_ref, v_ref, kappa_ref = ref_pt

        # 2. PID on Cross-Track Error + P on Heading Error + Curvature Feedforward
        self.integral_cte = np.clip(self.integral_cte + cte * dt, -0.5, 0.5)
        d_cte = (cte - self.prev_cte) / dt
        self.prev_cte = cte

        ff_steer = math.atan(self.wheelbase * kappa_ref)
        pid_steer = (
            ff_steer
            + self.kp_cte * cte
            + self.ki_cte * self.integral_cte
            + self.kd_cte * d_cte
            + self.kp_heading * heading_err
        )
        steer_cmd = float(np.clip(pid_steer, -self.max_steer, self.max_steer))

        # Reduce speed smoothly when steering error is large
        v_cmd = v_ref * max(0.4, 1.0 - 0.5 * abs(steer_cmd) / self.max_steer)

        # 3. LiDAR Safety Filter
        v_safe, steer_safe, min_dist, status = self.safety.filter_command(
            v_cmd, steer_cmd, self.scan_ranges, self.scan_angle_min, self.scan_angle_inc
        )
        steer_safe = float(np.clip(steer_safe, -self.max_steer, self.max_steer))

        # 4. Convert (v_safe, steer_safe) to Ackermann Twist (omega = v * tan(delta) / L)
        omega_cmd = (v_safe / self.wheelbase) * math.tan(steer_safe) if abs(v_safe) > 1e-4 else 0.0

        twist = Twist()
        twist.linear.x = float(v_safe)
        twist.angular.z = float(omega_cmd)
        self.cmd_pub.publish(twist)

        # 5. Record metrics
        self.cte_history.append(cte)
        self.heading_err_history.append(heading_err)
        self.csv_writer.writerow([
            f'{elapsed:.3f}', f'{self.x:.4f}', f'{self.y:.4f}', f'{self.yaw:.4f}',
            f'{self.v_actual:.4f}', f'{v_safe:.4f}', f'{omega_cmd:.4f}', f'{steer_safe:.4f}',
            f'{x_ref:.4f}', f'{y_ref:.4f}', f'{yaw_ref:.4f}',
            f'{cte:.4f}', f'{heading_err:.4f}', f'{min_dist:.3f}', status
        ])

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
        self.csv_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = PidTrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if not node.csv_file.closed:
            node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
