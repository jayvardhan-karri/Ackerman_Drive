import csv
import math
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, Imu

from limo_gz_sim.trajectory_generator import TrajectoryGenerator
from limo_gz_sim.lidar_safety import LidarSafetyFilter


class MpcTrackerNode(Node):
    """
    Hardware-transferable LTV Riccati Kinematic Error-State MPC Tracker.
    Solves the finite-horizon discrete Riccati recursion on the 4D augmented Frenet
    state x_e = [e_cte, d(e_cte)/dt, e_heading, u_{k-1}]^T at 50 Hz (< 0.5 ms).
    """

    def __init__(self):
        super().__init__('mpc_tracker_node')

        self.declare_parameter('profile', 'figure8')
        self.declare_parameter('v_nominal', 0.40)
        self.declare_parameter('wheelbase', 0.20)
        self.declare_parameter('max_steer', 0.48869)
        self.declare_parameter('horizon_steps', 16)
        self.declare_parameter('mpc_dt', 0.04)
        self.declare_parameter('q_cte', 520.0)
        self.declare_parameter('q_dcte', 35.0)
        self.declare_parameter('q_heading', 42.0)
        self.declare_parameter('r_steer', 0.28)
        self.declare_parameter('rd_steer', 0.45)
        self.declare_parameter('lookahead_pts', 9)
        self.declare_parameter('log_dir', 'evaluation')
        self.declare_parameter('run_duration', 25.0)

        profile = self.get_parameter('profile').value
        v_nom = float(self.get_parameter('v_nominal').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.N = int(self.get_parameter('horizon_steps').value)
        self.mpc_dt = float(self.get_parameter('mpc_dt').value)
        self.q_cte = float(self.get_parameter('q_cte').value)
        self.q_dcte = float(self.get_parameter('q_dcte').value)
        self.q_heading = float(self.get_parameter('q_heading').value)
        self.r_steer = float(self.get_parameter('r_steer').value)
        self.rd_steer = float(self.get_parameter('rd_steer').value)
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

        self.prev_steer = 0.0
        self.prev_cte = 0.0
        self.int_cte = 0.0
        self.prev_time = None
        self.start_time = None
        self.cte_history = []
        self.heading_err_history = []
        self.solve_times_ms = []

        log_dir = self.get_parameter('log_dir').value
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f'mpc_{profile}_telemetry.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't', 'x', 'y', 'yaw', 'v_actual', 'v_cmd', 'omega_cmd', 'steer_rad',
            'x_ref', 'y_ref', 'yaw_ref', 'cte', 'heading_err', 'min_lidar_dist', 'safety_status', 'solve_ms'
        ])

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/reference_path', 1)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Imu, '/imu', self.imu_cb, 20)

        self.control_timer = self.create_timer(0.02, self.control_loop)
        self.log_timer = self.create_timer(2.0, self.print_status)

        self.get_logger().info(
            f'LTV-Riccati MPC Tracker initialized | profile={profile} | N={self.N}, dt={self.mpc_dt}s | logging to {self.csv_path}'
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

    def _solve_ltv_mpc(self, cte: float, d_cte: float, heading_err: float, nearest_idx: int):
        """
        Solves the finite-horizon LTV-MPC problem over N steps using backward Riccati
        recursion on the 4-state Frenet error model:
          z_k = [e_cte, d(e_cte)/dt, e_heading, du_{k-1}]^T
        """
        n_pts = len(self.traj.waypoints)
        dt = self.mpc_dt
        L = self.wheelbase

        # 1. Sample time-varying reference horizon (v_k, kappa_k) along trajectory
        ref_horizon = np.zeros((self.N, 5), dtype=np.float64)
        curr_idx = float(nearest_idx)
        for k in range(self.N):
            idx_int = int(round(curr_idx)) % n_pts
            pt_k = self.traj.waypoints[idx_int]
            nxt_pt = self.traj.waypoints[(idx_int + 1) % n_pts]
            ds = max(math.hypot(nxt_pt[0] - pt_k[0], nxt_pt[1] - pt_k[1]), 1e-3)
            ref_horizon[k] = pt_k
            curr_idx += (pt_k[3] * dt) / ds

        # 2. Backward Riccati recursion on z_k = [cte, d_cte, heading_err, prev_du]^T
        Q = np.diag([self.q_cte, self.q_dcte, self.q_heading, self.rd_steer])
        P = Q * 3.0

        K_seq = np.zeros((self.N, 4), dtype=np.float64)
        ff_steers = np.zeros(self.N, dtype=np.float64)

        for k in range(self.N - 1, -1, -1):
            vk = max(float(ref_horizon[k, 3]), 0.15)
            kk = float(ref_horizon[k, 4])
            ff_steers[k] = math.atan(L * kk)
            sec2 = 1.0 + (L * kk) ** 2
            b_yaw = (vk * dt * sec2) / L

            A = np.array([
                [1.0, 0.35 * dt, vk * dt, 0.0],
                [0.0, 0.20,      vk,      0.0],
                [0.0, 0.0,       1.0,     0.0],
                [0.0, 0.0,       0.0,     0.0]
            ], dtype=np.float64)
            B = np.array([
                [-0.5 * vk * dt * b_yaw],
                [-vk * b_yaw / dt],
                [-b_yaw],
                [1.0]
            ], dtype=np.float64)

            R_eff = self.r_steer + self.rd_steer
            S = float(R_eff + (B.T @ P @ B)[0, 0])
            K_k = (B.T @ P @ A) / S
            K_seq[k] = K_k[0]
            P = Q + A.T @ P @ (A - B @ K_k)

        # 3. Preview feedforward over upcoming horizon + optimal Riccati state feedback K_0
        weights = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
        d_ff_preview = float(np.dot(weights, ff_steers[:5]))

        prev_du = self.prev_steer - d_ff_preview
        state_vec = np.array([cte, d_cte, heading_err, prev_du], dtype=np.float64)
        du_opt = -float(np.dot(K_seq[0], state_vec))

        steer_opt = d_ff_preview + du_opt + 0.06 * self.int_cte
        steer_opt = float(np.clip(steer_opt, -self.max_steer, self.max_steer))

        # Match speed profile with PID for exact apples-to-apples speed comparison
        v_ref_0 = float(ref_horizon[0, 3])
        v_opt = float(v_ref_0 * max(0.4, 1.0 - 0.5 * abs(steer_opt) / self.max_steer))

        return v_opt, steer_opt

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
        self.prev_time = now
        elapsed = now - self.start_time

        if self.run_duration > 0.0 and elapsed >= self.run_duration:
            self.stop_robot()
            rms_cte = math.sqrt(np.mean(np.square(self.cte_history))) if self.cte_history else 0.0
            max_cte = float(np.max(np.abs(self.cte_history))) if self.cte_history else 0.0
            rms_head = math.sqrt(np.mean(np.square(self.heading_err_history))) if self.heading_err_history else 0.0
            mean_solve = float(np.mean(self.solve_times_ms)) if self.solve_times_ms else 0.0
            self.get_logger().info(
                f'=== MPC RUN COMPLETE ({elapsed:.1f}s) | RMS CTE: {rms_cte:.4f}m | Max CTE: {max_cte:.4f}m | RMS Heading Err: {rms_head:.4f}rad | Avg Solve: {mean_solve:.2f}ms ==='
            )
            raise SystemExit

        cte, heading_err, ref_pt, nearest_idx = self.traj.compute_tracking_errors(
            self.x, self.y, self.yaw, lookahead_pts=self.lookahead_pts
        )
        x_ref, y_ref, yaw_ref, v_ref, kappa_ref = ref_pt

        d_cte = (cte - self.prev_cte) / dt
        self.prev_cte = cte
        self.int_cte = float(np.clip(self.int_cte + cte * dt, -0.5, 0.5))

        t0 = time.perf_counter()
        v_mpc, steer_mpc = self._solve_ltv_mpc(cte, d_cte, heading_err, nearest_idx)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        v_safe, steer_safe, min_dist, status = self.safety.filter_command(
            v_mpc, steer_mpc, self.scan_ranges, self.scan_angle_min, self.scan_angle_inc
        )
        steer_safe = float(np.clip(steer_safe, -self.max_steer, self.max_steer))
        self.prev_steer = steer_safe

        omega_cmd = (v_safe / self.wheelbase) * math.tan(steer_safe) if abs(v_safe) > 1e-4 else 0.0

        twist = Twist()
        twist.linear.x = float(v_safe)
        twist.angular.z = float(omega_cmd)
        self.cmd_pub.publish(twist)

        self.cte_history.append(cte)
        self.heading_err_history.append(heading_err)
        self.solve_times_ms.append(solve_ms)
        self.csv_writer.writerow([
            f'{elapsed:.3f}', f'{self.x:.4f}', f'{self.y:.4f}', f'{self.yaw:.4f}',
            f'{self.v_actual:.4f}', f'{v_safe:.4f}', f'{omega_cmd:.4f}', f'{steer_safe:.4f}',
            f'{x_ref:.4f}', f'{y_ref:.4f}', f'{yaw_ref:.4f}',
            f'{cte:.4f}', f'{heading_err:.4f}', f'{min_dist:.3f}', status, f'{solve_ms:.3f}'
        ])

    def print_status(self):
        if not self.odom_received or not self.cte_history:
            return
        elapsed = time.monotonic() - self.start_time if self.start_time else 0.0
        cte = self.cte_history[-1]
        herr = self.heading_err_history[-1]
        rms_cte = math.sqrt(np.mean(np.square(self.cte_history)))
        solve_ms = self.solve_times_ms[-1] if self.solve_times_ms else 0.0
        self.get_logger().info(
            f't={elapsed:4.1f}s | pos=({self.x:5.2f}, {self.y:5.2f}) | CTE={cte:+6.3f}m (RMS={rms_cte:.3f}m) | HeadErr={herr:+5.2f}rad | solve={solve_ms:.2f}ms'
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
    node = MpcTrackerNode()
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
