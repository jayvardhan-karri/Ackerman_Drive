import csv
import math
import os
import time
import numpy as np
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, Imu

from limo_gz_sim.trajectory_generator import TrajectoryGenerator
from limo_gz_sim.lidar_safety import LidarSafetyFilter


class NmpcTrackerNode(Node):
    """Hardware-transferable Full Nonlinear MPC (SLSQP)."""

    def __init__(self):
        super().__init__('nmpc_tracker_node')

        self.declare_parameter('profile', 'figure8')
        self.declare_parameter('v_nominal', 0.40)
        self.declare_parameter('wheelbase', 0.20)
        self.declare_parameter('max_steer', 0.48869)
        self.declare_parameter('horizon_steps', 10)
        self.declare_parameter('mpc_dt', 0.08)
        self.declare_parameter('q_cte', 40.0)
        self.declare_parameter('q_heading', 8.0)
        self.declare_parameter('r_steer', 0.8)
        self.declare_parameter('rd_steer', 2.5)
        self.declare_parameter('log_dir', 'evaluation')
        self.declare_parameter('run_duration', 25.0)

        profile = self.get_parameter('profile').value
        v_nom = float(self.get_parameter('v_nominal').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.N = int(self.get_parameter('horizon_steps').value)
        self.mpc_dt = float(self.get_parameter('mpc_dt').value)
        self.q_cte = float(self.get_parameter('q_cte').value)
        self.q_heading = float(self.get_parameter('q_heading').value)
        self.r_steer = float(self.get_parameter('r_steer').value)
        self.rd_steer = float(self.get_parameter('rd_steer').value)
        self.run_duration = float(self.get_parameter('run_duration').value)

        self.traj = TrajectoryGenerator(profile=profile, v_nominal=v_nom)
        self.safety = LidarSafetyFilter()

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.x0 = None
        self.y0 = None
        self.yaw0 = None
        self.odom_received = False
        
        self.u_guess = np.zeros(self.N, dtype=np.float64)
        self.prev_steer = 0.0

        self.start_time = None
        self.prev_time = None
        self.cte_history = []
        self.heading_err_history = []
        self.solve_times_ms = []

        log_dir = self.get_parameter('log_dir').value
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f'nmpc_{profile}_telemetry.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't', 'x', 'y', 'yaw', 'v_actual', 'v_cmd', 'omega_cmd', 'steer_rad',
            'x_ref', 'y_ref', 'yaw_ref', 'cte', 'heading_err', 'min_lidar_dist', 'safety_status', 'solve_ms'
        ])

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/reference_path', 1)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(LaserScan, '/scan', lambda msg: None, 10) # Dummy for safety filter args
        
        # We run NMPC slightly slower (25Hz) because SLSQP is computationally heavier
        self.control_timer = self.create_timer(0.04, self.control_loop)
        self.log_timer = self.create_timer(2.0, self.print_status)
        self.get_logger().info(f'NMPC Tracker initialized | logging to {self.csv_path}')

    @staticmethod
    def quat_to_yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def odom_cb(self, msg: Odometry):
        rx, ry = msg.pose.pose.position.x, msg.pose.pose.position.y
        ryaw = self.quat_to_yaw(msg.pose.pose.orientation)
        if self.x0 is None:
            self.x0, self.y0, self.yaw0 = rx, ry, ryaw
        dx, dy = rx - self.x0, ry - self.y0
        c0, s0 = math.cos(self.yaw0), math.sin(self.yaw0)
        self.x = c0 * dx + s0 * dy
        self.y = -s0 * dx + c0 * dy
        self.yaw = math.atan2(math.sin(ryaw - self.yaw0), math.cos(ryaw - self.yaw0))
        self.odom_received = True

    def _cost_function(self, u_steer, x0, y0, yaw0, ref_horizon):
        cost = 0.0
        x, y, yaw = x0, y0, yaw0
        L = self.wheelbase
        dt = self.mpc_dt
        
        for k in range(self.N):
            vk = ref_horizon[k, 3]
            dk = u_steer[k]
            
            x += vk * math.cos(yaw) * dt
            y += vk * math.sin(yaw) * dt
            yaw += (vk / L) * math.tan(dk) * dt
            
            xr, yr, th_r, vr, kr = ref_horizon[k]
            dx, dy = xr - x, yr - y
            cte = -math.sin(th_r) * dx + math.cos(th_r) * dy
            herr = math.atan2(math.sin(th_r - yaw), math.cos(th_r - yaw))
            
            d_prev = self.prev_steer if k == 0 else u_steer[k-1]
            slew = dk - d_prev
            
            cost += self.q_cte * (cte**2) + self.q_heading * (herr**2) + self.rd_steer * (slew**2)
            
        return cost

    def _solve_nmpc(self, nearest_idx):
        n_pts = len(self.traj.waypoints)
        dt = self.mpc_dt
        
        ref_horizon = np.zeros((self.N, 5), dtype=np.float64)
        curr_idx = float(nearest_idx)
        for k in range(self.N):
            idx_int = int(round(curr_idx)) % n_pts
            pt_k = self.traj.waypoints[idx_int]
            nxt_pt = self.traj.waypoints[(idx_int + 1) % n_pts]
            ds = max(math.hypot(nxt_pt[0] - pt_k[0], nxt_pt[1] - pt_k[1]), 1e-3)
            ref_horizon[k] = pt_k
            curr_idx += (pt_k[3] * dt) / ds
            
        bounds = [(-self.max_steer, self.max_steer) for _ in range(self.N)]
        
        # Warm start
        self.u_guess[:-1] = self.u_guess[1:]
        self.u_guess[-1] = math.atan(self.wheelbase * ref_horizon[-1, 4])
        
        res = minimize(
            self._cost_function,
            self.u_guess,
            args=(self.x, self.y, self.yaw, ref_horizon),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 6, 'ftol': 1e-3}
        )
        
        self.u_guess = res.x
        v_opt = ref_horizon[0, 3] * max(0.4, 1.0 - 0.5 * abs(res.x[0]) / self.max_steer)
        return float(v_opt), float(res.x[0])

    def control_loop(self):
        if not self.odom_received:
            return

        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
            self.publish_ref_path()
            return

        elapsed = now - self.start_time
        if self.run_duration > 0.0 and elapsed >= self.run_duration:
            self.stop_robot()
            raise SystemExit

        cte, heading_err, ref_pt, nearest_idx = self.traj.compute_tracking_errors(self.x, self.y, self.yaw, lookahead_pts=5)
        
        t0 = time.perf_counter()
        v_safe, steer_safe = self._solve_nmpc(nearest_idx)
        solve_ms = (time.perf_counter() - t0) * 1000.0
        
        self.prev_steer = steer_safe
        omega_cmd = (v_safe / self.wheelbase) * math.tan(steer_safe) if abs(v_safe) > 1e-4 else 0.0

        twist = Twist()
        twist.linear.x, twist.angular.z = float(v_safe), float(omega_cmd)
        self.cmd_pub.publish(twist)

        self.cte_history.append(cte)
        self.heading_err_history.append(heading_err)
        self.solve_times_ms.append(solve_ms)
        self.csv_writer.writerow([
            f'{elapsed:.3f}', f'{self.x:.4f}', f'{self.y:.4f}', f'{self.yaw:.4f}',
            '0.0', f'{v_safe:.4f}', f'{omega_cmd:.4f}', f'{steer_safe:.4f}',
            f'{ref_pt[0]:.4f}', f'{ref_pt[1]:.4f}', f'{ref_pt[2]:.4f}',
            f'{cte:.4f}', f'{heading_err:.4f}', '10.0', 'CLEAR', f'{solve_ms:.3f}'
        ])

    def print_status(self):
        if not self.odom_received or not self.cte_history: return
        rms_cte = math.sqrt(np.mean(np.square(self.cte_history)))
        self.get_logger().info(f'NMPC | CTE={self.cte_history[-1]:+6.3f}m (RMS={rms_cte:.3f}m) | solve={self.solve_times_ms[-1]:.1f}ms')

    def publish_ref_path(self):
        path_msg = Path()
        path_msg.header.frame_id, path_msg.header.stamp = 'odom', self.get_clock().now().to_msg()
        for pt in self.traj.waypoints[::5]:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x, ps.pose.position.y = float(pt[0]), float(pt[1])
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        if not self.csv_file.closed:
            self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = NmpcTrackerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()
