import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import math
import threading
import numpy as np

class LivePlotterNode(Node):
    """Off-board ROS 2 Matplotlib Live Plotter."""
    def __init__(self):
        super().__init__('live_plotter_node')
        
        # State buffers
        self.ref_x = []
        self.ref_y = []
        self.trail_x = []
        self.trail_y = []
        
        self.x0 = None
        self.y0 = None
        self.yaw0 = None
        
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.odom_received = False
        
        self.create_subscription(Path, '/reference_path', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.get_logger().info("Live Plotter Node Started. Waiting for data...")

    @staticmethod
    def quat_to_yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def path_cb(self, msg: Path):
        self.ref_x = [p.pose.position.x for p in msg.poses]
        self.ref_y = [p.pose.position.y for p in msg.poses]
        if self.ref_x:
            # Close the loop visually
            self.ref_x.append(msg.poses[0].pose.position.x)
            self.ref_y.append(msg.poses[0].pose.position.y)
        
        # Reset trail when a new path is published (new run started)
        self.trail_x.clear()
        self.trail_y.clear()
        self.x0 = None  # Force re-zeroing of odometry for the new run

    def odom_cb(self, msg: Odometry):
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_yaw = self.quat_to_yaw(msg.pose.pose.orientation)
        
        if self.x0 is None:
            self.x0, self.y0, self.yaw0 = raw_x, raw_y, raw_yaw
            
        dx = raw_x - self.x0
        dy = raw_y - self.y0
        c0, s0 = math.cos(self.yaw0), math.sin(self.yaw0)
        
        self.curr_x = c0 * dx + s0 * dy
        self.curr_y = -s0 * dx + c0 * dy
        self.curr_yaw = math.atan2(math.sin(raw_yaw - self.yaw0), math.cos(raw_yaw - self.yaw0))
        
        # Downsample trail for performance (only add point if moved > 2cm)
        if not self.trail_x or math.hypot(self.curr_x - self.trail_x[-1], self.curr_y - self.trail_y[-1]) > 0.02:
            self.trail_x.append(self.curr_x)
            self.trail_y.append(self.curr_y)
            
        self.odom_received = True

def main(args=None):
    rclpy.init(args=args)
    node = LivePlotterNode()
    
    # Run rclpy.spin in a background thread so Matplotlib owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # --- Setup Matplotlib Figure ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_title("LIMO Trajectory Tracker (Live)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Draw Arena Walls
    wall_rect = patches.Rectangle((-5, -5), 10, 10, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(wall_rect)

    # Draw Pillars
    for px, py in [(2.5, 1.5), (2.5, -1.5), (-2.5, 1.5), (-2.5, -1.5)]:
        pillar = patches.Circle((px, py), radius=0.15, color='gray', zorder=3)
        ax.add_patch(pillar)

    # Plot lines
    ref_line, = ax.plot([], [], 'g--', lw=2, label="Reference Path")
    trail_line, = ax.plot([], [], 'r-', lw=2, label="Live Trail")
    
    # Robot representation (triangle / arrow)
    robot_marker = patches.RegularPolygon((0, 0), numVertices=3, radius=0.2, orientation=0, color='blue', zorder=4, label="LIMO")
    ax.add_patch(robot_marker)
    
    ax.legend(loc="upper right")

    def update_plot(frame):
        # Update Reference Path
        if node.ref_x and len(ref_line.get_xdata()) != len(node.ref_x):
            ref_line.set_data(node.ref_x, node.ref_y)
            
        # Update Trail
        if node.trail_x:
            trail_line.set_data(node.trail_x, node.trail_y)
            
        # Update Robot Marker
        if node.odom_received:
            robot_marker.xy = (node.curr_x, node.curr_y)
            # RegularPolygon orientation is offset by pi/6 from the point natively
            robot_marker.orientation = node.curr_yaw - math.pi/6 
            
        return ref_line, trail_line, robot_marker

    # 25 Hz animation update
    ani = animation.FuncAnimation(fig, update_plot, interval=40, blit=True)
    
    try:
        plt.show()  # Blocks until the window is closed
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
