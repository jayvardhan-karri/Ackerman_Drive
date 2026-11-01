import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    limo_desc_share = FindPackageShare('limo_description').find('limo_description')
    limo_gz_share = FindPackageShare('limo_gz_sim').find('limo_gz_sim')
    model_path = os.path.abspath(os.path.join(limo_desc_share, '..'))
    world_file = os.path.join(limo_gz_share, 'worlds', 'arena.sdf')

    set_ign_env = SetEnvironmentVariable(name='IGN_GAZEBO_RESOURCE_PATH', value=model_path)
    set_gz_env = SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=model_path)

    ros_gz_sim_pkg = FindPackageShare('ros_gz_sim')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': f'-r "{world_file}"'}.items()
    )

    limo_description_pkg = FindPackageShare('limo_description')
    xacro_file = PathJoinSubstitution([limo_description_pkg, 'urdf', 'limo_ackerman.xacro'])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'robot_description': ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
        }]
    )

    spawn_robot = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'limo',
                    '-topic', 'robot_description',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.15'
                ],
                output='screen',
                emulate_tty=True
            )
        ]
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/model/limo/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/model/limo/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/limo/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/limo/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
            '/limo/depth_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/limo/depth_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/limo/depth_camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked'
        ],
        remappings=[
            ('/model/limo/odometry', '/odom'),
            ('/model/limo/tf', '/tf'),
            ('/limo/scan', '/scan'),
            ('/limo/imu', '/imu')
        ],
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        set_ign_env,
        set_gz_env,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        gz_bridge
    ])
