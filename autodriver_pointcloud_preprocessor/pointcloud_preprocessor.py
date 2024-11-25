"""
1. Subscribe to pointcloud [done]
1i. Add QoS profiles [done]
2. Passthrough pointcloud and profile [done]
2. Publish transformed pointcloud [done]
3. Publish roi removed pointcloud [done]
4. Publish voxelized pointcloud [done]
3. Publish preprocessed pointcloud [done]

Todo:
    * add azimuth, distance and other Autoware fields
    * add onetime height estimation from my autodriving transformer code
"""

import time
import struct
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy import qos
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2
from sensor_msgs_py.point_cloud2 import read_points, create_cloud
from image_geometry import PinholeCameraModel
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros
from tf2_ros import TransformBroadcaster, TransformListener, Buffer, LookupException, ConnectivityException, \
    ExtrapolationException
import tf_transformations
import numpy as np
import transforms3d
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from cv_bridge import CvBridge
import cv2
import torch
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c


class PointcloudPreprocessorNode(Node):
    def __init__(self):
        super(PointcloudPreprocessorNode, self).__init__('pointcloud_preprocessor')

        # Declare parameters
        self.declare_parameter(name='input_topic', value="/camera/camera/depth/color/points",
                               descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='output_topic', value="/camera/camera/depth/color/points/processed",
                               descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('queue_size', 1)
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('cpu_backend', 'numpy')  # numpy or pytorch
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('static_camera_to_robot_tf', True)
        self.declare_parameter('transform_timeout', 0.1)

        self.declare_parameter('crop_to_roi', True)
        self.declare_parameter('roi_min', [-6.0, -6.0, -0.05])
        self.declare_parameter('roi_max', [6.0, 6.0, 2.0])
        self.declare_parameter('voxel_size', 0.01)  # 0.01
        self.declare_parameter('remove_statistical_outliers', False)
        self.declare_parameter('estimate_normals', False)
        self.declare_parameter('remove_ground', False)
        self.declare_parameter('ground_plane', [0.0, 1.0, 0.0, 0.0])
        self.declare_parameter('use_height', True)  # if true, remove the ground based on height

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.qos = self.get_parameter('qos').get_parameter_value().string_value
        self.queue_size = self.get_parameter('queue_size').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.cpu_backend = self.get_parameter('cpu_backend').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.static_camera_to_robot_tf = self.get_parameter('static_camera_to_robot_tf').value
        self.transform_timeout = self.get_parameter('transform_timeout').value
        self.crop_to_roi = self.get_parameter('crop_to_roi').value
        self.roi_min = self.get_parameter('roi_min').value
        self.roi_max = self.get_parameter('roi_max').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.remove_statistical_outliers = self.get_parameter('remove_statistical_outliers').value
        self.estimate_normals = self.get_parameter('estimate_normals').value
        self.remove_ground = self.get_parameter('remove_ground').value
        self.ground_plane = self.get_parameter('ground_plane').value
        self.use_height = self.get_parameter('use_height').value

        # Setup the device
        self.torch_device = torch.device('cpu')
        self.o3d_device = o3d.core.Device('CPU:0')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.torch_device = torch.device('cuda:0')
                if o3d.core.cuda.is_available():
                    self.o3d_device = o3d.core.Device('CUDA:0')

        # Initialize variables
        self.camera_to_robot_tf = None

        # TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)

        # Debugging parameters
        self.processing_times = {}

        # setup QoS
        qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=self.queue_size
        )
        if self.qos.lower() == "sensor_data":
            qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=self.queue_size
            )

        # Setup subscribers
        self.poincloud_sub = self.create_subscription(PointCloud2, self.input_topic,
                                                      self.callback, qos_profile=qos_profile)

        # Setup publishers
        self.pointcloud_pub = self.create_publisher(PointCloud2, self.output_topic, self.queue_size)

        # self.pointcloud_timer = self.create_timer(1 / self.odom_rate, self.rgbd_timer_callback)
        self.get_logger().info(f"pointcloud_preprocessor node started on device: {self.o3d_device}")


    def callback(self, ros_cloud):
        frame_id = ros_cloud.header.frame_id

        try:
            start_time = time.time()
            # Get field indices for faster access
            xyz_offset = [None, None, None]
            rgb_offset = None

            for idx, field in enumerate(ros_cloud.fields):
                if field.name == 'x':
                    xyz_offset[0] = idx
                elif field.name == 'y':
                    xyz_offset[1] = idx
                elif field.name == 'z':
                    xyz_offset[2] = idx
                elif field.name == 'rgb':
                    rgb_offset = idx

            if None in xyz_offset or rgb_offset is None:
                self.get_logger().error("Required point cloud fields not found")
                return

            # Convert ROS PointCloud2 to numpy arrays
            # https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs_py/sensor_msgs_py/point_cloud2.py
            # https://gist.github.com/SebastianGrans/6ae5cab66e453a14a859b66cd9579239?permalink_comment_id=4345802#gistcomment-4345802
            cloud_array = point_cloud2.read_points_numpy(
                    ros_cloud,
                    field_names=('x', 'y', 'z', 'rgb'),
                    skip_nans=True
            )

            self.processing_times['ros_to_numpy'] = time.time() - start_time

            # Extract XYZ points
            start_time = time.time()
            points_np = cloud_array[:, :3].astype(np.float32)

            # Extract and convert RGB values
            rgb_float = cloud_array[:, 3].copy()
            rgb_bytes = rgb_float.view(np.uint32)

            # Extract RGB channels
            r = ((rgb_bytes >> 16) & 0xFF).astype(np.float32) / 255.0
            g = ((rgb_bytes >> 8) & 0xFF).astype(np.float32) / 255.0
            b = (rgb_bytes & 0xFF).astype(np.float32) / 255.0

            # Stack RGB channels
            colors_np = np.vstack((r, g, b)).T
            self.processing_times['data_preparation'] = time.time() - start_time

            # Convert numpy arrays to tensors and move to device
            start_time = time.time()
            self.o3d_pointcloud.point.positions = o3d.core.Tensor(
                    points_np,
                    dtype=o3d.core.Dtype.Float32,
                    device=self.o3d_device
            )

            self.o3d_pointcloud.point.colors = o3d.core.Tensor(
                    colors_np,
                    dtype=o3d.core.Dtype.Float32,
                    device=self.o3d_device
            )
            self.processing_times['tensor_transfer'] = time.time() - start_time

            # transform to robot frame
            start_time = time.time()
            self.get_camera_to_robot_tf(frame_id, ros_cloud.header.stamp)
            self.processing_times['tf_lookup'] = time.time() - start_time

            if self.camera_to_robot_tf is not None:
                start_time = time.time()
                self.o3d_pointcloud = self.o3d_pointcloud.transform(self.camera_to_robot_tf)
                frame_id = self.robot_frame
                self.processing_times['transform'] = time.time() - start_time

                # # Remove duplicate points. todo: test
                # start_time = time.time()
                # mask = self.o3d_pointcloud.remove_duplicate_points()
                # self.o3d_pointcloud = self.o3d_pointcloud.select_by_mask(mask)
                # self.processing_times['remove_duplicate_points'] = time.time() - start_time

                # # Remove NaN points. todo: test
                # start_time = time.time()
                # self.o3d_pointcloud = self.o3d_pointcloud.remove_non_finite_points(remove_nan=True,
                #                                                                    remove_infinite=True)
                self.processing_times['remove_nan_points'] = time.time() - start_time

            # ROI cropping
            if self.crop_to_roi:
                start_time = time.time()
                # points_o3d = points_o3d.crop(self.roi_min, self.roi_max)
                mask = (
                        (self.o3d_pointcloud.point.positions[:, 0] >= self.roi_min[0]) &
                        (self.o3d_pointcloud.point.positions[:, 0] <= self.roi_max[0]) &
                        (self.o3d_pointcloud.point.positions[:, 1] >= self.roi_min[1]) &
                        (self.o3d_pointcloud.point.positions[:, 1] <= self.roi_max[1]) &
                        (self.o3d_pointcloud.point.positions[:, 2] >= self.roi_min[2]) &
                        (self.o3d_pointcloud.point.positions[:, 2] <= self.roi_max[2])
                )
                self.o3d_pointcloud = self.o3d_pointcloud.select_by_mask(mask)
                self.processing_times['crop'] = time.time() - start_time

            # Voxel downsampling
            if self.voxel_size > 0.0:
                start_time = time.time()
                self.o3d_pointcloud = self.o3d_pointcloud.voxel_down_sample(self.voxel_size)
                self.processing_times['voxel_downsampling'] = time.time() - start_time

            if self.remove_statistical_outliers:
                start_time = time.time()
                self.o3d_pointcloud, _ = self.o3d_pointcloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
                self.processing_times['remove_statistical_outliers'] = time.time() - start_time

            if self.estimate_normals:
                start_time = time.time()
                self.o3d_pointcloud.estimate_normals(
                        radius=0.1,  # Use a radius of 10 cm for local geometry
                        max_nn=30  # Use up to 30 nearest neighbors
                )
                self.processing_times['normal_estimation'] = time.time() - start_time

            # Ground segmentation.
            if self.remove_ground:
                start_time = time.time()
                plane_model, inliers = self.o3d_pointcloud.segment_plane(
                        distance_threshold=0.2,
                        ransac_n=5,
                        num_iterations=100
                )
                # ground_cloud = self.o3d_pointcloud.select_by_index(inliers)  # ground
                self.o3d_pointcloud = self.o3d_pointcloud.select_by_index(inliers, invert=True)  #
                self.processing_times['ground_segmentation'] = time.time() - start_time

            # Publish processed point cloud
            start_time = time.time()
            pc_msg = self.tensor_to_ros_cloud(self.o3d_pointcloud, frame_id, stamp=None)
            # pc_msg = self.o3d_pcd_to_ros_pcd2(points_o3d, frame_id, stamp=None)
            self.processing_times['pointcloud_msg_parsing'] = time.time() - start_time

            start_time = time.time()
            self.pointcloud_pub.publish(pc_msg)
            self.processing_times['pointcloud_pub'] = time.time() - start_time

            # # Log processing info
            # self.get_logger().info(
            #         f"Published processed pointcloud with "
            #         f"{self.o3d_pointcloud.point.positions.shape[0]} points"
            # )

            # self.get_logger().info(
            #         f"\n Ros to numpy: {1 / self.processing_times['ros_to_numpy']}, "
            #         f"data preparation: {1 / self.processing_times['data_preparation']}, "
            #         f"pcd creation: {1 / self.processing_times['pcd_creation']}, "
            #         f"tensor transfer: {1 / self.processing_times['tensor_transfer']}, "
            #         f"tf_lookup: {1 / self.processing_times['tf_lookup']}, "
            #         f"tf transform: {1 / self.processing_times['transform']}, "
            #         f"crop: {1 / self.processing_times['crop']}, "
            #         f"voxel downsampling: {1 / self.processing_times['voxel_downsampling']}, "
            #         f"statistical outlier removal: {1 / self.processing_times['remove_statistical_outliers']}, "
            #         f"normal estimation: {1 / self.processing_times['normal_estimation']}, "
            #         # f"ground segmentation: {1 / self.processing_times['ground_segmentation']}, "
            #         f"pointcloud parsing: {1 / self.processing_times['pointcloud_msg_parsing']}, "
            #         f"pointcloud publishing: {1 / self.processing_times['pointcloud_pub']} \n"
            # )
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def get_camera_to_robot_tf(self, source_frame_id, timestamp=None):
        if self.camera_to_robot_tf is not None and self.static_camera_to_robot_tf:
            return

        if timestamp is None:
            timestamp = rclpy.time.Time()
        if self.robot_frame:
            # Try to get the transform from camera to robot
            try:
                transform = self.tf_buffer.lookup_transform(
                        self.robot_frame,
                        source_frame_id,
                        timestamp,  # this could also be the depth msg timestamp
                        rclpy.duration.Duration(seconds=self.transform_timeout)
                )
            except tf2_ros.LookupException as e:
                self.get_logger().error(f"TF Lookup Error: {str(e)}")
                return
            except tf2_ros.ConnectivityException as e:
                self.get_logger().error(f"TF Connectivity Error: {str(e)}")
                return
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().error(f"TF Extrapolation Error: {str(e)}")
                return

            # Convert the TF transform to a 4x4 transformation matrix
            self.camera_to_robot_tf = self.transform_to_matrix(transform)
            return


    def transform_to_matrix(self, transform: TransformStamped):
        """Convert TransformStamped to 4x4 transformation matrix."""
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        matrix[:3, 3] = [translation.x, translation.y, translation.z]

        tf_matrix = o3c.Tensor(matrix, dtype=o3c.float32, device=self.o3d_device)

        # self.camera_to_robot_tf = tf_matrix
        return tf_matrix


    def o3d_pcd_to_ros_pcd2(self, o3d_pcd, frame_id, stamp=None):
        """Convert Open3D point cloud to ROS PointCloud2 message."""
        # Get points and colors from Open3D point cloud
        points = o3d_pcd.point.positions.cpu().numpy()
        colors = o3d_pcd.point.colors.cpu().numpy()

        if stamp is None:
            stamp = self.get_clock().now().to_msg()


        # Create PointCloud2 message
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        msg = PointCloud2()
        msg.header = header

        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        # Combine XYZ and RGB into a structured array
        cloud_data = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])

        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]

        # Convert RGB colors from float [0,1] to uint8 [0,255] and pack into float32
        rgb_colors = (colors * 255).astype(np.uint8)
        rgb = rgb_colors.astype(np.uint32)
        rgb = np.array((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2]))
        cloud_data['rgb'] = rgb.view(np.float32)

        msg.data = cloud_data.tobytes()

        return msg

    def tensor_to_ros_cloud(self, pcd_tensor, frame_id="camera_link", stamp=None):
        """Convert Open3D tensor pointcloud to ROS PointCloud2 message"""
        # Get points and colors from tensor (moved to CPU if necessary)
        points = pcd_tensor.point.positions.cpu().numpy()
        colors = pcd_tensor.point.colors.cpu().numpy()

        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        # Convert colors from float [0,1] to uint32 packed RGB
        colors_uint32 = (colors * 255).astype(np.uint8)
        rgb_packed = np.zeros(len(points), dtype=np.uint32)
        rgb_packed = np.left_shift(colors_uint32[:, 0].astype(np.uint32), 16) | \
                    np.left_shift(colors_uint32[:, 1].astype(np.uint32), 8) | \
                    colors_uint32[:, 2].astype(np.uint32)
        rgb_packed_float = rgb_packed.view(np.float32)

        # Combine XYZ and RGB data
        cloud_data = np.column_stack((points, rgb_packed_float))

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        # Create header
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp

        # Create and return PointCloud2 message
        return point_cloud2.create_cloud(header, fields, cloud_data)

    def convert_to_open3d_tensor(self, input_array):
        if isinstance(input_array, np.ndarray):
            # could also initialize as an Open3D tensor directly
            input_array = o3c.Tensor(input_array, device=self.o3d_device)

        if isinstance(input_array, torch.Tensor):
            input_array = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(input_array))

        return input_array


def main(args=None):
    rclpy.init(args=args)
    pcd_preprocessor = PointcloudPreprocessorNode()
    rclpy.spin(pcd_preprocessor)
    pcd_preprocessor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()