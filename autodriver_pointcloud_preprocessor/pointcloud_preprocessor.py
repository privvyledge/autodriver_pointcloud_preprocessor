"""
1. Subscribe to pointcloud [done]
1i. Add QoS profiles [done]
2. Passthrough pointcloud and profile [done]
3. Publish transformed pointcloud [done]
4. Publish roi removed pointcloud [done]
5. Publish voxelized pointcloud [done]
6. Publish preprocessed pointcloud [done]

Todo:
    * add support for other fields such as intensity, time, azimuth, distance and other Autoware/LIVOX fields [done]
    * add open3d visualization [done]
    * add header overrides [done]
    * add parameters for processing functions [done]
    * add support for tranforming frame [done]
    * add dynamic parameter support for online tuning [done]
    * move functions to utils [done]
    * switch to using time.perf_counter instead of time.time [done]
    * add support for saving visualization view [done]
    * add support for saving to PCDs [done]
    * move duplicate removal to a function in utils [done]
    * Move transformation, pointcloud msg parsing (to/from), conversion to/from Open3D tensors to  utils file [done]
    * Move preprocessing to a separate function for reusing in different nodes. [done]
    * optimize ROS <--> Open3D extraction, i.e PointCloud from dictonary (and test with GPU) [done]
    * use "partial" from functools and initialize the timing dict once
    * optionally publish processing times as a ROS message
    * test pointcloud transformation to different frames
    * add other preprocessing steps such as furthest point downsampling, uniform downsampling, random downsampling, radius outlier removal, etc.
        * See https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html
    * if estimate_normals is True, also publish a marker of normals (could also be done in a separate thread or by another node)
    * create fpfh feature estimation node https://www.open3d.org/docs/latest/python_api/open3d.t.pipelines.registration.compute_fpfh_feature.html
    * add good defaults for all parameters, e.g n * voxel_size. See Open3D documentation for more details.
    * finalize ROS <--> Numpy conversion by comparing to other git repos
    * Optimize different names/fields from different vendors and unify in a dictionary
    * replace infinite/nan removal with numpy and torch native operations
    * add onetime height/ground estimation from my autodriving transformer code
    * Create a Python "package" for standalone non-ROS use then just import that here
    * make torch an optional dependency, i.e. only use it if the user has it installed. Device checks for Open3D do not need torch.
    * Add parameters for all preprocessing functions and make them modular to be used in composable nodes
    * Remove Todo comments and add to package description/features
"""
import os
import sys
import pathlib  # avoid importing Path to avoid clashes with nav2_msgs/msg/Path.py

from utils import check_field, crop_pointcloud

import time
import struct
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy import qos
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
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

from autodriver_pointcloud_preprocessor.utils import (convert_pointcloud_to_numpy, numpy_struct_to_pointcloud2,
                                                      get_current_time, get_time_difference,
                                                      dict_to_open3d_tensor_pointcloud,
                                                      pointcloud_to_dict, get_pointcloud_metadata,
                                                      extract_rgb_from_pointcloud, get_fields_from_dicts,
                                                      remove_duplicates, rgb_int_to_float,
                                                      FIELD_DTYPE_MAP, FIELD_DTYPE_MAP_INV)



class PointcloudPreprocessorNode(Node):
    def __init__(self):
        super(PointcloudPreprocessorNode, self).__init__('pointcloud_preprocessor')

        # Declare parameters
        self.declare_parameter(name='input_topic', value="/velodyne_front/velodyne_points",  # /camera/camera/depth/color/points, /lidar1/velodyne_points , /velodyne_front/velodyne_points,
                               descriptor=ParameterDescriptor(
                                   description='',
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='output_topic', value="/lidar1/velodyne_points/processed",
                               # /camera/camera/depth/color/points/processed
                               descriptor=ParameterDescriptor(
                                   description='',
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
            description='',
            type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('pointcloud_fields', [])
        self.declare_parameter('queue_size', 1)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('cpu_backend', 'torch')  # numpy, pytorch or open3d
        self.declare_parameter('gpu_backend', 'open3d')  # pytorch or open3d
        self.declare_parameter('robot_frame', '')
        self.declare_parameter('static_camera_to_robot_tf', True)
        self.declare_parameter('transform_timeout', 0.1)
        self.declare_parameter('organize_cloud', False)
        self.declare_parameter('save_pointcloud', False)
        self.declare_parameter('pointcloud_save_directory', './pointclouds/')
        self.declare_parameter('pointcloud_save_prepend_str', '')
        self.declare_parameter('pointcloud_save_extension', '.pcd')  # .pcd, .ply, .pts, .xyzrgb, .xyzn, .xyzn
        self.declare_parameter('pointcloud_save_ascii', False)  # False: 'binary', True: 'ascii'
        self.declare_parameter('pointcloud_save_compressed', False)

        self.declare_parameter('remove_duplicates', True)
        self.declare_parameter('remove_nans', True)
        self.declare_parameter('remove_infs', True)  # False
        self.declare_parameter('crop_to_roi', True)
        self.declare_parameter('crop_to_roi.invert', False)  # True: keeps points outside the ROI, False: keeps points inside the ROI. Default is False.
        self.declare_parameter('roi_min', [-60.0, -60.0, -20.0])  # [-6.0, -6.0, -0.05]
        self.declare_parameter('roi_max', [60.0, 60.0, 20.0])  # [6.0, 6.0, 2.0]
        self.declare_parameter('voxel_size', 0.01)  # 0.01
        self.declare_parameter('remove_statistical_outliers', False)
        self.declare_parameter('remove_statistical_outliers.nb_neighbors', 20)
        self.declare_parameter('remove_statistical_outliers.std_ratio', 2.0)
        self.declare_parameter('estimate_normals', True)
        self.declare_parameter('estimate_normals.search_radius', 0.1)
        self.declare_parameter('estimate_normals.max_neighbors', 30)
        self.declare_parameter('remove_ground', False)
        self.declare_parameter('remove_ground.distance_threshold', 0.2)  # distance threshold for ground segmentation
        self.declare_parameter('remove_ground.ransac_number', 5)  # number of points to sample for RANSAC
        self.declare_parameter('remove_ground.num_iterations', 100)
        self.declare_parameter('remove_ground.probability', 0.99)  # probability of finding a plane
        self.declare_parameter('ground_plane', [0.0, 1.0, 0.0, 0.0])
        self.declare_parameter('use_height', True)  # if true, remove the ground based on height

        self.declare_parameter('override_header', False)
        self.declare_parameter('override_header.stamp_source', 'latest')  # copy, latest

        self.declare_parameter('visualize', False)
        self.declare_parameter('visualize.window_name', 'Open3D')
        self.declare_parameter('visualize.window_width', 1920)
        self.declare_parameter('visualize.window_height', 1080)
        self.declare_parameter('visualize.zoom', 0.0)
        self.declare_parameter('visualize.front', [])
        self.declare_parameter('visualize.lookat', [])
        self.declare_parameter('visualize.up', [])
        self.declare_parameter('visualize.save_visualizer_image', False)
        self.declare_parameter('visualize.visualizer_image_path', './images')

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.qos = self.get_parameter('qos').get_parameter_value().string_value
        self.pointcloud_fields = self.get_parameter('pointcloud_fields').value
        self.queue_size = self.get_parameter('queue_size').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.cpu_backend = self.get_parameter('cpu_backend').value
        self.gpu_backend = self.get_parameter('gpu_backend').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.static_camera_to_robot_tf = self.get_parameter('static_camera_to_robot_tf').value
        self.transform_timeout = self.get_parameter('transform_timeout').value
        self.organize_cloud = self.get_parameter('organize_cloud').value
        self.save_pointcloud = self.get_parameter('save_pointcloud').value
        self.pointcloud_save_directory = self.get_parameter('pointcloud_save_directory').value
        if self.save_pointcloud:
            os.makedirs(self.pointcloud_save_directory, exist_ok=True)
        if not self.pointcloud_save_directory:
            self.pointcloud_save_directory = '.'  # os.getcwd()
        self.pointcloud_save_prepend_str = self.get_parameter('pointcloud_save_prepend_str').value
        self.pointcloud_save_extension = self.get_parameter('pointcloud_save_extension').value
        self.pointcloud_save_ascii = self.get_parameter('pointcloud_save_ascii').value
        self.pointcloud_save_compressed = self.get_parameter('pointcloud_save_compressed').value

        self.remove_duplicates = self.get_parameter('remove_duplicates').get_parameter_value().bool_value
        self.remove_nans = self.get_parameter('remove_nans').get_parameter_value().bool_value
        self.remove_infs = self.get_parameter('remove_infs').get_parameter_value().bool_value
        self.crop_to_roi = self.get_parameter('crop_to_roi').value
        self.crop_to_roi_invert = self.get_parameter('crop_to_roi.invert').value
        self.roi_min = self.get_parameter('roi_min').value
        self.roi_max = self.get_parameter('roi_max').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.remove_statistical_outliers = self.get_parameter('remove_statistical_outliers').value
        self.remove_statistical_outliers_nb_neighbors = self.get_parameter('remove_statistical_outliers.nb_neighbors').get_parameter_value().integer_value
        self.remove_statistical_outliers_std_ratio = self.get_parameter('remove_statistical_outliers.std_ratio').get_parameter_value().double_value
        self.estimate_normals = self.get_parameter('estimate_normals').value
        self.estimate_normals_search_radius = self.get_parameter('estimate_normals.search_radius').get_parameter_value().double_value
        self.estimate_normals_max_neighbors = self.get_parameter('estimate_normals.max_neighbors').get_parameter_value().integer_value
        self.remove_ground = self.get_parameter('remove_ground').value
        self.remove_ground_distance_threshold = self.get_parameter('remove_ground.distance_threshold').get_parameter_value().double_value
        self.remove_ground_ransac_number = self.get_parameter('remove_ground.ransac_number').get_parameter_value().integer_value
        self.remove_ground_num_iterations = self.get_parameter('remove_ground.num_iterations').get_parameter_value().integer_value
        self.remove_ground_probability = self.get_parameter('remove_ground.probability').get_parameter_value().double_value
        self.ground_plane = self.get_parameter('ground_plane').value
        self.use_height = self.get_parameter('use_height').value
        self.override_header = self.get_parameter('override_header').value
        if self.override_header:
            self.new_header_data = {
                'frame_id': self.robot_frame,
                'stamp_source': self.get_parameter('override_header.stamp_source').value
            }
        self.visualize = self.get_parameter('visualize').value

        # Setup the device
        self.torch_device = torch.device('cpu')
        self.o3d_device = o3d.core.Device('CPU:0')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.torch_device = torch.device('cuda:0')
            if o3d.core.cuda.is_available():
                self.o3d_device = o3d.core.Device('CUDA:0')
            else:
                self.use_gpu = False

        # Initialize variables
        self.camera_to_robot_tf = None
        self.pointcloud_dictionary = {
            'header': None,
            'positions': None,
            # 'rgb': None,
            # 'intensity': None,
            # 'ring': None,
            # 'time': None,
            # 'return_type': None
        }  # todo: get from ros param
        self.frame_count = 0

        # TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)
        if self.crop_to_roi:
            min_bound = o3c.Tensor(self.roi_min, dtype=o3c.Dtype.Float32)
            max_bound = o3c.Tensor(self.roi_max, dtype=o3c.Dtype.Float32)
            self.crop_aabb = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound).to(self.o3d_device)

        self.pointcloud_metadata = None

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

        # Setup Open3D visualization
        if self.visualize:
            save_visualizer_image = bool(self.get_parameter('visualize.save_visualizer_image').value)
            visualizer_image_path = str(self.get_parameter('visualize.visualizer_image_path').value)
            if not visualizer_image_path:
                visualizer_image_path = '.'  # os.getcwd()
            if save_visualizer_image and not os.path.exists(visualizer_image_path):
                os.makedirs(visualizer_image_path, exist_ok=True)
            self.visualizer_options = {
                'window_name': str(self.get_parameter('visualize.window_name').get_parameter_value().string_value),
                'window_width': int(self.get_parameter('visualize.window_width').get_parameter_value().integer_value),
                'window_height': int(self.get_parameter('visualize.window_height').get_parameter_value().integer_value),
                'zoom': self.get_parameter('visualize.zoom').get_parameter_value().double_value,
                'front': self.get_parameter('visualize.front').get_parameter_value().double_array_value,
                'lookat': self.get_parameter('visualize.lookat').get_parameter_value().double_array_value,
                'up': self.get_parameter('visualize.up').get_parameter_value().double_array_value,
                'save_visualizer_image': save_visualizer_image,
                'visualizer_image_path': visualizer_image_path
            }
            # Fix GLFW issues with Wayland by switching to 'x11' context
            os.environ['XDG_SESSION_TYPE'] = 'x11'

            # Create Open3D non-blocking visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.visualizer_options['window_name'],
                width=self.visualizer_options['window_width'],
                height=self.visualizer_options['window_height']
            )
            # self.vis.get_render_option().point_size = 2
            self.vis.add_geometry(self.o3d_pointcloud.to_legacy())

        # Setup dynamic parameter reconfiguring.
        # Register a callback function that will be called whenever there is an attempt to
        # change one or more parameters of the node.
        self.add_on_set_parameters_callback(self.parameter_change_callback)

        # Setup subscribers
        self.poincloud_sub = self.create_subscription(PointCloud2, self.input_topic,
                                                      self.callback, qos_profile=qos_profile)

        # Setup publishers
        self.pointcloud_pub = self.create_publisher(PointCloud2, self.output_topic, self.queue_size)

        # self.pointcloud_timer = self.create_timer(1 / self.odom_rate, self.rgbd_timer_callback)
        self.get_logger().info(f"{self.get_fully_qualified_name()} node started on device: {self.o3d_device}")


    def preprocess(self):
        # Remove duplicate points.
        if self.remove_duplicates:
            start_time = get_current_time(monotonic=True)
            if 'cpu' in str(self.o3d_device).lower():
                self.o3d_pointcloud, dupl_msg = remove_duplicates(self.o3d_pointcloud, self.cpu_backend)

            elif 'cuda' in str(self.o3d_device).lower() or 'gpu' in str(self.o3d_device).lower():
                self.o3d_pointcloud, dupl_msg = remove_duplicates(self.o3d_pointcloud, self.gpu_backend)

            else:
                self.get_logger().info('No valid device/backend found. Using torch for duplicate removal.')
                self.o3d_pointcloud, dupl_msg = remove_duplicates(self.o3d_pointcloud, 'torch')

            self.get_logger().info(dupl_msg)
            self.processing_times['remove_duplicate_points'] = get_time_difference(start_time, get_current_time(monotonic=True))

        # Remove NaN points.
        if self.remove_nans or self.remove_infs:
            # todo: add numpy and torch operations for potential speedups
            start_time = get_current_time(monotonic=True)
            self.o3d_pointcloud, non_finite_masks = self.o3d_pointcloud.remove_non_finite_points(
                remove_nan=self.remove_nans,
                remove_infinite=self.remove_infs)
            self.processing_times['remove_nan_points'] = get_time_difference(start_time, get_current_time(monotonic=True))

        # transform to robot frame
        start_time = get_current_time(monotonic=True)
        self.get_camera_to_robot_tf(self.pointcloud_metadata["header"].frame_id,
                                    rclpy.time.Time().from_msg(self.pointcloud_metadata["header"].stamp))
        self.processing_times['tf_lookup'] = get_time_difference(start_time, get_current_time(monotonic=True))

        if self.camera_to_robot_tf is not None:
            start_time = get_current_time(monotonic=True)
            self.o3d_pointcloud = self.o3d_pointcloud.transform(self.camera_to_robot_tf)
            frame_id = self.robot_frame
            self.processing_times['transform'] = get_time_difference(start_time, get_current_time(monotonic=True))

        # ROI cropping
        if self.crop_to_roi:
            start_time = get_current_time(monotonic=True)
            # todo: create partial function to reduce min/max bound, invert and aabb
            if 'cpu' in str(self.o3d_device).lower():
                self.o3d_pointcloud, crop_msg = crop_pointcloud(self.o3d_pointcloud, backend=self.cpu_backend,
                                                                min_bound=self.roi_min, max_bound=self.roi_max,
                                                                invert=self.crop_to_roi_invert, aabb=self.crop_aabb)

            elif 'cuda' in str(self.o3d_device).lower() or 'gpu' in str(self.o3d_device).lower():
                self.o3d_pointcloud, crop_msg = crop_pointcloud(self.o3d_pointcloud, backend=self.gpu_backend,
                                                                min_bound=self.roi_min, max_bound=self.roi_max,
                                                                invert=self.crop_to_roi_invert, aabb=self.crop_aabb)

            else:
                self.get_logger().info('No valid device/backend found. Using open3d backend for pointcloud cropping.')
                self.o3d_pointcloud, crop_msg = crop_pointcloud(self.o3d_pointcloud, backend='open3d',
                                                                min_bound=self.roi_min, max_bound=self.roi_max,
                                                                invert=self.crop_to_roi_invert, aabb=self.crop_aabb)
            self.get_logger().info(crop_msg)
            self.processing_times['crop'] = get_time_difference(start_time, get_current_time(monotonic=True))

        # Voxel downsampling
        if self.voxel_size > 0.0:
            start_time = get_current_time(monotonic=True)
            self.o3d_pointcloud = self.o3d_pointcloud.voxel_down_sample(self.voxel_size)
            self.processing_times['voxel_downsampling'] = get_time_difference(start_time, get_current_time(monotonic=True))

        if self.remove_statistical_outliers:
            start_time = get_current_time(monotonic=True)
            self.o3d_pointcloud, _ = self.o3d_pointcloud.remove_statistical_outliers(
                    nb_neighbors=self.remove_statistical_outliers_nb_neighbors,
                    std_ratio=self.remove_statistical_outliers_std_ratio)
            self.processing_times['remove_statistical_outliers'] = get_time_difference(start_time, get_current_time(monotonic=True))

        if self.estimate_normals:
            start_time = get_current_time(monotonic=True)
            self.o3d_pointcloud.estimate_normals(
                radius=self.estimate_normals_search_radius,  # Use a radius of 10 cm for local geometry
                max_nn=self.estimate_normals_max_neighbors,  # Use up to 30 nearest neighbors
            )
            # todo: normalize_normals and add other normal estimation options like orientation
            # https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.normalize_normals
            self.pointcloud_metadata['has_normals'] = True
            self.processing_times['normal_estimation'] = get_time_difference(start_time, get_current_time(monotonic=True))

        # Ground segmentation.
        if self.remove_ground:
            start_time = get_current_time(monotonic=True)
            plane_model, inliers = self.o3d_pointcloud.segment_plane(
                distance_threshold=self.remove_ground_distance_threshold,
                ransac_n=self.remove_ground_ransac_number,
                num_iterations=self.remove_ground_num_iterations,
                probability=self.remove_ground_probability
            )
            # ground_cloud = self.o3d_pointcloud.select_by_index(inliers)  # ground
            self.o3d_pointcloud = self.o3d_pointcloud.select_by_index(inliers, invert=True)  #
            self.processing_times['ground_segmentation'] = get_time_difference(start_time, get_current_time(monotonic=True))
        return self.o3d_pointcloud

    def callback(self, ros_cloud):
        # Check if there are subscribers to the topic published by this node
        if self.pointcloud_pub.get_subscription_count() == 0:
            return

        frame_id = ros_cloud.header.frame_id

        try:
            callback_start_time = get_current_time(monotonic=True)
            try:
                start_time = get_current_time(monotonic=True)
                field_names = self.pointcloud_fields if self.pointcloud_fields else None  # ('x', 'y', 'z', 'rgb'),
                self.pointcloud_dictionary, pointcloud_metadata = pointcloud_to_dict(
                    ros_cloud, field_names, self.remove_nans, self.organize_cloud)

                if self.pointcloud_metadata is None:
                    # Optional: Extract additional attributes if present
                    self.pointcloud_metadata = get_pointcloud_metadata(pointcloud_metadata['field_names'])

                self.pointcloud_metadata.update(pointcloud_metadata)
                cloud_field_names = self.pointcloud_metadata.get('field_names', None)
                num_fields = self.pointcloud_metadata['num_fields']

            except Exception as e:
                self.get_logger().error(f"Failed to convert PointCloud2 message to numpy: {str(e)}")
                return

            if self.pointcloud_dictionary['cloud_array'].size == 0:
                self.get_logger().warn("Received an empty PointCloud. Skipping...")
                return

            self.pointcloud_dictionary.pop('cloud_array')

            # test if x, y, and z are present
            if not {"x", "y", "z"}.issubset(cloud_field_names):
                self.get_logger().error("Incoming PointCloud does not have x, y, z fields.")
                return

            self.processing_times['ros_to_numpy'] = get_time_difference(start_time, get_current_time(monotonic=True))

            # Extract XYZ points
            start_time = get_current_time(monotonic=True)

            # Extract and convert RGB values
            if 'rgb' in cloud_field_names and self.pointcloud_metadata['has_rgb']:
                rgb_arr = extract_rgb_from_pointcloud(self.pointcloud_dictionary['rgb'])
            else:
                rgb_arr = None

            self.processing_times['data_preparation'] = get_time_difference(start_time, get_current_time(monotonic=True))

            # Clear previous point and attributes
            start_time = get_current_time(monotonic=True)
            self.o3d_pointcloud.clear() # todo: maybe also delete non-header keys from the dictionary
            self.processing_times['point_clearing'] = get_time_difference(start_time, get_current_time(monotonic=True))

            # Convert numpy arrays to tensors and move to device. We
            start_time = get_current_time(monotonic=True)
            self.pointcloud_dictionary['positions'] = o3c.Tensor.from_numpy(self.pointcloud_dictionary['positions'])

            if rgb_arr is not None:
                rgb_f = rgb_arr.astype(np.float32) / 255.0
                if check_field('rgb'):
                    self.pointcloud_dictionary['rgb'] = o3c.Tensor.from_numpy(rgb_f)

            # todo: switch to functools.partial to reduce code duplication
            self.pointcloud_dictionary = get_fields_from_dicts('intensity', self.pointcloud_dictionary,
                                                                            self.pointcloud_metadata)
            self.pointcloud_dictionary = get_fields_from_dicts('ring', self.pointcloud_dictionary,
                                                                            self.pointcloud_metadata)
            self.pointcloud_dictionary = get_fields_from_dicts('time', self.pointcloud_dictionary,
                                                                            self.pointcloud_metadata)
            self.pointcloud_dictionary = get_fields_from_dicts('return_type', self.pointcloud_dictionary,
                                                                            self.pointcloud_metadata)

            self.o3d_pointcloud = dict_to_open3d_tensor_pointcloud(self.pointcloud_dictionary, device=self.o3d_device)
            self.processing_times['tensor_transfer'] = get_time_difference(start_time, get_current_time(monotonic=True))

            # preprocess the PointCloud
            preprocessing_start_time = get_current_time(monotonic=False)
            self.preprocess()
            self.processing_times['preprocessing_time'] = get_time_difference(start_time, get_current_time(monotonic=False))

            # Publish processed point cloud
            start_time = get_current_time(monotonic=True)
            processed_positions = self.o3d_pointcloud.point.positions.cpu().numpy()
            num_points = processed_positions.shape[0]
            # todo: this field name and type extraction should be done once since it is most likely static unless a parameter is updated or the pointcloud type changes which is unlikely
            orig_field_names = []
            orig_field_types = []

            for f in ros_cloud.fields:
                orig_field_names.append(f.name)
                orig_field_types.append(f.datatype)

            new_dtype = []
            for name, datatype in zip(orig_field_names, orig_field_types):
                np_type = FIELD_DTYPE_MAP[datatype]
                new_dtype.append((name, np_type))

            if self.estimate_normals:
                orig_field_names.extend(['normal_x', 'normal_y', 'normal_z'])
                orig_field_types.extend([PointField.FLOAT32, PointField.FLOAT32, PointField.FLOAT32])
                new_dtype.extend([
                    ('normal_x', FIELD_DTYPE_MAP[PointField.FLOAT32]),
                    ('normal_y', FIELD_DTYPE_MAP[PointField.FLOAT32]),
                    ('normal_z', FIELD_DTYPE_MAP[PointField.FLOAT32])
                ])

            processed_struct = np.zeros(num_points, dtype=new_dtype)
            processed_struct["x"] = processed_positions[:, 0]
            processed_struct["y"] = processed_positions[:, 1]
            processed_struct["z"] = processed_positions[:, 2]

            # Handle other fields. todo: could do this once instead of for each pointcloud message
            rgb_np = self.copy_fields('rgb', self.o3d_pointcloud)
            if rgb_np:
                rgb_float32 = rgb_int_to_float(rgb_np)
                processed_struct["rgb"] = rgb_float32

            if check_field('intensity', self.pointcloud_dictionary, self.pointcloud_metadata):
                intensity_np = self.copy_fields('intensity', self.o3d_pointcloud)
                processed_struct["intensity"] = intensity_np.astype(processed_struct["intensity"].dtype)

            if check_field('ring', self.pointcloud_dictionary, self.pointcloud_metadata):
                ring_np = self.copy_fields('ring', self.o3d_pointcloud)
                processed_struct["ring"] = ring_np.astype(processed_struct["ring"].dtype)

            if check_field('time', self.pointcloud_dictionary, self.pointcloud_metadata):
                time_np = self.copy_fields('time', self.o3d_pointcloud)
                processed_struct["time"] = time_np.astype(processed_struct["time"].dtype)

            if self.estimate_normals or check_field('normal_x', self.pointcloud_dictionary, self.pointcloud_metadata):
                normals_np = self.o3d_pointcloud.point.normals.cpu().numpy()
                processed_struct["normal_x"] = normals_np[:, 0].astype(processed_struct["normal_x"].dtype)  # normal_* or n*
                processed_struct["normal_y"] = normals_np[:, 1].astype(processed_struct["normal_y"].dtype)  # normal_* or n*
                processed_struct["normal_z"] = normals_np[:, 2].astype(processed_struct["normal_z"].dtype)  # normal_* or n*

            # Create PointCloud2 message.
            new_header = ros_cloud.header
            if self.override_header:
                if self.new_header_data['stamp_source'].lower() == 'latest':
                    # latest gets the most recent time
                    new_header.stamp = self.get_clock().now().to_msg()

                if self.new_header_data['frame_id'].lower():
                    new_header.frame_id = self.new_header_data['frame_id']

            # todo: this should be done once and not for each pointcloud message
            pointfields, point_offset = numpy_struct_to_pointcloud2(
                    processed_struct,
                    field_names=orig_field_names,
                    field_datatypes=orig_field_types,
                    is_dense=self.remove_nans and self.remove_infs
            )
            pc_msg = self.tensor_to_ros_cloud(processed_struct, pointfields, header=new_header)
            pc_msg.is_dense = self.remove_nans and self.remove_infs

            self.processing_times['pointcloud_msg_parsing'] = get_time_difference(start_time, get_current_time(monotonic=True))

            start_time = get_current_time(monotonic=True)
            self.pointcloud_pub.publish(pc_msg)
            self.processing_times['pointcloud_pub'] = get_time_difference(start_time, get_current_time(monotonic=True))

            pcd_number = str(self.frame_count).zfill(8)
            self.pointcloud_saver(pcd_number)
            self.pointcloud_visualizer(pcd_number)

            self.frame_count += 1
            self.processing_times['total_callback_time'] = get_current_time(monotonic=True) - callback_start_time

            # Log processing info
            self.get_logger().info(
                    f"Published processed pointcloud with "
                    f"{self.o3d_pointcloud.point.positions.shape[0]} points"
            )

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
            timestamp = rclpy.time.Time()  # self.get_clock().now()
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

        tf_matrix = o3c.Tensor(matrix, dtype=o3c.float32, device=self.o3d_device)  # todo: declare once then update here

        # self.camera_to_robot_tf = tf_matrix
        return tf_matrix

    def o3d_pcd_to_ros_pcd2(self, o3d_pcd, frame_id, stamp=None):
        """Todo: remove. Convert Open3D point cloud to ROS PointCloud2 message."""
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

    def tensor_to_ros_cloud(self, cloud_data, fields, header=None):
        """Convert Open3D tensor pointcloud to ROS PointCloud2 message"""
        if header is None:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()

        # Create and return PointCloud2 message
        return point_cloud2.create_cloud(header, fields, cloud_data)

    def convert_to_open3d_tensor(self, input_array):
        if isinstance(input_array, np.ndarray):
            # could also initialize as an Open3D tensor directly
            if 'cpu' in str(self.o3d_device).lower():
                # use direct memory mapping to avoid copying
                output_array = o3c.Tensor.from_numpy(input_array)
            else:
                output_array = o3c.Tensor(input_array, device=self.o3d_device)

        elif isinstance(input_array, torch.Tensor):
            output_array = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(input_array))

        else:
            self.get_logger().warn("The input array is neither a numpy ndarray nor a torch tensor. Passing through.")
            output_array = o3c.Tensor(input_array, device=self.o3d_device)

        return output_array

    def copy_fields(self, field_name, pointcloud=None):
        field_names = field_name
        if isinstance(field_name, str):
            field_names = [field_name]
        if pointcloud is None:
            pointcloud = self.o3d_pointcloud

        processed_fields = {}

        for field_name_ in field_names:
            try:
                field_tensor = pointcloud.point[field_name_]  # self.o3d_pointcloud.point[field_name_], getattr(pointcloud.point, field_name_)
                field_arr = field_tensor.cpu().numpy().reshape(-1)
                processed_fields[field_name_] = field_arr
            except KeyError:
                # the field does not exist
                self.get_logger().warn(f"Field name: {field_name_} not found in pointcloud {pointcloud}.",
                                       throttle_duration_sec=5.0)
                processed_fields[field_name_] = None

        if isinstance(field_name, str):
            processed_fields = processed_fields[field_name]
        return processed_fields

    def publish_normals_marker_array(self, pointcloud):
        pass

    def parameter_change_callback(self, params):
        """
        Todo:
            * change topics (input/output) and destroy subscribers/publishers
        Triggered whenever there is a change request for one or more parameters.

        Args:
            params (List[Parameter]): A list of Parameter objects representing the parameters that are
                being attempted to change.

        Returns:
            SetParametersResult: Object indicating whether the change was successful.
        """
        result = SetParametersResult()
        result.successful = True

        # Iterate over each parameter in this node
        for param in params:
            if param.name == 'use_gpu' and param.type_ == Parameter.Type.BOOL:
                # first set all to CPU
                self.torch_device = torch.device('cpu')
                self.o3d_device = o3d.core.Device('CPU:0')
                self.use_gpu = False

                # Then check GPU availability if use_gpu
                use_gpu = param.value
                if use_gpu:
                    if torch.cuda.is_available():
                        self.torch_device = torch.device('cuda:0')
                    else:
                        self.get_logger().warn("Torch was not installed/built with CUDA support. "
                                               "GPU backend cannot use torch functions.")

                    if o3d.core.cuda.is_available():
                        self.o3d_device = o3d.core.Device('CUDA:0')
                        self.use_gpu = True
                    else:
                        self.get_logger().warn("Open3D was not installed/built with CUDA support. "
                                               "Using CPU for Open3D functions instead instead.")
                        self.use_gpu = False
                        result.successful = False

            elif param.name == 'cpu_backend' and param.type_ == Parameter.Type.STRING:
                self.cpu_backend = param.value
            elif param.name == 'gpu_backend' and param.type_ == Parameter.Type.STRING:
                self.gpu_backend = param.value
            elif param.name == 'robot_frame' and param.type_ == Parameter.Type.STRING:
                self.robot_frame = param.value
                try:
                    self.new_header_data['stamp_source'] = param.value
                except NameError:
                    pass
            elif param.name == 'static_camera_to_robot_tf' and param.type_ == Parameter.Type.BOOL:
                self.static_camera_to_robot_tf = param.value
            elif param.name == 'transform_timeout' and param.type_ == Parameter.Type.DOUBLE:
                self.transform_timeout = param.value
            elif param.name == 'organize_cloud' and param.type_ == Parameter.Type.BOOL:
                self.organize_cloud = param.value
            elif param.name == 'save_pointcloud' and param.type_ == Parameter.Type.BOOL:
                self.save_pointcloud = param.value
            elif param.name == 'pointcloud_save_directory' and param.type_ == Parameter.Type.STRING:
                self.pointcloud_save_directory = param.value
            elif param.name == 'pointcloud_save_prepend_str' and param.type_ == Parameter.Type.STRING:
                self.pointcloud_save_prepend_str = param.value
            elif param.name == 'pointcloud_save_extension' and param.type_ == Parameter.Type.STRING:
                self.pointcloud_save_extension = param.value
            elif param.name == 'pointcloud_save_ascii' and param.type_ == Parameter.Type.STRING:
                self.pointcloud_save_ascii = param.value
            elif param.name == 'pointcloud_save_compressed' and param.type_ == Parameter.Type.BOOL:
                self.pointcloud_save_compressed = param.value
            elif param.name == 'remove_duplicates' and param.type_ == Parameter.Type.BOOL:
                self.remove_duplicates = param.value
            elif param.name == 'remove_nans' and param.type_ == Parameter.Type.BOOL:
                self.remove_nans = param.value
            elif param.name == 'remove_infs' and param.type_ == Parameter.Type.BOOL:
                self.remove_infs = param.value
            elif param.name == 'crop_to_roi' and param.type_ == Parameter.Type.BOOL:
                self.crop_to_roi = param.value
                min_bound = o3c.Tensor(self.roi_min, dtype=o3c.Dtype.Float32)
                max_bound = o3c.Tensor(self.roi_max, dtype=o3c.Dtype.Float32)
                self.crop_aabb = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound).to(self.o3d_device)
            elif param.name == 'crop_to_roi.invert' and param.type_ == Parameter.Type.BOOL:
                self.crop_to_roi_invert = param.value
            elif param.name in ['roi_min', 'roi_max'] and param.type_ == Parameter.Type.DOUBLE_ARRAY:
                roi_ = param.value
                if len(roi_) == 3:
                    if param.name == 'roi_min':
                        self.roi_min = roi_
                    else:
                        self.roi_max = roi_
                else:
                    result.successful = False
            elif param.name == 'voxel_size' and param.type_ == Parameter.Type.DOUBLE:
                self.voxel_size = param.value
            elif param.name == 'remove_statistical_outliers' and param.type_ == Parameter.Type.BOOL:
                self.remove_statistical_outliers = param.value
            elif param.name == 'remove_statistical_outliers.nb_neighbors' and param.type_ == Parameter.Type.INT:
                self.remove_statistical_outliers_nb_neighbors = param.value
            elif pram.name == 'remove_statistical_outliers.std_ratio' and param.type_ == Parameter.Type.DOUBLE:
                self.remove_statistical_outliers_std_ratio = param.value
            elif param.name == 'estimate_normals' and param.type_ == Parameter.Type.BOOL:
                self.estimate_normals = param.value
            elif param.name == 'estimate_normals.search_radius' and param.type_ == Parameter.Type.DOUBLE:
                self.estimate_normals_search_radius = param.value
            elif param.name == 'estimate_normals.max_neighbors' and param.type_ == Parameter.Type.INT:
                self.estimate_normals_max_neighbors = param.value
            elif param.name == 'remove_ground' and param.type_ == Parameter.Type.BOOL:
                self.remove_ground = param.value
            elif param.name == 'remove_ground.distance_threshold' and param.type_ == Parameter.Type.DOUBLE:
                self.remove_ground_distance_threshold = param.value
            elif param.name == 'remove_ground.ransac_number' and param.type_ == Parameter.Type.INT:
                self.remove_ground_ransac_number = param.value
            elif param.name == 'remove_ground.num_iterations' and param.type_ == Parameter.Type.INT:
                self.remove_ground_num_iterations = param.value
            elif param.name == 'remove_ground.probability' and param.type_ == Parameter.Type.DOUBLE:
                self.remove_ground_probability = param.value
            elif param.name == 'ground_plane' and param.type_ == Parameter.Type.DOUBLE_ARRAY:
                self.ground_plane = param.value
            elif param.name == 'use_height' and param.type_ == Parameter.Type.BOOL:
                self.use_height = param.value
            elif param.name == 'override_header' and param.type_ == Parameter.Type.BOOL:
                self.override_header = param.value
                if self.override_header:
                    self.new_header_data = {
                        'frame_id': self.robot_frame,
                        'stamp_source': self.get_parameter('override_header.stamp_source').value
                    }
            elif param.name == 'override_header.stamp_source' and param.type_ == Parameter.Type.STRING:
                self.new_header_data['stamp_source'] = param.value
            # todo: setup visualizer and options. Also add visualizer destruction if set to False
            elif param.name == 'visualize' and param.type_ == Parameter.Type.BOOL:
                self.visualize = param.value
            else:
                result.successful = False
            self.get_logger().info(f"Success = {result.successful} for param {param.name} to value {param.value}")
        return result

    def pointcloud_saver(self, pcd_number):
        if self.save_pointcloud:
            pointcloud_extension = self.pointcloud_save_extension.strip('.')
            pcd_file_name = os.path.join(
                    self.pointcloud_save_directory,
                    f"{self.pointcloud_save_prepend_str}{pcd_number}.{pointcloud_extension}")
            o3d.t.io.write_point_cloud(filename=pcd_file_name, pointcloud=self.o3d_pointcloud,
                                       write_ascii=self.pointcloud_save_ascii,
                                       compressed=self.pointcloud_save_compressed, print_progress=False)
            # # to load the point_cloud
            # self.o3d_pointcloud2 = o3d.t.io.read_point_cloud(
            #     filename=pcd_file_name, remove_nan_points=False, remove_infinite_points=False,
            #     print_progress=True).to(self.o3d_device)

    def pointcloud_visualizer(self, pcd_number):
        if self.visualize:
            # o3d.visualization.draw_geometries([self.o3d_pointcloud.to_legacy()],
            #                                   zoom=0.3412,
            #                                   front=[0.4257, -0.2125, -0.8795],
            #                                   lookat=[2.6172, 2.0475, 1.532],
            #                                   up=[-0.0694, -0.9768, 0.2024])

            # self.vis.update_geometry(self.o3d_pointcloud.to_legacy())
            self.vis.clear_geometries()
            self.vis.add_geometry(self.o3d_pointcloud.to_legacy())
            ctr = self.vis.get_view_control()
            if self.visualizer_options['front']:
                ctr.set_front(self.visualizer_options['front'])
            if self.visualizer_options['lookat']:
                ctr.set_lookat(self.visualizer_options['lookat'])
            if self.visualizer_options['up']:
                ctr.set_up(self.visualizer_options['up'])
            if abs(self.visualizer_options["zoom"]) > 0.0:
                ctr.set_zoom(self.visualizer_options["zoom"])
            self.vis.poll_events()
            self.vis.update_renderer()

            if self.visualizer_options['save_visualizer_image']:
                image_filename = (f"{self.visualizer_options['visualizer_image_path']}/"
                                  f"{self.pointcloud_save_prepend_str}{pcd_number}.png")
                self.vis.capture_screen_image(image_filename, do_render=True)

def main(args=None):
    rclpy.init(args=args)
    pcd_preprocessor = PointcloudPreprocessorNode()
    try:
        rclpy.spin(pcd_preprocessor)
    except (KeyboardInterrupt, SystemExit):
        pcd_preprocessor.get_logger().info("Shutting down node...")
    finally:
        if pcd_preprocessor.visualize:
            pcd_preprocessor.vis.destroy_window()
        pcd_preprocessor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
