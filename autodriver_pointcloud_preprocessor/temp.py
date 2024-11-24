#!/usr/bin/env python3

import struct
import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import torch
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('pointcloud_processor')

        # Initialize subscriber
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',  # Default RealSense pointcloud topic
            self.pointcloud_callback,
            10
        )

        # Initialize publisher
        self.pc_pub = self.create_publisher(
            PointCloud2,
            'processed_pointcloud',
            10
        )

        # Initialize Open3D device
        self.device = o3d.core.Device("CUDA:0" if torch.cuda.is_available() else "CPU:0")
        self.get_logger().info(f"Using device: {self.device}")

    def tensor_to_ros_cloud(self, pcd_tensor, frame_id="camera_link"):
        """Convert Open3D tensor pointcloud to ROS PointCloud2 message"""
        # Get points and colors from tensor (moved to CPU if necessary)
        points = pcd_tensor.point.positions.cpu().numpy()
        colors = pcd_tensor.point.colors.cpu().numpy()

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
        header.stamp = self.get_clock().now().to_msg()

        # Create and return PointCloud2 message
        return point_cloud2.create_cloud(header, fields, cloud_data)

    def pointcloud_callback(self, ros_cloud):
        try:
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
            cloud_array = point_cloud2.read_points_numpy(
                ros_cloud,
                field_names=('x', 'y', 'z', 'rgb'),
                skip_nans=True
            )

            # Extract XYZ points
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

            # Create Open3D tensor PointCloud directly
            pcd_tensor = o3d.t.geometry.PointCloud(self.device)

            # Convert numpy arrays to tensors and move to device
            pcd_tensor.point.positions = o3d.core.Tensor(
                    points_np,
                    dtype=o3d.core.Dtype.Float32,
                    device=self.device
            )

            pcd_tensor.point.colors = o3d.core.Tensor(
                    colors_np,
                    dtype=o3d.core.Dtype.Float32,
                    device=self.device
            )

            # Example GPU processing using tensor operations
            # 1. Estimate normals
            pcd_tensor.estimate_normals()

            # 2. Example: Remove statistical outliers
            pcd_tensor_cleaned, _ = pcd_tensor.remove_statistical_outliers(
                    nb_neighbors=20,
                    std_ratio=2.0
            )

            # # 3. Example: Compute covariance
            # covariance = pcd_tensor.compute_covariance()

            # Convert processed tensor pointcloud back to ROS message
            processed_cloud_msg = self.tensor_to_ros_cloud(
                pcd_tensor_cleaned,
                frame_id=ros_cloud.header.frame_id
            )

            # Publish processed pointcloud
            self.pc_pub.publish(processed_cloud_msg)

            # Log processing info
            self.get_logger().info(
                f"Published processed pointcloud with "
                f"{pcd_tensor_cleaned.point.positions.shape[0]} points"
            )

            # Here you can add more GPU-accelerated processing
            # The pointcloud is now in tensor format on the specified device

        except Exception as e:
            self.get_logger().error(f"Error processing pointcloud: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    processor = PointCloudProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()