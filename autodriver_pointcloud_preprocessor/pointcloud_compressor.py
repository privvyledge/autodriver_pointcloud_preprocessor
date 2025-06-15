"""
See:
    * https://github.com/ros-perception/point_cloud_transport/tree/humble?tab=readme-ov-file#c
    * https://github.com/ros-perception/point_cloud_transport/tree/humble/point_cloud_transport_py
"""

# import point_cloud_transport_py as pct
# from point_cloud_interfaces.msg import CompressedPointCloud2
# from rclpy.serialization import serialize_message
# ...
#
# pct_codec = pct.PointCloudCodec()
# pct_vec_str = pct.VectorString()
# pct_names_vec_str = pct.VectorString()
# codec.getLoadableTransports(self._pct_vec_str, self._pct_names_vec_str)
# # Verified: Draco Transport is available
# ...
#
# # load compressed_pointcloud2_msg from bag
# # Verified: compressed_pointcloud2_msg properly loaded
#
# compressed_pointcloud2_buffer = serialize_message(compressed_pointcloud2_msg)
# pointcloud2_buffer = pct_codec.decode(compressed_pointcloud2_msg.format, bytes(compressed_pointcloud2_buffer))