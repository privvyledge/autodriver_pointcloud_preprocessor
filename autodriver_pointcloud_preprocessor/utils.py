"""

Todo:
    * ros pointcloud to dict
    * dict to open3d (Open3D accepts initialization from dict so just profile)
    * ros pointcloud to open3d pointcloud
    * handle individual R, G, B or combined  (https://docs.ros.org/en/jade/api/ros_numpy/html/namespaceros__numpy_1_1point__cloud2.html#a83fe725ae892944ced7d5283d5e1c643)
"""
import time
from typing import Any
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

try:
    import torch
    from torch.utils.dlpack import from_dlpack as torch_from_dlpack
    from torch.utils.dlpack import to_dlpack as torch_to_dlpack
except ImportError:
    torch = None
    torch_from_dlpack = None
    torch_to_dlpack = None
    print("Torch not installed")

try:
    import open3d as o3d
    import open3d.core as o3c
except ImportError:
    o3d = None
    o3c = None
    print("Open3D not installed")

FIELD_DTYPE_MAP = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}

FIELD_DTYPE_MAP_INV = {v: k for k, v in FIELD_DTYPE_MAP.items()}


def convert_pointcloud_to_numpy(structured_cloud_array, cloud_field_names=None, num_fields=None, field_names=None):
    # todo: handle different field names from different vendors
    # todo: unpack to Open3D tensor directly
    if cloud_field_names is None:
        cloud_field_names = structured_cloud_array.dtype.names

    if num_fields is None:
        num_fields = len(cloud_field_names)

    # Optional: Extract additional attributes if present
    has_rgb = "rgb" in cloud_field_names
    has_intensity = "intensity" in cloud_field_names
    has_ring = "ring" in cloud_field_names
    has_time = "time" in cloud_field_names
    has_return_type = "return_type" in cloud_field_names

    """
    xyz = []
    rgb = []
    intensity = []
    ring = []
    time = []
    return_type = []

    for point_idx in range(structured_cloud_array.size):
        xyz_ = []
        for field_idx in range(num_fields):
            if cloud_field_names[field_idx].lower() in 'xyz':
                xyz_.append(structured_cloud_array[point_idx][field_idx])
            elif cloud_field_names[field_idx].lower() in ('i', 'intensity'):
                intensity.append(structured_cloud_array[point_idx][field_idx])
            elif cloud_field_names[field_idx].lower() in ('c', 'ring', 'line'):
                # livox uses 'line', autoware uses 'C', velodyne uses 'ring'
                ring.append(structured_cloud_array[point_idx][field_idx])
            elif cloud_field_names[field_idx].lower() in ('t', 'time', 'timestamp'):
                # livox uses 'timestamp', Autoware uses 'time
                time.append(structured_cloud_array[point_idx][field_idx])
            elif cloud_field_names[field_idx].lower() == 'rgb':
                rgb.append(structured_cloud_array[point_idx][field_idx])
            elif cloud_field_names[field_idx].lower() in ('tag', 'r'):
                # livox uses 'tag', autoware uses 'R'
                return_type.append(structured_cloud_array[point_idx][field_idx])
        xyz.append(xyz_)

    xyz_arr = np.array(xyz)
    rgb_arr = np.array(rgb)
    intensity_arr = np.array(intensity)
    ring_arr = np.array(ring)
    time_arr = np.array(time)  
    return_type_array = np.array(return_type)      
    """

    # Test
    xyz_arr = np.vstack(
        (structured_cloud_array["x"], structured_cloud_array["y"], structured_cloud_array["z"])
    ).T.astype(np.float32)
    rgb_arr = structured_cloud_array["rgb"].astype(np.float32) if has_rgb else None
    intensity_arr = structured_cloud_array["intensity"].astype(np.float32) if has_intensity else None
    ring_arr = structured_cloud_array["ring"].astype(np.uint16) if has_ring else None
    time_arr = structured_cloud_array["time"].astype(np.float64) if has_time else None
    return_type_arr = structured_cloud_array["return_type"].astype(np.uint8) if has_return_type else None

    pointcloud_dictionary = {
        'xyz': xyz_arr,
        'rgb': rgb_arr,
        'intensity': intensity_arr,
        'ring': ring_arr,
        'time': time_arr,
        'return_type': return_type_arr,
    }
    return pointcloud_dictionary


def numpy_struct_to_pointcloud2(points: np.ndarray,
                                field_names: list,
                                field_datatypes: list, is_dense: bool = True) -> tuple[list[Any], int | Any]:
    """
        Convert a NumPy structured (or regular) array back into a sensor_msgs/PointCloud2.
        - points: either a structured array (with named columns) or a regular NxM array.
        - header: std_msgs/Header (should carry frame_id and stamp).
        - field_names: list of field names to publish, in order.
        - field_datatypes: list of ROS PointField datatypes (e.g. PointField.FLOAT32, etc.),
          exactly matching the order of field_names.
        """
    # Build the list of PointField definitions
    fields = []
    offset = 0
    for name, datatype in zip(field_names, field_datatypes):
        # Determine the byte size of this datatype
        np_dt = FIELD_DTYPE_MAP[datatype]
        byte_size = np.dtype(np_dt).itemsize
        pf = PointField()
        pf.name = name
        pf.offset = offset
        pf.datatype = datatype
        pf.count = 1
        fields.append(pf)
        offset += byte_size

    # Now flatten the data into a single contiguous buffer
    # If points is structured dtype, we can view it as raw bytes per record:
    if isinstance(points.dtype, np.dtype) and points.dtype.names is not None:
        # Ensure the fields in points.dtype.names exactly match field_names
        # (up to ordering). We'll reorder/cast if necessary.
        reorder_needed = list(points.dtype.names) != field_names
        if reorder_needed:
            # Create a new array with exactly field_names order
            new_arr = np.zeros(points.shape, dtype=np.dtype([(n, points.dtype.fields[n][0]) for n in field_names]))
            for n in field_names:
                new_arr[n] = points[n]
            points = new_arr

        # Now raw bytes:
        data_bytes = points.tobytes(order="C")

    else:
        # If points is a (N, M) float32/float64 array or something:
        # We assume that the user knows to pass the correct flat array form.
        data_bytes = points.tobytes(order="C")

    # # Fill in PointCloud2 fields
    # width = points.shape[0]
    # cloud_msg = PointCloud2()
    # cloud_msg.header = header
    # cloud_msg.height = 1
    # cloud_msg.width = width
    # cloud_msg.fields = fields
    # cloud_msg.is_bigendian = sys.byteorder != 'little'
    # cloud_msg.point_step = offset
    # cloud_msg.row_step = offset * width
    # cloud_msg.is_dense = is_dense  # assume no invalid/NaN points
    # cloud_msg.data = data_bytes  # this line causes performance issues and is very slow.

    return fields, offset


def get_current_time(monotonic=True):
    """
    Reference function to make switching time sources as easy as overriding the time returned.
    Can be overridden, e.g., ROS clock.
    :param monotonic: If true, returns values that are guaranteed to monotonically increase.
    :return:
    """
    if not monotonic:
        return time.time()
    return time.perf_counter()  # time.perf_counter() or time.monotonic()


def get_time_difference(start_time, end_time, return_absolute_difference=False):
    """
    Reference implementation for time difference calculation.
    Can be overridden, e.g., ROS clock message or ROS Time object

    :param start_time:
    :param end_time:
    :param return_absolute_difference: If true, returns absolute difference so the order of start/end time
                                        arguments are irrelevant.
    :return:
    """
    time_difference = end_time - start_time
    if return_absolute_difference:
        return abs(end_time - start_time)
    return time_difference


def remove_duplicates(pointcloud, backend='torch'):
    """
    Todo: check pointcloud instance first, i.e Open3D.t.geometry.PointCloud, numpy, torch then convert
    :param pointcloud: Open3D.t.geometry.PointCloud object
    :param backend:
    :return:
    """
    # Depending on how numpy/torch/open3d are installed/built, one backend may be preferred
    if backend.lower() == 'numpy':
        points = pointcloud.point.positions  # pointcloud.point.positions.cpu().numpy()
        # optional: transfer to cpu then numpy first
        if not points.is_cpu:
            points = points.cpu()

        points = points.numpy()

        # use numpy operations since they can be significantly faster than Open3D on CPU
        np_pointcloud, duplicates_indices = np.unique(points,
                                                      axis=0, return_index=True)
        pointcloud = pointcloud.select_by_index(o3c.Tensor.from_numpy(duplicates_indices).to(pointcloud.device))
    elif backend.lower() == 'torch':
        # torch alternative. Fastest by default
        points = torch_from_dlpack(pointcloud.point.positions.to_dlpack())
        torch_pointcloud, duplicates_indices = torch.unique_consecutive(
                points, dim=0,
                return_inverse=True)  # torch.unique is slower due to sorting
        pointcloud = pointcloud.select_by_index(
                o3c.Tensor.from_dlpack(torch_to_dlpack(duplicates_indices)))
    else:
        pointcloud, duplicates_mask = pointcloud.remove_duplicated_points()
    return pointcloud