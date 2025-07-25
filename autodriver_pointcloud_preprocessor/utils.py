"""
"""
import time
from typing import Any
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

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
    import open3d.t.geometry as t
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

VENDOR_MAPPINGS = {
            "intensity": ["I", "intensity"],
            "ring": ["C", "ring", "line"],  # Autoware, Velodyne, Livox
            "time": ["t", "time", "timestamp"],  # , Autoware/velodyne, Livox
            "return_type": ["return_type", "tag", "R"],  # Velodyne, Livox, Autoware
            "azimuth": ["azimuth"],
            "distance": ["distance", "depth", "d"],
        }


def convert_pointcloud_to_numpy(structured_cloud_array, metadata_dict):
    has_rgb = metadata_dict.get('has_rgb', False)
    has_intensity = metadata_dict.get('has_intensity', False)
    has_ring = metadata_dict.get('has_ring', False)
    has_time = metadata_dict.get('has_time', False)
    has_return_type = metadata_dict.get('has_return_type', False)

    intensity_field_name = metadata_dict.get('intensity_field_name', None)
    ring_field_name = metadata_dict.get('ring_field_name', None)
    time_field_name = metadata_dict.get('time_field_name', None)
    return_type_field_name = metadata_dict.get('return_type_field_name', None)

    field_names = metadata_dict.get('field_names', ['x', 'y', 'z'])
    num_fields = metadata_dict.get('num_fields', -1)

    """
    positions = []
    rgb = []
    intensity = []
    ring = []
    time = []
    return_type = []

    for point_idx in range(structured_cloud_array.size):
        positions_ = []
        for field_idx in range(num_fields):
            if cloud_field_names[field_idx].lower() in 'xyz':
                positions_.append(structured_cloud_array[point_idx][field_idx])
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
        positions.append(positions_)

    positions_arr = np.array(positions)
    rgb_arr = np.array(rgb)
    intensity_arr = np.array(intensity)
    ring_arr = np.array(ring)
    time_arr = np.array(time)  
    return_type_array = np.array(return_type)      
    """

    positions_arr = np.vstack(
        (structured_cloud_array["x"], structured_cloud_array["y"], structured_cloud_array["z"])
    ).T.astype(np.float32)

    pointcloud_dictionary = {
        'positions': positions_arr,
    }

    if has_rgb:
        if {"r", "g", "b"}.issubset(field_names):
            # separate 'r', 'g', 'b' bytes and fields in cloud_field_names
            rgb_arr = merge_rgb_fields(structured_cloud_array["r"],
                                       structured_cloud_array["g"],
                                       structured_cloud_array["b"], return_int=True)
        else:
            # packed 'rgb' bytes and fields in cloud_field_names. The input can be from above if return_int=Fasle
            rgb_arr = extract_rgb_from_pointcloud(structured_cloud_array["rgb"].astype(np.float32))
        pointcloud_dictionary['rgb'] = rgb_arr
    if has_intensity:
        intensity_arr = structured_cloud_array[intensity_field_name].astype(np.float32)
        pointcloud_dictionary['intensity'] = intensity_arr
    if has_ring:
        ring_arr = structured_cloud_array[ring_field_name].astype(np.uint16)
        pointcloud_dictionary['ring'] = ring_arr
    if has_time:
        time_arr = structured_cloud_array[time_field_name].astype(np.float64)
        pointcloud_dictionary['time'] = time_arr
    if has_return_type:
        return_type_arr = structured_cloud_array[return_type_field_name].astype(np.uint8)
        pointcloud_dictionary['return_type'] = return_type_arr

    return pointcloud_dictionary

def dict_to_open3d_tensor_pointcloud(pointcloud_dict, device="CPU:0"):
    pointcloud = t.PointCloud(pointcloud_dict).to(device)
    return pointcloud


def numpy_struct_to_pointcloud2(field_names: list,
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

    # # Now flatten the data into a single contiguous buffer
    # # If points is structured dtype, we can view it as raw bytes per record:
    # if isinstance(points.dtype, np.dtype) and points.dtype.names is not None:
    #     # Ensure the fields in points.dtype.names exactly match field_names
    #     # (up to ordering). We'll reorder/cast if necessary.
    #     reorder_needed = list(points.dtype.names) != field_names
    #     if reorder_needed:
    #         # Create a new array with exactly field_names order
    #         new_arr = np.zeros(points.shape, dtype=np.dtype([(n, points.dtype.fields[n][0]) for n in field_names]))
    #         for n in field_names:
    #             new_arr[n] = points[n]
    #         points = new_arr
    #
    #     # Now raw bytes:
    #     data_bytes = points.tobytes(order="C")
    #
    # else:
    #     # If points is a (N, M) float32/float64 array or something:
    #     # We assume that the user knows to pass the correct flat array form.
    #     data_bytes = points.tobytes(order="C")

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


def pointcloud_to_dict(ros_cloud, field_names=None, skip_nans=True, organize_cloud=False, metadata_dict=None):
    if not metadata_dict:
        metadata_dict = {}
    metadata_dict.update({'header': ros_cloud.header, 'field_names': None})
    cloud_array = point_cloud2.read_points(
        ros_cloud,
        field_names=field_names,
        skip_nans=skip_nans,
        reshape_organized_cloud=organize_cloud
    )
    cloud_dict = dict()
    metadata_dict['field_names'] = cloud_array.dtype.names  # the field names extracted
    metadata_dict['num_fields'] = len(metadata_dict['field_names'])

    # Optional: Extract additional attributes if present.
    if not metadata_dict.get('has_intensity', False):
        field_mapping_dict = get_pointcloud_metadata(metadata_dict['field_names'])
        metadata_dict.update(field_mapping_dict)

    cloud_dict.update(
        convert_pointcloud_to_numpy(cloud_array, metadata_dict))
    return cloud_dict, metadata_dict


def check_field(field, pointcloud_dict, metadata_dict):
    if pointcloud_dict.get(field, None) is not None or metadata_dict.get(f'has_{field}', None):
        return True
    return False

def get_fields_from_dicts(key, pointcloud_dict, metadata_dict):
    key_tensor = None
    if check_field(key, pointcloud_dict, metadata_dict):
        key_tensor =  o3c.Tensor.from_numpy(pointcloud_dict[key].reshape(-1, 1))  # reshape might not be needed
    if key_tensor is not None:
        pointcloud_dict[key] = key_tensor
    return pointcloud_dict


def crop_pointcloud(pointcloud, backend='open3d', min_bound=None, max_bound=None, invert=False, aabb=None):
    """
    Crop the pointcloud based on min and max bounds.
    Todo: check pointcloud instance first, i.e Open3D.t.geometry.PointCloud, numpy, torch then convert
    :param pointcloud: Open3D.t.geometry.PointCloud object or dictionary containing pointcloud data.
    :param backend:
    :param min_bound: Minimum bound for cropping (3D vector).
    :param max_bound: Maximum bound for cropping (3D vector).
    :param invert: If True, crops outside the bounds instead of inside.
    :param aabb: Optional Axis-Aligned Bounding Box (AABB) to crop the pointcloud.
    :return: Cropped pointcloud object or dictionary.
    """
    msg = ''
    # Depending on how numpy/torch/open3d are installed/built, one backend may be preferred
    if backend.lower() in ['np', 'numpy']:
        points = pointcloud.point.positions  # pointcloud.point.positions.cpu().numpy()
        # optional: transfer to cpu then numpy first
        if not points.is_cpu:
            msg = f'{msg} Transferring points to cpu...'
            points = points.cpu()

        if not isinstance(points, np.ndarray):
            msg = f'{msg} Converting points to numpy...'
            points = points.numpy()

        # use numpy operations since they can be significantly faster than Open3D on CPU
        if invert:
            crop_mask = np.any((points <= min_bound) | (points >= max_bound), axis=1)
        else:
            crop_mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)

        pointcloud = pointcloud.select_by_mask(o3c.Tensor.from_numpy(crop_mask).to(pointcloud.device))
    elif backend.lower() in ['torch', 'pytorch']:
        # torch alternative
        points = torch_from_dlpack(pointcloud.point.positions.to_dlpack())
        min_bound = torch.as_tensor(min_bound, device=points.device)
        max_bound = torch.as_tensor(max_bound, device=points.device)
        if invert:
            crop_mask = torch.any((points <= min_bound) | (points >= max_bound), dim=1)
        else:
            # crop_mask = torch.any((points >= min_bound) & (points <= max_bound), dim=1)
            crop_mask = torch.all((points >= min_bound) & (points <= max_bound), dim=1)
            # # since this is a sequence of bitwise operations, should be fast for numpy, torch and open3d backends
            # crop_mask = (
            #         (self.o3d_pointcloud.point.positions[:, 0] >= self.roi_min[0]) &
            #         (self.o3d_pointcloud.point.positions[:, 0] <= self.roi_max[0]) &
            #         (self.o3d_pointcloud.point.positions[:, 1] >= self.roi_min[1]) &
            #         (self.o3d_pointcloud.point.positions[:, 1] <= self.roi_max[1]) &
            #         (self.o3d_pointcloud.point.positions[:, 2] >= self.roi_min[2]) &
            #         (self.o3d_pointcloud.point.positions[:, 2] <= self.roi_max[2])
            # )
        # we convert the boolean array to an integer array since dlpack does not support zero-copy transfer for bool
        crop_mask = crop_mask.to(dtype=torch.uint8)
        # transfer to Open3D
        crop_mask = o3c.Tensor.from_dlpack(torch_to_dlpack(crop_mask))
        # convert back to a boolean mask
        crop_mask = crop_mask.to(o3c.Dtype.Bool)
        pointcloud = pointcloud.select_by_mask(crop_mask)
    else:
        pointcloud = pointcloud.crop(aabb, invert=invert)
        msg = f'{msg} Using Open3D pointcloud.crop()'
    return pointcloud, msg


def merge_rgb_fields(r, g, b, return_int=False):
    """
    Takes separate r, g, b fields of type np.uint8 and returns a merged rgb array of type np.float32 or np.uint8.
    PCL typically outputs separate r, g, b fields.
    Source: https://docs.ros.org/en/kinetic/api/ros_numpy/html/namespaceros__numpy_1_1point__cloud2.html#af3d3551aaadd53513bb382aa8092fe4b
    """
    if return_int:
        r = r.astype(np.uint8)  # np.asarray(cloud_arr['r'], dtype=np.uint8)
        g = g.astype(np.uint8)
        b = b.astype(np.uint8)
        rgb_arr = np.vstack((r, g, b)).T
    else:
        r = r.astype(np.uint32)  # np.asarray(cloud_arr['r'], dtype=np.uint32)
        g = g.astype(np.uint32)
        b = b.astype(np.uint32)
        # todo: test setting dtype=np.float32 vs using astype vs .view(np.float32)
        rgb_arr = np.array((r << 16) | (g << 8) | (b << 0)).view(np.float32)

    return rgb_arr

def extract_rgb_from_pointcloud(rgb):
    """
    Extracts a packed 'rgb' float32 array/field and returns an RGB array of type uint8.
    """
    # Many ROS drivers, such as RealSense and Zed pack RGB as a float32 where bytes are [R,G,B,0].
    # We reinterpret the float as a uint32, then bit-shift.
    rgb_floats = rgb  # self.pointcloud_dictionary['cloud_array']["rgb"].astype(np.float32)
    rgb_bytes = rgb_floats.view(np.uint32)  # rgb_floats.dtype = np.uint32
    # Extract RGB channels
    r = ((rgb_bytes >> 16) & 0xFF).astype(np.uint8)  # np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = ((rgb_bytes >> 8) & 0xFF).astype(np.uint8)
    b = (rgb_bytes & 0xFF).astype(np.uint8)

    # Stack RGB channels
    # rgb_arr = np.zeros(rgb_floats.shape, dtype=np.uint8)  # todo: use zeros and replace
    # rgb_array[:, :] = r, g, b
    rgb_arr = np.vstack((r, g, b)).T.astype(np.uint8)

    # r = ((rgb_bytes >> 16) & 0xFF).astype(np.float32) / 255.0
    # g = ((rgb_bytes >> 8) & 0xFF).astype(np.float32) / 255.0
    # b = (rgb_bytes & 0xFF).astype(np.float32) / 255.0
    return rgb_arr

def rgb_int_to_float(rgb_np):
    # Convert colors from float [0,1] to uint32 packed RGB. todo: refactor
    colors_u8 = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    r_u = colors_u8[:, 0].astype(np.uint32)
    g_u = colors_u8[:, 1].astype(np.uint32)
    b_u = colors_u8[:, 2].astype(np.uint32)

    rgb_uint32 = (r_u << 16) | (g_u << 8) | b_u  # np.left_shift(r_u, 16)
    rgb_float32 = rgb_uint32.view(np.float32)
    return rgb_float32

def rgb_to_intensity(color):
    # todo: research how to convert intensity to RGB correctly
    # Convert intensity to RGB
    rgb = np.asarray(color)  # [N,3] in [0,1]
    intensity = (0.2126 * rgb[:,0] + 0.7152 * rgb[:,1] + 0.0722 * rgb[:,2]).astype(np.float32)

    # # # Faster vectorized method
    # weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb.device, dtype=rgb.dtype)
    # intensity = (rgb * weights).sum(dim=1)
    return intensity


def intensity_to_rgb(intensity):
    # todo: research how to convert intensity to RGB correctly
    # todo: check the type and use the right backend
    # todo: check http://wiki.ros.org/rviz/DisplayTypes/PointCloud
    # Convert intensity to RGB
    intensity = intensity.astype(np.float32)  # shape (N,)
    # # Method 1:
    # Normalize to [0,1]
    i_min, i_max = intensity.min(), intensity.max()  # 0.0, 255.0
    i_norm = (intensity - i_min) / max(i_max - i_min, 1e-6)
    # Replicate into three channels
    rgb = np.stack([i_norm, i_norm, i_norm], axis=1).astype(np.float32)  # [N,3]

    # # # Method 2:
    # i_min, i_max = intensity.min(), intensity.max()  # 0.0, 255.0
    # ratio = 2 # 2 is a magic number, can be adjusted
    # b = 255 * (1 - ratio)  # 255 * (1 - ratio * (intensity - i_min) / (i_max - i_min))
    # b[b < 0] = 0
    # b = b.astype(int)
    #
    # r = 255 * (ratio - 1)  # 255 * (ratio * (intensity - i_min) / (i_max - i_min) - 1)
    # r[r < 0] = 0
    # r = r.astype(int)
    #
    # g = 255 - b - r
    #
    # rgb = np.stack([r, g, b], axis=1).astype(np.float32)

    # # # Method 3:
    # VIRIDIS = np.array(colormap.get_cmap('plasma').colors)
    # VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
    # intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    # int_color = np.c_[
    #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
    #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
    #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
    # rgb = int_color.astype(np.float32)  # [N,3]

    # # Method 4: Using matplotlib colormap
    # colors_map = plt.get_cmap('tab20', 256)
    # source_attribute = intensity / 255.0
    # source_colors = colors_map(source_attribute)
    # rgb = source_colors[:, :3]

    # import matplotlib.cm as cm
    # cmap = cm.get_cmap("jet")
    # rgb = cmap(i_norm)[:, :3].astype(np.float32)  # drop alpha
    # pcd.point.colors = o3d.core.Tensor(rgb, device=self.device)

    # Upload as colors:
    color = o3d.core.Tensor(rgb)  # todo: get device from intensity or just use Open3D/Torch operations
    return color

def parse_differing_fields(options, field_names):
    """
    Takes in either a list of options or an option and checks if it exists in the field_names.
    Used to handle different field names from various LIDAR vendors/manufacturers/drivers.
    """
    if isinstance(options, str):
        options = [options]

    # return any(option.lower() in field_names for option in options), None
    option_in_field_names = []
    corresponding_field_name = None
    for option in options:
        if option.lower() in field_names:
            option_in_field_names.append(option)
            corresponding_field_name = option
    return any(option_in_field_names), corresponding_field_name


def get_pointcloud_metadata(field_names, vendor_mappings: dict =None):
    # todo: rethink and refactor the logic here and parse_differing_fields
    # todo: could split corresponding field name mapping to dictionary
    # Todo: let the user add custom options, vendor mappings and pointcloud_metadata keys
    # todo: add support for all field_names, i.e parse the non-standard keys
    if vendor_mappings is None:
        vendor_mappings = VENDOR_MAPPINGS
    field_names = [field_name.lower() for field_name in field_names]

    if {"r", "g", "b"}.issubset(field_names):
        has_rgb = True  # separate fields per channel
        rgb_field_name = ['r', 'g', 'b']
    else:
        has_rgb, rgb_field_name = parse_differing_fields("rgb", field_names)   # r, g, b or rgb

    has_intensity, intensity_field_name = parse_differing_fields(vendor_mappings["intensity"], field_names)
    has_ring, ring_field_name = parse_differing_fields(vendor_mappings["ring"], field_names)
    has_time, time_field_name = parse_differing_fields(vendor_mappings["time"], field_names)
    has_return_type, return_type_field_name = parse_differing_fields(vendor_mappings["return_type"], field_names)

    pointcloud_metadata = {
        'has_rgb': has_rgb,
        'has_intensity': has_intensity,
        'intensity_field_name': intensity_field_name,
        'has_ring': has_ring,
        'ring_field_name': ring_field_name,
        'has_time': has_time,
        'time_field_name': time_field_name,
        'has_return_type': has_return_type,
        'return_type_field_name': return_type_field_name,
    }
    return pointcloud_metadata

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


def structured_numpy_array_to_open3d_tensor_pointcloud(structured_numpy_array):
    pointcloud = o3d.t.geometry.PointCloud()
    pointcloud.points = o3c.Tensor.from_numpy(structured_numpy_array["positions"])
    return pointcloud


def remove_duplicates(pointcloud, backend='torch'):
    """
    Todo:
        * check pointcloud instance first, i.e Open3D.t.geometry.PointCloud, numpy, torch then convert
        * implement for loop to remove duplicates without sorting since numpy and torch unique sorts by default regardless of sorted=False
    :param pointcloud: Open3D.t.geometry.PointCloud object
    :param backend:
    :return:
    """
    msg = ''
    # Depending on how numpy/torch/open3d are installed/built, one backend may be preferred
    if backend.lower() in ['np', 'numpy']:
        points = pointcloud.point.positions  # pointcloud.point.positions.cpu().numpy()
        # optional: transfer to cpu then numpy first
        if not points.is_cpu:
            msg = f'{msg} Transferring points to cpu...'
            points = points.cpu()

        if not isinstance(points, np.ndarray):
            msg = f'{msg} Converting points to numpy...'
            points = points.numpy()

        # use numpy operations since they can be significantly faster than Open3D on CPU
        np_pointcloud, duplicates_indices = np.unique(points,
                                                      axis=0, return_index=True, sorted=False)
        pointcloud = pointcloud.select_by_index(o3c.Tensor.from_numpy(duplicates_indices).to(pointcloud.device))
    elif backend.lower() in ['torch', 'pytorch']:
        # torch alternative. Fastest by default
        points = torch_from_dlpack(pointcloud.point.positions.to_dlpack())
        torch_pointcloud, duplicates_indices = torch.unique(
                points, dim=0, return_inverse=True,
                sorted=False)  # torch.unique is slower due to sorting but unique_consecutive is does not remove all duplicates only consecutive ones.
        pointcloud = pointcloud.select_by_index(
                o3c.Tensor.from_dlpack(torch_to_dlpack(duplicates_indices)))
    else:
        pointcloud, duplicates_mask = pointcloud.remove_duplicated_points()
        # pointcloud = pointcloud.select_by_mask(duplicates_mask)
        msg = f'{msg} Using Open3D pointcloud.remove_duplicated_points()'
    return pointcloud, msg