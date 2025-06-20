"""
Concatenates and syncronizes multiple (n) point clouds into a single point cloud.
Can transform to a target frame, e.g robot frame, lidar frame, etc.
Should have a synchonization mode that uses message_filters or a robust mode that makes sure to publish even if some sensors fail.
"""