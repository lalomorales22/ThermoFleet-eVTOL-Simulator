"""
Sensor module for FlyingCarRL

Provides camera, LiDAR, and IMU sensor emulation for eVTOL vehicles.
"""

from .camera import CameraSensor, CameraConfig
from .lidar import LiDARSensor, LiDARConfig
from .imu import IMUSensor, IMUConfig

__all__ = [
    # Camera
    'CameraSensor',
    'CameraConfig',

    # LiDAR
    'LiDARSensor',
    'LiDARConfig',

    # IMU
    'IMUSensor',
    'IMUConfig',
]
