"""
Data compression utilities for FlyingCarRL.
Handles compression of trajectory data, sensor logs, and large JSON blobs.
"""

import json
import zlib
import base64
import pickle
import numpy as np
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class CompressionUtils:
    """
    Utilities for compressing and decompressing simulation data.

    Methods:
    - JSON compression (zlib)
    - Numpy array compression
    - Trajectory data compression
    - Sensor data compression
    """

    @staticmethod
    def compress_json(data: Union[Dict, list], level: int = 6) -> str:
        """
        Compress JSON-serializable data.

        Args:
            data: Dictionary or list to compress
            level: Compression level (1-9, higher = more compression)

        Returns:
            Base64-encoded compressed string
        """
        try:
            json_str = json.dumps(data)
            compressed = zlib.compress(json_str.encode('utf-8'), level=level)
            encoded = base64.b64encode(compressed).decode('utf-8')

            original_size = len(json_str)
            compressed_size = len(encoded)
            ratio = (1 - compressed_size / original_size) * 100

            logger.debug(
                f"Compressed JSON: {original_size} -> {compressed_size} bytes "
                f"({ratio:.1f}% reduction)"
            )

            return encoded

        except Exception as e:
            logger.error(f"Error compressing JSON: {e}")
            raise

    @staticmethod
    def decompress_json(compressed_str: str) -> Union[Dict, list]:
        """
        Decompress JSON data.

        Args:
            compressed_str: Base64-encoded compressed string

        Returns:
            Decompressed dictionary or list
        """
        try:
            decoded = base64.b64decode(compressed_str.encode('utf-8'))
            decompressed = zlib.decompress(decoded)
            data = json.loads(decompressed.decode('utf-8'))

            return data

        except Exception as e:
            logger.error(f"Error decompressing JSON: {e}")
            raise

    @staticmethod
    def compress_trajectory(
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, str]:
        """
        Compress trajectory data (positions, velocities, timestamps).

        Args:
            positions: Array of positions (N, 3)
            velocities: Array of velocities (N, 3)
            timestamps: Array of timestamps (N,)

        Returns:
            Dictionary with compressed data
        """
        try:
            # Convert to float32 for space savings
            positions_f32 = positions.astype(np.float32)
            velocities_f32 = velocities.astype(np.float32)
            timestamps_f32 = timestamps.astype(np.float32)

            # Compress each array
            pos_compressed = zlib.compress(positions_f32.tobytes())
            vel_compressed = zlib.compress(velocities_f32.tobytes())
            time_compressed = zlib.compress(timestamps_f32.tobytes())

            # Encode to base64
            compressed_data = {
                'positions': base64.b64encode(pos_compressed).decode('utf-8'),
                'velocities': base64.b64encode(vel_compressed).decode('utf-8'),
                'timestamps': base64.b64encode(time_compressed).decode('utf-8'),
                'shape': positions.shape,
                'dtype': 'float32'
            }

            original_size = (
                positions.nbytes + velocities.nbytes + timestamps.nbytes
            )
            compressed_size = (
                len(pos_compressed) + len(vel_compressed) + len(time_compressed)
            )
            ratio = (1 - compressed_size / original_size) * 100

            logger.debug(
                f"Compressed trajectory: {original_size} -> {compressed_size} bytes "
                f"({ratio:.1f}% reduction)"
            )

            return compressed_data

        except Exception as e:
            logger.error(f"Error compressing trajectory: {e}")
            raise

    @staticmethod
    def decompress_trajectory(compressed_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Decompress trajectory data.

        Args:
            compressed_data: Dictionary with compressed trajectory data

        Returns:
            Dictionary with positions, velocities, timestamps arrays
        """
        try:
            # Decode and decompress
            pos_bytes = zlib.decompress(
                base64.b64decode(compressed_data['positions'].encode('utf-8'))
            )
            vel_bytes = zlib.decompress(
                base64.b64decode(compressed_data['velocities'].encode('utf-8'))
            )
            time_bytes = zlib.decompress(
                base64.b64decode(compressed_data['timestamps'].encode('utf-8'))
            )

            # Reconstruct arrays
            shape = tuple(compressed_data['shape'])
            dtype = np.dtype(compressed_data['dtype'])

            positions = np.frombuffer(pos_bytes, dtype=dtype).reshape(shape)
            velocities = np.frombuffer(vel_bytes, dtype=dtype).reshape(shape)
            timestamps = np.frombuffer(time_bytes, dtype=dtype)

            return {
                'positions': positions,
                'velocities': velocities,
                'timestamps': timestamps
            }

        except Exception as e:
            logger.error(f"Error decompressing trajectory: {e}")
            raise

    @staticmethod
    def compress_sensor_data(sensor_data: Dict[str, Any]) -> str:
        """
        Compress sensor data (camera, lidar, etc.).

        Args:
            sensor_data: Dictionary with sensor data

        Returns:
            Base64-encoded compressed string
        """
        try:
            # Use pickle for complex data types (can include numpy arrays)
            pickled = pickle.dumps(sensor_data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(pickled, level=6)
            encoded = base64.b64encode(compressed).decode('utf-8')

            logger.debug(
                f"Compressed sensor data: {len(pickled)} -> {len(compressed)} bytes"
            )

            return encoded

        except Exception as e:
            logger.error(f"Error compressing sensor data: {e}")
            raise

    @staticmethod
    def decompress_sensor_data(compressed_str: str) -> Dict[str, Any]:
        """
        Decompress sensor data.

        Args:
            compressed_str: Base64-encoded compressed string

        Returns:
            Dictionary with sensor data
        """
        try:
            decoded = base64.b64decode(compressed_str.encode('utf-8'))
            decompressed = zlib.decompress(decoded)
            sensor_data = pickle.loads(decompressed)

            return sensor_data

        except Exception as e:
            logger.error(f"Error decompressing sensor data: {e}")
            raise

    @staticmethod
    def estimate_compression_ratio(data: Any) -> float:
        """
        Estimate compression ratio for data.

        Args:
            data: Data to estimate

        Returns:
            Estimated compression ratio (0-1, lower = better compression)
        """
        try:
            if isinstance(data, (dict, list)):
                json_str = json.dumps(data)
                original_size = len(json_str.encode('utf-8'))
                compressed_size = len(zlib.compress(json_str.encode('utf-8')))

            elif isinstance(data, np.ndarray):
                original_size = data.nbytes
                compressed_size = len(zlib.compress(data.tobytes()))

            else:
                pickled = pickle.dumps(data)
                original_size = len(pickled)
                compressed_size = len(zlib.compress(pickled))

            ratio = compressed_size / original_size
            return ratio

        except Exception as e:
            logger.warning(f"Error estimating compression ratio: {e}")
            return 1.0  # No compression
