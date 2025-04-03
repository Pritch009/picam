import os
import numpy as np

use_mock_camera = os.environ.get('USE_MOCK_CAMERA', 'False').lower() == 'true'

if use_mock_camera:
    from mock_camera import MockCamera as PiCamera
    print("Using MockCamera")
else:
    try:
        from picamera2 import Picamera2
        print("Using Raspberry Pi camera")
        PiCamera = Picamera2
    except ImportError:
        print("picamera not found.  Using MockCamera.  Set environment variable USE_MOCK_CAMERA=TRUE to suppress this message.")
        from mock_camera import MockCamera as PiCamera

class Camera:
    def __init__(self, resolution=(640, 480)):
        from libcamera import ColorSpace, Transform
        self.camera = PiCamera()
        sensor = None
        for sensor in self.camera.sensor_modes:
            if sensor["size"][0] == resolution[0] and sensor["size"][1] == resolution[1]:
                break
        if sensor is None:
            raise ValueError(f"No sensor mode found for resolution {resolution}")
        config = {
            "size": resolution, 
            "format": "XRGB8888", 
            "colour_space": ColorSpace.Srgb(),
            "exposure_mode": "auto",
            "main": { 
                "format": "XRGB8888",
                "size": resolution,
            },
            "lores": {
                "size": (640, 480), 
                "format": "YUV420",
            },
            "raw": {
                "size": (640, 480),
                "format": "SRGGB10",
            },
            "transform": Transform(),
            "buffer_count": 2,
            "display": None,
            "sensor": {
                "output_size": sensor["size"],
                "bit_depth": sensor["bit_depth"],
            }
        }
        self.camera.configure(config)

    def start_feed(self):
        # PiCamera2 starts automatically, so this is not needed
        if not self.is_running:
            self.is_running = True # Assume running after initialization
            self.camera.start()

    def stop_feed(self):
        # PiCamera2 does not have stop_preview. Stopping the camera instead.
        if self.is_running:
            self.camera.stop()
            self.is_running = False

    def capture_frame(self):
        frame = self.camera.capture_array()
        if frame is None:
            print("Error capturing frame")
            # Create a dummy image (e.g., a black image) as fallback
            frame = np.zeros((self.camera.configuration()["size"][1], self.camera.configuration()["size"][0], 3), dtype=np.uint8)

        return frame

    def close(self):
        self.camera.close()