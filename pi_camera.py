from io import BytesIO
import os

import cv2
import numpy as np

use_mock_camera = os.environ.get('USE_MOCK_CAMERA', 'False').lower() == 'true'

if use_mock_camera:
    from mock_camera import MockCamera as PiCamera
    print("Using MockCamera")
else:
    try:
        from picamera2 import PiCamera
        print("Using Raspberry Pi camera")
    except ImportError:
        print("picamera not found.  Using MockCamera.  Set environment variable USE_MOCK_CAMERA=TRUE to suppress this message.")
        from mock_camera import MockCamera as PiCamera

class Camera:
    def __init__(self, resolution=(640, 480)):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.is_running = False

    def start_feed(self):
        if not self.is_running:
            self.camera.start_preview()
            self.is_running = True

    def stop_feed(self):
        if self.is_running:
            self.camera.stop_preview()
            self.is_running = False

    def capture_frame(self):
        stream = BytesIO()
        self.camera.capture(stream, format='jpeg')
        stream.seek(0)
        frame_data = stream.read()

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print("Error decoding frame")
            # Create a dummy image (e.g., a black image) as fallback
            frame = np.zeros((self.camera.resolution[1], self.camera.resolution[0], 3), dtype=np.uint8)

        return frame

    def close(self):
        self.camera.close()