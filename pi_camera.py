import cv2
from libcamera import ColorSpace, Transform
from picamera2 import Picamera2 as PiCamera

class Camera:
    def __init__(self, resolution=(1920, 1080)):
        self.camera = PiCamera()
        self.resolution = resolution
        self.is_running = False
        self.configure()
    
    def configure(self):
        sensor = None
        for sensor in self.camera.sensor_modes:
            if sensor["size"][0] == self.resolution[0] and sensor["size"][1] == self.resolution[1]:
                break
        if sensor is None:
            raise ValueError(f"No sensor mode found for resolution {self.resolution}")
        self.config = {
            "size": self.resolution, 
            "main": { 
                "format": "XRGB8888",
                "size": (1920, 1080),
            },
            "lores": {
                "format": "YUV420",
                "size": (640, 480), 
            },
            "raw": {
                "format": "SRGGB10",
                "size": (1920, 1080),
            },
            "transform": Transform(),
            "buffer_count": 2,
            "display": None,
            "encode": None,
            "queue": True,
            "sensor": {},
            "controls": {},
        }
        self.camera.configure(self.config)

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
        frame = self.camera.capture_array("main")
        if frame is None:
            print("Error capturing frame")
            # Create a dummy image (e.g., a black image) as fallback
            return None
                    
        # Convert the frame to BGR format (OpenCV uses BGR)
        # return frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert from RGB to BGR
        return frame

    def close(self):
        self.camera.close()