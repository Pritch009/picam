import cv2
from threading import Lock
from libcamera import ColorSpace, Transform
from picamera2 import Picamera2 as PiCamera

class Camera:
    def __init__(self, resolution=(1920, 1080)):
        self.camera = PiCamera()
        self.resolution = resolution
        self.is_running = False
        self.lock = Lock()
        self.configure()
    
    def configure(self):
        self.config = self.camera.create_video_configuration(
            main={"size": self.resolution, "format": "XRGB8888"},
            lores={"size": (640, 480), "format": "YUV420"},
            raw={"size": (1920, 1080), "format": "SRGGB10"},
        )
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

    def lock(self):
        self.lock.acquire()
        print("Camera lock acquired")

    def release(self):
        self.lock.release()
        print("Camera lock released")

    def capture_frame(self, camera="main"):
        frame = None
        match camera:
            case "main":
                frame = self.camera.capture_array("main")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            case "lores":
                frame = self.camera.capture_array("lores")
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            case "raw":
                frame = self.camera.capture_array("raw")
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)
        return frame

    def close(self):
        self.camera.close()