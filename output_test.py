import time
import cv2
from rich_camera import RichCamera as Camera
from PIL import Image
import io

def test_still(image_path="test.jpg"):
    camera = Camera()
    camera.start_feed()
    print("Camera started, capturing frame...")
    frame = camera.capture_frame()
    if frame is not None:
        print("Frame captured successfully")
    else:
        print("Failed to capture frame")
    camera.stop_feed()
    camera.close()

    # output frame as jpg
    stream = io.BytesIO()
    img = Image.fromarray(frame)

    img.save(stream, format='JPEG')
    stream.seek(0)
    with open(image_path, "wb") as f:
        f.write(stream.read())
    stream.close()
    print("Camera closed")

def test_video(video_path="test.mp4"):
    camera = Camera()
    camera.start_feed()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    target_framerate = 30  # Target framerate
    # Define video codec and create VideoWriter object
    video_writer = cv2.VideoWriter(
        video_path, 
        fourcc,
        target_framerate,
        (camera.resolution[0], camera.resolution[1]),
        isColor=True
    )

    print("Camera started, capturing video...")
    # Capture video for 5 seconds
    end_time = time.time() + 5  # Record for 5 seconds
    i = 0
    while time.time() < end_time:  # Assuming 30 FPS
        frame = camera.capture_frame()
        if frame is not None:
            print("*" * (i % 10) + " " * (10 - (i % 10)), end="\r")
        else:
            print("Failed to capture frame")
        video_writer.write(frame)
        i += 1

    print("Saving video...")
    video_writer.release()
    print("Video saved successfully")
    camera.stop_feed()
    camera.close()
    print("Camera closed")

if __name__ == "__main__":
    test_still()
    test_video()