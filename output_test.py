import cv2
from pi_camera import Camera
from PIL import Image
import io

def test_still():
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
    with open("output.jpg", "wb") as f:
        f.write(stream.read())
    stream.close()
    print("Camera closed")

def test_video():
    camera = Camera()
    camera.start_feed()
    video_filename = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    target_framerate = 30  # Target framerate
    # Define video codec and create VideoWriter object
    video_writer = cv2.VideoWriter(
        video_filename, 
        fourcc,
        target_framerate,
        (frame.shape[1], frame.shape[0])
    )

    print("Camera started, capturing video...")
    # Capture video for 5 seconds
    for _ in range(150):  # Assuming 30 FPS
        frame = camera.capture_frame()
        if frame is not None:
            print("Frame captured successfully")
        else:
            print("Failed to capture frame")
        video_writer.write(frame)
    video_writer.release()
    print("Video saved successfully")
    camera.stop_feed()
    camera.close()
    print("Camera closed")

if __name__ == "__main__":
    test_still()
    test_video()