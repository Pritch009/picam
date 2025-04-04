from pi_camera import Camera
from PIL import Image
import io

if __name__ == "__main__":
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