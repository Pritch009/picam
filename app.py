import os
import threading
from flask import Flask, send_file

from rich_camera import RichCamera

# def generate_frames():
#     while True:
#         frame = camera.capture_frame()
#         # motion_status = motion_detector.detect_motion(frame)
        
#         # if motion_status:
#         animals = animal_recognizer.recognize_animal(frame)
#         if animals:
#             for animal in animals:
#                 frame = animal_recognizer.draw_bounding_box(frame, animal)

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)

# model_path = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# model_path = "model/"
model_path = "model/mobilenetv2_ssd_fixed_1280_720.tflite"
keywords = ['person', 'cat', 'bear']
threshold = 0.5
recording_duration = 60  # seconds
motion_timeout = 10  # seconds
video_folder = "videos"
resolution = (1280, 720)

if not os.path.exists(video_folder):
    os.makedirs(video_folder)

camera = RichCamera(
    model_path=model_path,
    video_folder=video_folder,
    keywords=keywords,
    threshold=threshold,
    recording_duration=recording_duration,
    timeout=motion_timeout,
    resolution=resolution,
)

@app.route('/list_videos')
def list_videos():
    videos = camera.video_database.get_all_videos()
    return {'videos': [v.to_dict() for v in videos]}


@app.route('/video/<video_id>')
def get_video(video_id):
    video = camera.video_database.get_video(video_id)
    if video:
        video_path = video.filename
        if os.path.exists(video_path):
            return send_file(video_path, mimetype='video/mp4')  # Adjust mimetype if needed
        else:
            return "Video not found", 404
    else:
        return "Video not found", 404
    

if __name__ == '__main__':
    threading.Thread(target=camera.run_and_process).start()

    app.run(host='0.0.0.0', port=6143)