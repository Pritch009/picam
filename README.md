# Raspberry Pi Camera Project

This project utilizes a Raspberry Pi to run a camera feed that includes motion detection and animal recognition capabilities. The camera feed will continuously run, displaying the live video along with bounding boxes around recognized animals.

## Project Structure

- **app.py**: Main entry point of the application. Initializes the camera feed, sets up motion detection, and handles the display of the camera feed along with recognized animals.
- **camera.py**: Contains the `Camera` class that manages camera operations, including methods to start and stop the camera feed.
- **motion_detection.py**: Exports the `MotionDetector` class, which includes methods to analyze the camera feed for motion and retrieve the current motion status.
- **animal_recognition.py**: Exports the `AnimalRecognizer` class, which identifies animals in the camera feed and draws bounding boxes around them.
- **config.py**: Contains configuration settings such as camera resolution, motion detection sensitivity, and model paths for animal recognition.
- **requirements.txt**: Lists the dependencies required for the project, including `picamera`, `opencv-python`, and any necessary machine learning libraries.

## Setup Instructions

1. **Install Dependencies**: Ensure you have Python installed on your Raspberry Pi. Install the required packages by running:
   ```
   pip install -r requirements.txt
   ```

2. **Configure Camera**: Make sure your Raspberry Pi camera is enabled. You can do this through the Raspberry Pi configuration settings.

3. **Run the Application**: Start the application by executing:
   ```
   python app.py
   ```

## Features

- Continuous camera feed with real-time motion detection.
- Animal recognition with bounding boxes drawn around detected animals.
- Configurable settings for camera resolution and motion detection sensitivity.

## Usage Guidelines

- Adjust the settings in `config.py` to optimize performance based on your environment.
- Ensure proper lighting conditions for better motion detection and animal recognition accuracy.

## Acknowledgments

This project leverages various libraries for image processing and machine learning. Special thanks to the contributors of these libraries for their invaluable work.