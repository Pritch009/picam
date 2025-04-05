import cv2
import tensorflow_hub as hub
import numpy as np
import os
from ai_edge_litert.interpreter import Interpreter


class AnimalRecognizer:
    def __init__(
        self,
        model_path=None, 
        keywords=["cat", "man"],
        threshold=0.3
    ):
        if model_path is None:
            raise ValueError("Model path cannot be None.")
        self.model_path = model_path
        self.keywords = keywords
        self.threshold = threshold
        self.model = None
        self.load_model()
        self.load_class_name_map()

    def load_model(self):
        print("Loading model...")
        if self.model_path.startswith("http"):
            self.model = hub.load(self.model_path)
            print("Model downloaded.")
        else:
            # Load from saved_model.pb
            self.model = Interpreter(
                model_path=self.model_path,
                num_threads=4,
            )
            self.model.allocate_tensors()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            _, input_height, input_width, _ = self.input_details[0]['shape']
            is_quantized_input = self.input_details[0]['dtype'] == np.uint8
            is_quantized_output = self.output_details[0]['dtype'] == np.uint8

            print("Model loaded.")
        if self.model is None:
            raise ValueError("Failed to load the model.")

    def load_class_name_map(self, class_names_path="./coco-classes.txt"):
        self.labels = {}
        path = os.path.dirname(self.model_path)
        path = os.path.join(path, class_names_path)
        with open(path, "r") as f:
            ind = 1
            for line in f:
                class_name = line.strip()
                self.labels[ind] = class_name
                ind += 1

        # # Extract metadata
        # displayer = _metadata.MetadataDisplayer.with_model_file(self.model_path)
        # raw_json_file = displayer.get_metadata_json()

        # # Load json as a dictionary
        # json_file = json.loads(raw_json_file)

        # # Get the name of the labels file 
        # labels_file = None
        # for item in json_file["subgraph_metadata"]:
        #     if "output_tensor_metadata" in item:
        #         output_tensor_metadata = item["output_tensor_metadata"]
        #         for metadata in output_tensor_metadata:
        #             if metadata.get("name", "") == "logit":
        #                 associated_files = metadata.get("associated_files", [])
        #                 for file in associated_files:
        #                     if file["type"] == "TENSOR_AXIS_LABELS":
        #                         labels_file = file["name"]
        #                         break
        #                 if labels_file:
        #                     break
        #         if labels_file:
        #             break
        #     if labels_file:
        #         break

        # if labels_file is None:
        #     raise ValueError("No labels file found in the model metadata.")
        # print(f"Labels file: {labels_file}")

        # model_dir = os.path.dirname(self.model_path)
        # labels_file_path = os.path.join(model_dir, labels_file)
        # if not os.path.exists(labels_file_path):
        #     # unzip the model
        #     os.system(f"unzip {self.model_path} -d {model_dir}")
        #     print(f"Labels file path: {labels_file_path}")

        # # Load the labels file
        # with open(labels_file_path, "r") as f:
        #     # Remove the new line characters
        #     self.labels = [line.strip() for line in f.readlines()]

        print("Class names loaded.")


    def recognize_animal(self, frame):
        if self.model is None:
            print("Model not loaded.")
            return []

        # Get input shape   
        input_shape = self.input_details[0]['shape']
        input_height, input_width = input_shape[1:3]

        # Resize the frame to a fixed size (e.g., 640x480)
        resized_frame = cv2.resize(frame, (input_width, input_height))

        # Expand dimensions since the model expects images to have shape: [1, height, width, 3]
        input_tensor = np.expand_dims(resized_frame, 0)

        # Set the input tensor
        self.model.set_tensor(self.input_details[0]['index'], input_tensor)

        # Perform the object detection
        self.model.invoke()

        # Extract detection boxes, scores, class names, and class labels
        detection_anchor_indices = self.model.get_tensor(self.output_details[0]['index'])[0]
        detection_boxes = self.model.get_tensor(self.output_details[1]['index'])[0]
        detection_classes = self.model.get_tensor(self.output_details[2]['index'])[0]
        detection_multiclass_scores = self.model.get_tensor(self.output_details[3]['index'])[0]
        detection_scores = self.model.get_tensor(self.output_details[4]['index'])[0]
        num_detections = self.model.get_tensor(self.output_details[5]['index'])[0].astype(np.uint32)
        raw_detection_boxes = self.model.get_tensor(self.output_details[6]['index'])[0]
        raw_detection_scores = self.model.get_tensor(self.output_details[7]['index'])[0]

        # Filter detections based on a confidence threshold (e.g., 30%)
        animal_detections = []
        for i in range(num_detections):
            if detection_scores[i] > self.threshold:
                class_name_raw = detection_classes[i].astype(np.uint32) # Decode bytes to string
                class_name = self.labels.get(class_name_raw, "").lower()
                if class_name == "":
                    continue

                # Check if the class name contains keywords for detection
                if class_name in self.keywords:
                    box = detection_boxes[i]
                    ymin, xmin, ymax, xmax = box
                    im_height, im_width, _ = frame.shape  # Use original frame dimensions
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                    ymin * im_height, ymax * im_height)
                    animal_detections.append((class_name, int(left), int(top), int(right-left), int(bottom-top)))  # x, y, width, height

        return animal_detections

    def draw_bounding_boxes(self, frame, boxes):
        # Draw bounding boxes around recognized animals on the frame
        # Check if boxes is a tuple or a list of tuples
        if isinstance(boxes, tuple):
            boxes = [boxes]
        elif not isinstance(boxes, list):
            raise ValueError("boxes should be a tuple or a list of tuples")
        
        # Draw rectangles around the detected animals
        for box in boxes:
            class_name, x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw name under the rectangle
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame