import cv2
import numpy as np
import os
import time

# Option to flip the image horizontally
flip_horizontally = True  # Set to True to enable horizontal flip

# Paths to the YOLO configuration, weights, and coco names files
config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
class_names_path = 'coco.names'

# Check if the class names file exists
if not os.path.exists(class_names_path):
    raise FileNotFoundError(f"{class_names_path} file not found. Please make sure it exists in the specified directory.")

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load class names
with open(class_names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Define the confidence threshold
confidence_threshold = 0.5

def detect_objects(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:  # Detect all objects with confidence above threshold
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    result_boxes = []
    result_class_ids = []
    if len(indices) > 0:
        for i in indices.flatten():
            result_boxes.append(boxes[i])
            result_class_ids.append(class_ids[i])

    return result_boxes, result_class_ids

def draw_boxes(image, boxes, class_ids):
    for (x, y, w, h), class_id in zip(boxes, class_ids):
        label = str(classes[class_id])  # Get the class name
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# Try different camera indices to find an available webcam
cap = None
for i in range(5):  # Try camera indices 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("Error: Could not open any webcam.")
    exit()

# Define frame rate limit
frame_rate = 5
prev_time = 0

# Process video stream
while True:
    # Capture frame-by-frame only if the time passed is greater than the frame interval
    time_elapsed = time.time() - prev_time
    if time_elapsed > 1.0 / frame_rate:
        prev_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally if the option is enabled
        if flip_horizontally:
            frame = cv2.flip(frame, 1)
        
        # Detect all objects in the current frame
        boxes, class_ids = detect_objects(frame)
        
        # Draw bounding boxes around detected objects
        output_frame = draw_boxes(frame, boxes, class_ids)

        # Display the output
        cv2.imshow('Webcam - Object Detection', output_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
