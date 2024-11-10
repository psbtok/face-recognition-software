import cv2
import dlib
import os
import json
import numpy as np

# Paths
output_json = 'facial_features.json'

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictors/shape_predictor_68_face_landmarks.dat")  # Download this file from dlib

# Function to extract facial features
def extract_facial_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    all_landmarks = []
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = []
        for n in range(68):  # 68 facial landmarks
            x, y = landmarks.part(n).x, landmarks.part(n).y
            landmarks_points.append((x, y))
        all_landmarks.append(landmarks_points)
    
    return all_landmarks

# Function to average the facial landmarks
def average_landmarks(all_landmarks):
    all_landmarks = np.array(all_landmarks)
    avg_landmarks = np.mean(all_landmarks, axis=0)
    return avg_landmarks.tolist()

# Function to capture frames and extract facial features
def capture_and_extract_features(num_frames=100):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    all_landmarks = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Extract facial features from the frame
        landmarks = extract_facial_features(frame)
        
        # If faces are detected, accumulate the landmarks
        if landmarks:
            for face_landmarks in landmarks:
                all_landmarks.append(face_landmarks)
        
    cap.release()
    return all_landmarks

# Load existing facial features from the JSON file
def load_facial_data():
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            return json.load(f)
    else:
        return {}

# Save or update the facial features in the JSON file
def save_facial_data(facial_data):
    with open(output_json, 'w') as f:
        json.dump(facial_data, f, indent=4)
    print(f"Facial features saved to {output_json}")

# Main function to process the facial recognition and update the JSON file
def main():
    print("Capturing 100 frames from webcam...")
    all_landmarks = capture_and_extract_features(num_frames=100)
    
    if all_landmarks:
        # Average the landmarks
        avg_landmarks = average_landmarks(all_landmarks)
        
        # Ask for the name of the person
        person_name = input("Enter the person's name: ")
        
        # Load the existing data from facial_features.json
        facial_data = load_facial_data()
        
        # Update the facial data with the averaged landmarks
        facial_data[person_name] = {"landmarks": avg_landmarks}
        
        # Save the updated facial data
        save_facial_data(facial_data)
    else:
        print("No facial landmarks detected. Exiting.")

if __name__ == "__main__":
    main()
