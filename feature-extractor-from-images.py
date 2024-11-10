import cv2
import dlib
import os
import json
import numpy as np

# Paths
images_folder = 'images'
output_json = 'facial_features.json'

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictors/shape_predictor_68_face_landmarks.dat")  # Ensure this file is present

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

# Function to process all images in a folder and extract facial features
def process_images_in_folder(folder_path):
    all_landmarks = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Only process image files
            image_path = os.path.join(folder_path, filename)
            frame = cv2.imread(image_path)
            if frame is not None:
                landmarks = extract_facial_features(frame)
                if landmarks:
                    for face_landmarks in landmarks:
                        all_landmarks.append(face_landmarks)
    
    return all_landmarks

# Load existing facial features from the JSON file (if it exists)
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

# Main function to process all people in the images folder and update the JSON file
def main():
    # Load existing facial data if it exists
    facial_data = load_facial_data()

    # Iterate through all folders in the images directory (each folder is a person)
    for person_folder in os.listdir(images_folder):
        person_folder_path = os.path.join(images_folder, person_folder)
        
        if os.path.isdir(person_folder_path):  # Check if it's a folder
            print(f"Processing images for {person_folder}...")
            
            # Process all images inside the person's folder
            all_landmarks = process_images_in_folder(person_folder_path)
            
            if all_landmarks:
                # Average the landmarks
                avg_landmarks = average_landmarks(all_landmarks)
                
                # If the person already exists in facial_data, update their landmarks
                if person_folder in facial_data:
                    print(f"Updating landmarks for {person_folder}...")
                else:
                    print(f"Adding new person {person_folder}...")
                
                # Update the facial data with the averaged landmarks
                facial_data[person_folder] = {"landmarks": avg_landmarks}
    
    # Save the updated facial data to JSON
    save_facial_data(facial_data)

if __name__ == "__main__":
    main()
