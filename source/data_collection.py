import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# --- 1. Download Required Task Models ---
# The new API requires physical model files. This downloads them automatically.
HAND_MODEL = 'hand_landmarker.task'
POSE_MODEL = 'pose_landmarker.task'

def download_models():
    if not os.path.exists(HAND_MODEL):
        print("Downloading Hand model (this only happens once)...")
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", HAND_MODEL)
    if not os.path.exists(POSE_MODEL):
        print("Downloading Pose model (this only happens once)...")
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", POSE_MODEL)

download_models()

# --- 2. Initialize the New MediaPipe Tasks API ---
base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL)
options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL)
options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

# --- 3. Variables for Dataset ---
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['hello', 'thanks', 'iloveyou','sorry','yes','no','help','please','eat','sleep']) # Add more signs here later
no_sequences = 30
sequence_length = 30

# Create folder structure
for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

def draw_landmarks_custom(image, hand_result, pose_result):
    """Custom drawing function since mp.solutions was removed"""
    h, w, _ = image.shape
    # Draw pose (Blue dots)
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1) 
    # Draw hands (Green dots)
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1) 

def extract_keypoints(hand_result, pose_result):
    """Formats the data from the new API into our original 258-value array."""
    # Pose
    if pose_result.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # Hands
    lh = np.zeros(21*3) # Right hands
    rh = np.zeros(21*3) # Left hands
    
    if hand_result.hand_landmarks:
        for idx, handedness in enumerate(hand_result.handedness):
            label = handedness[0].category_name # Returns 'Left' or 'Right'
            landmarks = np.array([[res.x, res.y, res.z] for res in hand_result.hand_landmarks[idx]]).flatten()
            
            if label == 'Left':
                lh = landmarks
            else:
                rh = landmarks
                
    return np.concatenate([pose, lh, rh])

def main():
    cap = cv2.VideoCapture(0)
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: break
                
                # Convert frame for MediaPipe Tasks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Run Inference
                hand_result = hand_landmarker.detect(mp_image)
                pose_result = pose_landmarker.detect(mp_image)
                
                # Draw and Extract
                draw_landmarks_custom(frame, hand_result, pose_result)
                keypoints = extract_keypoints(hand_result, pose_result)
                
                # UI Logic
                if frame_num == 0: 
                    cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('Sign Language Data Collection', frame)
                    cv2.waitKey(2000) # 2 second pause to get in position
                else: 
                    cv2.putText(frame, f'Collecting: {action} | Video: {sequence}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Sign Language Data Collection', frame)
                
                # Save the array to disk
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()