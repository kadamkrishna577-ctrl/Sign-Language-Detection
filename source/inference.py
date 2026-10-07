import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

actions = np.array(['hello', 'thanks', 'iloveyou','sorry','yes','no','help','please','eat','sleep'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('action.h5')
print("Loaded")

HAND_MODEL = 'hand_landmarker.task'
POSE_MODEL = 'pose_landmarker.task'

base_options_hand = python.BaseOptions(model_asset_path = HAND_MODEL)
options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL)
options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

def draw_landmarks_custom(image, hand_result, pose_result):
    h, w, _ = image.shape
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1) 
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1) 

def extract_keypoints(hand_result, pose_result):
    if pose_result.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_result.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(33*4)
        
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if hand_result.hand_landmarks:
        for idx, handedness in enumerate(hand_result.handedness):
            label = handedness[0].category_name
            landmarks = np.array([[res.x, res.y, res.z] for res in hand_result.hand_landmarks[idx]]).flatten()
            if label == 'Left': lh = landmarks
            else: rh = landmarks    
                
    return np.concatenate([pose, lh, rh])

# --- 4. Main Inference Loop ---
def main():
    sequence = []
    current_action = ""
    threshold = 0.85 # The network must be 85% confident to display a word

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Format for MediaPipe Tasks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Extract features
        hand_result = hand_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)
        draw_landmarks_custom(frame, hand_result, pose_result)
        
        keypoints = extract_keypoints(hand_result, pose_result)
        
        # Sequence buffering logic (keep only the last 30 frames)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        # When we have a full second of footage, run inference
        if len(sequence) == 30:
            # Expand dimensions from (30, 258) to (1, 30, 258) for the model
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            # Check if the highest probability beats our confidence threshold
            if res[np.argmax(res)] > threshold:
                current_action = actions[np.argmax(res)]
            
        # Draw a beautiful UI overlay for the text
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, f'Translation: {current_action}', (3,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign Language Translation', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()