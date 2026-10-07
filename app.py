import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import av

# --- 1. Streamlit Page Setup ---
st.set_page_config(page_title="Sign Language Translator", layout="wide")
st.title("Real-Time Sign Language Translator")
st.markdown("Perform a sign ('hello', 'thanks', 'iloveyou') to see the real-time translation.")

# --- 2. Define the WebRTC Video Processor ---
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize variables
        self.actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no', 'please', 'sorry', 'help', 'eat', 'sleep'])
        self.sequence = []
        self.sentence = []
        self.threshold = 0.85
        
        # Load LSTM Model
        self.model = Sequential()
        self.model.add(Input(shape=(30, 258), name='input_layer'))
        self.model.add(LSTM(64, return_sequences=True, activation='relu', name='lstm_layer_1'))
        self.model.add(LSTM(128, return_sequences=True, activation='relu', name='lstm_layer_2'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu', name='lstm_layer_3'))
        self.model.add(Dense(64, activation='relu', name='dense_layer_1'))
        self.model.add(Dense(32, activation='relu', name='dense_layer_2'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax', name='dense_output_layer'))
        
        self.model.load_weights('action.h5')

        # Load MediaPipe Tasks API
        base_options_hand = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

        base_options_pose = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

    def draw_landmarks(self, image, hand_result, pose_result):
        h, w, _ = image.shape
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1) 
        if hand_result.hand_landmarks:
            for hand in hand_result.hand_landmarks:
                for lm in hand:
                    cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1) 

    def extract_keypoints(self, hand_result, pose_result):
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

    def recv(self, frame):
        # Convert WebRTC frame to OpenCV array
        img = frame.to_ndarray(format="bgr24")
        
        # Format for MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Extract features
        hand_result = self.hand_landmarker.detect(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)
        self.draw_landmarks(img, hand_result, pose_result)
        
        keypoints = self.extract_keypoints(hand_result, pose_result)
        
        # Sequence buffering logic
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]
        
        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            
            if res[np.argmax(res)] > self.threshold:
                predicted_word = self.actions[np.argmax(res)]
                
                if len(self.sentence) > 0: 
                    if predicted_word != self.sentence[-1]:
                        self.sentence.append(predicted_word)
                else:
                    self.sentence.append(predicted_word)

            if len(self.sentence) > 5: 
                self.sentence = self.sentence[-5:]
                
        # Draw UI
        cv2.rectangle(img, (0, 400), (640, 480), (0, 0, 0), -1)
        display_text = ' '.join(self.sentence)
        cv2.putText(img, display_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Return processed frame to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. Streamlit Component ---
webrtc_streamer(
    key="sign-language-translator",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.sidebar.markdown("### Controls")
if st.sidebar.button("Clear Sentence"):
    # Note: State management across WebRTC threads requires session state in a more advanced setup, 
    # but the user can always just stop/start the stream to reset.
    st.sidebar.info("To clear the text, toggle the webcam off and back on.")