# Sign Language Translator 🤟

**🔴 Live Demo:** [Click here to try the web app!](https://your-deployed-link-here.com)  
*(Make sure to allow camera permissions when the site loads!)*

I built this project to translate sign language into text in real-time using just a standard webcam. It tracks hand and body movements, passes that data into a neural network I trained, and strings the detected signs together into continuous sentences on the screen.

## What it does
* **Real-Time Translation:** Uses a live webcam feed directly in the browser to detect signs.
* **Continuous Sentences:** Instead of just flashing single words, I wrote an "Action Latch" logic that waits for you to finish a sign before adding it to the sentence. This prevents the screen from getting spammed with the same word 30 times a second.
* **Currently Trained On:** 10 words (`hello`, `thanks`, `iloveyou`, `yes`, `no`, `please`, `sorry`, `help`, `eat`, `sleep`).

## Tech Stack
* **Python** 
* **OpenCV & MediaPipe (Tasks API):** For extracting the 3D coordinates of my hands and pose frame-by-frame.
* **TensorFlow / Keras:** I used a 3-layer LSTM neural network to process the sequence of frames over time since sign language is dynamic.
* **Streamlit & WebRTC:** To wrap the ML backend into a clean web interface.
