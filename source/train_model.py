import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou','sorry','yes','no','help','please','eat','sleep'])
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

print("Loading data from MP_Data folder...")
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        
        sequences.append(window)
        labels.append(label_map[action])

# labels are nothing but the names of signs now 
# Convert to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)
print("Splitting the data into train and test categories")
X_train, X_test, y_train ,y_test = train_test_split(X,y, test_size=0.1)

model = Sequential()
model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))

model.add(Dense(actions.shape[0],activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model.fit(X_train,y_train,epochs=200, callbacks=[tb_callback])

model.save('action.h5')
