import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30

def load_dataset():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy'))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return train_test_split(X, y, test_size=0.05)

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train():
    X_train, X_test, y_train, y_test = load_dataset()
    
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = build_model((30, 1662), actions.shape[0])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print('Training model...')
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
    model.summary()

    # Evaluate
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat_classes = np.argmax(yhat, axis=1).tolist()

    print('Confusion Matrix:')
    print(multilabel_confusion_matrix(ytrue, yhat_classes))
    print('Accuracy:', accuracy_score(ytrue, yhat_classes))

    model.save('action.h5')
    print('Model saved to action.h5')

if __name__ == '__main__':
    train()
