import cv2
import mediapipe as mp
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import random
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense # fully connected layer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
import tensorflow as tf

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos])
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


FACE_LIST_LEN = 468
LEFT_LIST_LEN = 21
RIGHT_LIST_LEN = 21
POSE_LIST_LEN = 33

TOTAL_LIST_LEN = FACE_LIST_LEN*3 + LEFT_LIST_LEN*3 + RIGHT_LIST_LEN*3 + POSE_LIST_LEN*4

def extract_photos(classes):
    path = '/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5'
    frames_path = os.path.join(path, 'frames_wlasl_10')
    poses_path = os.path.join(path, 'poses')
    os.makedirs(poses_path, exist_ok=True)

    for count, class_name in enumerate(classes):
        class_path = f'{frames_path}/{class_name}'

        os.makedirs(f'{poses_path}/{class_name}', exist_ok=True)

        video_name_list = os.listdir(class_path)

        for video in video_name_list:
            video_path = f'{class_path}/{video}'

            frame_list = os.listdir(video_path)
            os.makedirs(f'{poses_path}/{class_name}/{video}', exist_ok=True)

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for frame in frame_list:
                    frame_path = f'{class_path}/{video}/{frame}'
                    extracted_path = f'{poses_path}/{class_name}/{video}/{frame.split(".")[0].split("_")[-1]}'
                    bgr_frame, results = mediapipe_detection(cv2.imread(frame_path), holistic) #hands.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                    landmarks = extract_landmarks(results)
                    # rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    np.save(extracted_path, landmarks)

def extract_landmarks(results):
    pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_LIST_LEN*4)
    face = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.face_landmarks.landmark]).flatten()  if results.face_landmarks else np.zeros(FACE_LIST_LEN*3)
    left = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(LEFT_LIST_LEN*3)
    right = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(RIGHT_LIST_LEN*3)
    return np.concatenate([pose, face, left, right])

def create_model(classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(np.array(classes).shape[0], activation='softmax'))
    return model

def load(
    classes,
    label_map,
    sample_num = 20,
    samples_per_video = 10, 
    poses_path = '/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/poses'):
    
    class_dict = {}
    for count, class_name in enumerate(classes):
        class_path = f'{poses_path}/{class_name}'

        video_name_list = os.listdir(class_path)
        class_dict[class_path] = video_name_list
    
    print(class_dict)
    min_vids = np.min([len(value) for _, value in class_dict.items()])
    print(f'Min Vids is {min_vids}')

    sequences, labels = [], []
    pose_lens = []
    for class_path, video_name_list in class_dict.items():
        for vid_cnt, video in enumerate(video_name_list):
            if (vid_cnt) > min_vids:
                break
            window = []
            video_path = f'{class_path}/{video}'

            poses_list = os.listdir(video_path)

            window = np.empty((len(poses_list), TOTAL_LIST_LEN))
            for pose_cnt, pose in enumerate(poses_list):

                pose_path = f'{video_path}/{pose}'
                res = np.load(pose_path)

                window[pose_cnt] = res

            for _ in range(samples_per_video):
                # print(pose_cnt, np.array(window).shape)
                random_sequence = sorted(random.sample(range(pose_cnt+1), 20))
                # print(random_sequence)
                new_window = window[random_sequence, :]
                sequences.append(new_window)
                labels.append(label_map[class_name])
    return np.array(sequences), to_categorical(np.array(labels)).astype(int)

def main():
    seed = int(sys.argv[1])
    epochs = int(sys.argv[2])
    num_classes = int(sys.argv[3])
    logs = sys.argv[4]
    log_dir = os.path.join(logs)
    tb_callback = TensorBoard(log_dir=log_dir)
    print(f"Setting seed as {seed}")
    random.seed(seed)
    tf.random.set_seed(seed)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # Create a label map

    possible_classes = ['love', 'thankyou', 'face', 'no', 'thin', 'man', 'woman', 'cousin', 'deaf', 'no', 'who', 'help', 'soon'] 
    classes = []
    for i, cls_name in enumerate(possible_classes):
        if i+1 > num_classes:
            break
        classes.append(cls_name)

    print(classes)

    label_map = {label:num for num, label in enumerate(classes)}
    
    # sequences, y = load(classes, label_map)
    path = '/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5'
    poses_path = os.path.join(path, 'poses')
    sequences, labels = [], []
    pose_lens = []
    sample_num = 20
    samples_per_video = 10
    for count, class_name in enumerate(classes):
        class_path = f'{poses_path}/{class_name}'

        video_name_list = os.listdir(class_path)

        for vid_cnt, video in enumerate(video_name_list):
            if (vid_cnt) > 7:
                break
            window = []
            video_path = f'{class_path}/{video}'

            poses_list = os.listdir(video_path)

            window = np.empty((len(poses_list), TOTAL_LIST_LEN))
            for pose_cnt, pose in enumerate(poses_list):

                pose_path = f'{video_path}/{pose}'
                res = np.load(pose_path)

                window[pose_cnt] = res

            for _ in range(samples_per_video):
                # print(pose_cnt, np.array(window).shape)
                random_sequence = sorted(random.sample(range(pose_cnt+1), 20))
                # print(random_sequence)
                new_window = window[random_sequence, :]
                sequences.append(new_window)
                labels.append(label_map[class_name])

    print(f'Shape of sequences is {np.array(sequences).shape} and shape of labels is {np.array(labels).shape}')

    y = to_categorical(np.array(labels)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(np.array(sequences), y, test_size=0.2, random_state=seed)
    print(f'{X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}')

    model = create_model(classes)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback])

    res = model.predict(X_test)
    print(f'{classes} --> {epochs}: {(np.sum([classes[np.argmax(res[idx])] == classes[np.argmax(y_test[idx])] for idx in range(len(res))])/len(res))*100:02f}%')

    model.save('action.h5')
    
    # from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
    # model.load_weights('action.h5')
    # yhat = model.predict(X_test)
    # ytrue = np.argmax(y_test, axis=1).tolist()
    # yhat = np.argmax(yhat, axis=1).tolist()

    # multilabel_confusion_matrix(ytrue, yhat)
    # print(f'Accuracy Score: {accuracy}')

if __name__ == '__main__':
    main()