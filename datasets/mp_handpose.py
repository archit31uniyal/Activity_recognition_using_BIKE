import json

import torch
import torch.utils.data as data
import decord
import os
import numpy as np
from numpy.random import randint
import io
import pandas as pd
import random
from PIL import Image
import math
import copy
from clip.simple_tokenizer import SimpleTokenizer as ClipTokenizer
import mediapipe as mp
import cv2

from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from torchvision.datasets.video_utils import VideoClips

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'  # Suppresses TensorFlow logs if used

from absl import logging
logging.set_verbosity(logging.ERROR)

# default Python logging output behavior when present.
if 'absl.logging' in sys.modules:
  import absl.logging
  absl.logging.set_stderrthreshold('fatal')  
import os
os.environ["GLOG_minloglevel"] ="5"
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import mediapipe as mp
debug = False
import numpy as np
import cv2
import os
import sys

class MPHandpose():
    def __init__(self):
        # Redirect stdout and stderr to suppress logs
        # sys.stdout = open(os.devnull, 'w')
        # self.stderr = sys.stderr
        # sys.stderr = open(os.devnull, 'w')

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # print("initialize")
        # video = '/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/videos/56811.mp4'
        # self.get_hands(video)

    def __del__(self):
        # sys.stdout = sys.__stdout__
        try:
            sys.stderr = self.stderr
        except:
            print('MP deleted')

    def mediapipe_to_numpy(self, landmark):
        np_arr = np.zeros(shape=(len(landmark), 3), dtype=float)
        for i, _landmark in enumerate(landmark):
            np_arr[i, 0] = _landmark.x
            np_arr[i, 1] = _landmark.y
            np_arr[i, 2] = _landmark.z
        return np_arr

    def numpy_to_mediapipe(self, np_arr):
        # Create a LandmarkList object
        landmark_list_list = []
        
        # Iterate through each row in the NumPy array
        for row in np_arr:
            landmark_list = LandmarkList()
            # Create a new Landmark for each set of coordinates
            for coor in row:
                landmark = Landmark()
                landmark.x = coor[0]
                landmark.y = coor[1]
                landmark.z = coor[2]
                # Append the Landmark to the LandmarkList
                landmark_list.landmark.append(landmark)
            landmark_list_list.append(landmark_list)

        return landmark_list_list

    def make_annotated_frame(self, idx, results, image_width, image_height):
        for hand_landmarks in results.multi_hand_landmarks:
            np_arr = self.mediapipe_to_numpy(hand_landmarks.landmark)
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            print(type(hand_landmarks))
            print(len(hand_landmarks.landmark))
            print(len(mp_hands.HandLandmark))
            print(dir(hand_landmarks))
            print(np_arr.shape)
            # Create annotated image
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite(
                '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

    def get_hands(self, video, debug=False):
        cap = cv2.VideoCapture(video)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands


        with mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5) as hands:
            idx = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
            
                # Read an image, flip it around y-axis for correct handedness output (see
                # above).
                # frame = cv2.imread(directory + file)
                image = cv2.flip(frame, 1)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print handedness and draw hand landmarks on the image.
                # print('Handedness:', results.multi_handedness)
                if results.multi_hand_landmarks:
                    if debug:
                        image_height, image_width, _ = image.shape
                        annotated_image = image.copy()
                        self.make_annotated_frame(idx, results, image_width, image_height)

                    for hand_world_landmarks in results.multi_hand_world_landmarks:
                        # print(dir(hand_world_landmarks))
                        np_arr = self.mediapipe_to_numpy(hand_world_landmarks.landmark)
                        if debug:
                            self.mp_drawing.plot_landmarks(
                            hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
                        # print(np_arr)
                    
                    idx += 1
                    if idx == 1:
                        break

    def get_pose(self, frame):
        hand_pose = np.zeros((21, 3))
        with self.mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5) as hands:
            image = np.array(frame)
            image = cv2.flip(image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                return hand_pose
            else:
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    np_arr = self.mediapipe_to_numpy(hand_world_landmarks.landmark)

                return np_arr

    def get_3d_hand_pose_old(self, record, video_list, indices, debug=False):
        poses = list()
        if self.debug:
            print(indices)
        for seg_ind in indices:
            p = int(seg_ind)
            if self.modality == 'video':
                image = Image.fromarray(video_list[p - 1].asnumpy()).convert('RGB')
                image = cv2.flip(frame, 1)
                seg_imgs = [self.get_pose(image)]
            else:
                imgs = self._load_image(record.path, p)
                seg_imgs = []
                for img in imgs:
                    seg_imgs.append(self.get_pose(img))
            poses.extend(seg_imgs)
            if p < len(video_list):
                p += 1
        
        return torch.from_numpy(np.array(poses))

    def get_3d_hand_pose(self, video_clip):
        poses = list()
        for frame in video_clip:
            # image = Image.fromarray(frame).convert('RGB')
            # removed because we are only grabbing right hand motion
            # image = cv2.flip(frame, 1) 
            seg_imgs = [self.get_pose(frame)]
            poses.extend(seg_imgs)
        
        return torch.from_numpy(np.array(poses))