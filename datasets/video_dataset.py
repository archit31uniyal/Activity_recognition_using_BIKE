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
from .mp_handpose import MPHandpose
from .video_attr import VideoRecord
from torch import Tensor


class PoseDataset(VisionDataset):
    def __init__(self, 
                 root: Union[str, Path], 
                 annotation_path: str,
                 step_between_clips: int = 1,
                 frame_rate: Optional[int] = None,
                 fold: int = 1,
                 num_segments: int = 1, 
                 modality: str = 'RGB', 
                 new_length: int = 1,
                 image_tmpl: str = 'img_{:05d}.jpg', 
                 transform: Optional[Callable] = None,
                _precomputed_metadata: Optional[Dict[str, Any]] = None,
                _video_width: int = 0,
                _video_height: int = 0,
                 random_shift: bool = True, 
                 test_mode: bool = False,
                 index_bias: int = 1, 
                 dense_sample: bool = False, 
                 test_clips: int = 3,
                 num_sample: int = 1,
                _video_min_dimension: int = 0,
                 select_topk_attributes: int = 5,
                 attributes_path: str = None,
                 train: bool = True) -> None:

        super().__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError(f"Warning: fold should be between 1 and 3, got {fold}")

        self.root_path = root
        
        extensions = ("mp4",)
        self.fold = fold
        self.train = train

        self.class_list, class_to_idx = find_classes(root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        step_between_clips = 1
        video_clips = VideoClips(
            video_list,
            num_segments,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=1,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=0,
        )
        self.full_video_clips = video_clips
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform
        self.media_pipe = MPHandpose()

    def _select_fold(self, video_list: List[str], annotation_path: str, fold: int, train: bool) -> List[int]:
        name = "train" if train else "test"
        name = f"{name}list{fold:02d}.txt"
        f = os.path.join(annotation_path, name)
        selected_files = set()
        with open(f) as fid:
            data = fid.readlines()
            data = [x.strip().split(" ")[0] for x in data]
            data = [os.path.join(self.root, *x.split("/")) for x in data]
            selected_files.update(data)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, int, str]:
        video, audio, info, video_idx = self.video_clips.get_clip(index)
        label = self.samples[self.indices[video_idx]][1]
        # one hot
        tmp = np.zeros(len(self.class_list))
        tmp[label] = 1
        label = torch.from_numpy(tmp).type(torch.FloatTensor)
        sample_fname = self.samples[self.indices[video_idx]][0]

        pose = self.media_pipe.get_3d_hand_pose(video)

        if self.transform is not None:
            video = self.transform(video)
    
        return video, pose, label, sample_fname

    @property
    def classes(self):
        return self.class_list