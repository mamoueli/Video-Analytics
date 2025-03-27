import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
!pip install ultralytics
!pip install deep-sort-realtime
!pip install facenet-pytorch
!pip install opencv-python-headless
!pip install numpy
!pip install pillow
!pip install torch torchvision
!pip install scenedetect
!pip install insightface


# Import torchvision models
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN, InceptionResnetV1
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

from insightface.app import FaceAnalysis

# Force download or load with genderage model
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
