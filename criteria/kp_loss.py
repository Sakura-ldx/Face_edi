import os

import dlib
import facer
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.common import tensor2image, deduplicate
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
from loguru import logger


class KPLoss(nn.Module):
    def __init__(self, device):
        super(KPLoss, self).__init__()
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_aligner = facer.face_aligner('farl/ibug300w/448', device=device)

    def forward(self, x: torch.Tensor, key_points: torch.Tensor):
        # image = np.zeros(x.shape[1:]).astype(np.uint8)
        # image = np.ascontiguousarray(np.transpose(image, (1, 2, 0)))
        # save_path = os.path.join("/home/ssd2/ldx/workplace/GANInverter-dev/test_edit/e4e/edit1/kp183072", f"{i}.png")

        key_points = key_points.squeeze(0)
        x = x.clone()
        x = ((x + 1) / 2)
        x = x.clamp(0., 1.)
        x = x * 255
        faces = self.face_detector(x)
        if not faces:
            return 0
        faces = self.face_aligner(x, faces)
        ld = faces['alignment'][0]
        # for x_i, y_i in ld:
        #     cv2.circle(image, (int(x_i), int(y_i)), 2, (255, 255, 255), -1)
        # cv2.imwrite(save_path, image)
        loss = F.mse_loss(ld, key_points)
        return loss


class KPLossTrain(nn.Module):
    def __init__(self, device):
        super(KPLossTrain, self).__init__()
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_aligner = facer.face_aligner('farl/ibug300w/448', device=device)

    def forward(self, x: torch.Tensor, extra_info):
        x = tensor2image(x)
        x_landmarks, x_indices = self.get_landmarks(x[extra_info[1]])
        loss = F.mse_loss(x_landmarks, extra_info[0][x_indices])
        return loss

    def get_landmarks(self, x):
        x_faces = self.face_detector(x)
        if not x_faces:
            return None
        x_faces = self.face_aligner(x, x_faces)

        if x.shape[0] < x_faces['alignment'].shape[0]:
            return deduplicate(x_faces)
        return x_faces['alignment'], x_faces['image_ids']
