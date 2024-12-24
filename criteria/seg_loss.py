import os

import dlib
import facer
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
from loguru import logger

from models.unet.unet import unet
from models.bisenet.model import BiSeNet


class SegLoss(nn.Module):
    def __init__(self, device):
        super(SegLoss, self).__init__()
        # self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_parser = BiSeNet(19).to(device)
        checkpoint = torch.load("../pretrained_models/79999_iter.pth")
        self.face_parser.load_state_dict(checkpoint)

    def forward(self, x: torch.Tensor, segments: torch.Tensor):
        # image = np.zeros(x.shape[1:]).astype(np.uint8)
        # image = np.ascontiguousarray(np.transpose(image, (1, 2, 0)))
        # save_path = os.path.join("/home/ssd2/ldx/workplace/GANInverter-dev/test_edit/e4e/edit1/kp183072", f"{i}.png")

        seg_probs = self.face_parser(x)[0].softmax(dim=1)

        loss = F.cross_entropy(seg_probs, segments)
        return loss
