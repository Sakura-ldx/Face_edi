import os

import facer
import torch
from torch import nn
from models.encoders import psp_encoders
from configs.paths_config import model_paths
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from utils.common import tensor2image, deduplicate


class KpExtractor(nn.Module):
    def __init__(self, device):
        super(KpExtractor, self).__init__()
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_aligner = facer.face_aligner('farl/ibug300w/448', device=device)

    def forward(self, x: torch.Tensor):
        x = tensor2image(x)
        x_faces = self.face_detector(x)
        if not x_faces:
            return None
        x_faces = self.face_aligner(x, x_faces)

        # logger.info(f"x_faces before: {x_faces['alignment'].shape}")
        # logger.info(f"x_image id before: {x_faces['image_ids']}")
        if x.shape[0] < x_faces['alignment'].shape[0]:
            return deduplicate(x_faces)
        # logger.info(f"x_faces after: {x_faces['alignment'].shape}")
        return x_faces['alignment'], x_faces['image_ids']


class KpEncoder(nn.Module):
    def __init__(self, opts, layer_nums=8, g_layers=3):
        super(KpEncoder, self).__init__()

        self.g_layers = g_layers
        layers = [nn.LazyLinear(512), nn.LeakyReLU()]
        for _ in range(layer_nums):
            layers.append(nn.Linear(512, 512))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(512, 512 * g_layers))

        self.mlp = nn.Sequential(*layers)

        if 'dist' in opts and opts.dist:
            self.mlp = DistributedDataParallel(self.mlp, device_ids=[torch.cuda.current_device()],
                                               find_unused_parameters=True)
            self.dist = True
        else:
            self.dist = False

    def forward(self, x):
        latents = self.mlp(x)
        return latents.reshape(-1, self.g_layers, 512)
