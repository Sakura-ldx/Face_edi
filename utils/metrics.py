import facer
import math
import torch
from loguru import logger
from torch import nn
import torch.nn.functional as F

from configs.paths_config import model_paths
from models.bisenet.model import BiSeNet
from models.encoders.model_irse import Backbone


class KpMetric(nn.Module):
    def __init__(self, device):
        super(KpMetric, self).__init__()
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_aligner = facer.face_aligner('farl/ibug300w/448', device=device)

    def forward(self, x: torch.Tensor, key_points: torch.Tensor):
        n = key_points.shape[0]
        x = x.clone()
        x = ((x + 1) / 2)
        x = x.clamp(0., 1.)
        x = x * 255
        faces = self.face_detector(x)
        if not faces:
            return 0
        faces = self.face_aligner(x, faces)
        kp_predicts = faces['alignment'][0]
        rects = faces["rects"][0]
        distance = F.mse_loss(kp_predicts, key_points).item()
        area = ((rects[2] - rects[0]) * (rects[3] - rects[1])).item()
        return math.exp(-math.sqrt(distance / area))


class SegMetric(nn.Module):
    def __init__(self, device):
        super(SegMetric, self).__init__()
        self.face_parser = BiSeNet(19).to(device)
        checkpoint = torch.load("../pretrained_models/79999_iter.pth")
        self.face_parser.load_state_dict(checkpoint)

    def forward(self, x: torch.Tensor, segments: torch.Tensor):
        seg_probs = self.face_parser(x)[0].softmax(dim=1).argmax(dim=1)
        segments = segments.argmax(dim=1)
        n = segments.shape[0]
        result = 0
        for segment, seg_prob in zip(segments, seg_probs):
            confusion_metrix = torch.bincount((19 * segment + seg_prob).reshape(-1), minlength=19 * 19).reshape(19, 19)
            iou_sum = 0
            for i in range(19):
                iou_sum += confusion_metrix[i, i].item() / (torch.sum(confusion_metrix[i, :]) + torch.sum(confusion_metrix[:, i]) - confusion_metrix[i, i] + 1e-10).item()
            miou = iou_sum / 19
            result += miou

        return result / n


class IDMetric(nn.Module):
    def __init__(self):
        super(IDMetric, self).__init__()
        logger.info(f'Loading ResNet ArcFace from path {model_paths["ir_se50"]}.')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50'], map_location='cpu'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_hat_feats = self.extract_feats(y_hat)
        sim = 0
        count = 0
        for i in range(n_samples):
            sim += y_hat_feats[i].dot(x_feats[i])
            count += 1

        return sim / count
