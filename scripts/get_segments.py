import os
import sys
sys.path.append(".")
sys.path.append("..")
import cv2
import numpy as np
from PIL import Image
import torch
import facer
from torchvision.transforms import transforms

from models.unet.unet import unet
from models.bisenet.model import BiSeNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_parser = BiSeNet(19).to(device).eval()
checkpoint = torch.load("pretrained_models/79999_iter.pth")
face_parser.load_state_dict(checkpoint)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


def get_segments(filepath):
    image = Image.open(filepath)
    image = image.convert('RGB')
    image = transform(image).unsqueeze(dim=0).to(device)

    with (torch.inference_mode()):
        # seg_probs = torch.concat(face_parser(image), dim=0).softmax(dim=1)  # nfaces x nclasses x h x w
        seg_probs = face_parser(image)[0].softmax(dim=1)
    # n_classes = seg_probs.size(1)
    # seg_results = seg_probs.argmax(dim=1).int()
    vis_seg_probs = seg_probs.argmax(dim=1)

    vis_img = (vis_seg_probs.sum(0, keepdim=True) * 3).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
    return seg_probs[0], vis_img


def draw_segments(filepath):
    save_dir = os.path.join("../edit_data", "seg_bisenet")
    save_seg_dir = os.path.join("../edit_data", "seg_bisenet_pt")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_seg_dir, exist_ok=True)

    for filename in os.listdir(filepath):
        print(filename)
        read_path = os.path.join(filepath, filename)
        if os.path.splitext(filename)[1] == "pt":
            seg = torch.load(read_path)
        else:
            seg, image = get_segments(read_path)
            print(seg.size())

        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, image)
        # print(ld.shape)
        if seg is not None:
            save_seg_path = os.path.join(save_seg_dir, os.path.splitext(filename)[0] + ".pt")
            torch.save(seg, save_seg_path)


if __name__ == '__main__':
    draw_segments("/nas/Database/Public/CelebA-HQ/test100")
    # get_segments("/nas/Database/Public/CelebA-HQ/test100/183160.jpg")