import os
import sys

sys.path.append('.')
sys.path.append('..')

import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import WebDataset
from inference import TwoStageInference
from inference.edit_infer import OptimizerInference
from utils.common import tensor2im
from options.test_options import TestOptions
import torchvision.transforms as transforms
import gradio as gr
import facer
from models.bisenet.model import BiSeNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_aligner = facer.face_aligner('farl/ibug300w/448', device=device)
face_parser = BiSeNet(19).to(device).eval()
checkpoint = torch.load("../pretrained_models/79999_iter.pth")
face_parser.load_state_dict(checkpoint)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

def main(config_path, image, image_info):
    opts = TestOptions().parse(config_path)
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    save_folder = opts.save_folder

    if opts.output_dir is None:
        opts.output_dir = opts.exp_dir
    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, save_folder), exist_ok=True)

    if opts.output_resolution is not None and len(opts.output_resolution) == 1:
        opts.output_resolution = (opts.output_resolution, opts.output_resolution)
    
    editor = OptimizerInference(opts)
    image = transform(image).unsqueeze(dim=0).to(device)
    if opts.mode == "kp":
        np_image_info = np.array(image_info)
        image_info = torch.from_numpy(np_image_info)
        image_info = facer.hwc2bchw(image_info).to(device)
        with torch.inference_mode():
            faces = face_detector(image_info)
            faces = face_aligner(image_info, faces)
            info = faces['alignment']
    else:
        image_info = transform(image_info).unsqueeze(dim=0).to(device)
        with torch.inference_mode():
            info = face_parser(image_info)[0].softmax(dim=1)[0]
    info = info.clone().detach()

    image_edit, _, _, _ = editor.inverse(image, info, return_marks=False, return_inter=False)
    image_edit = tensor2im(image_edit[0])
    return image_edit

# if __name__ == "__main__":
#     main("/home/liudongxv/workplace/GANInverter-dev/configs/edit/edit_kp.yaml",
#         "/nas/Database/Public/CelebA-HQ/test100/183226.jpg",
#         "/home/liudongxv/workplace/GANInverter-dev/edit_data/kp_pt/184454.pt")

iface = gr.Interface(
    fn=main,
    inputs=["text", "image", "image"],
    outputs="image",
    title="edit",
    description="choose a config, input your image and info, generate the result."
)

iface.launch()