import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

sys.path.append('.')
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.inference_dataset import TestDataset
from utils.metrics import KpMetric, IDMetric, SegMetric


def main(model, edit_images_root, source_images_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path1 = os.path.join(output_dir, f"{model}_layer_exp.txt")
    output_path2 = os.path.join(output_dir, f"{model}_avg_data.txt")

    edit_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    source_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    mode_map = defaultdict(list)
    for data in os.listdir(edit_images_root):
        if "seg" in data:
            mode = "seg"
        elif "kp" in data:
            mode = "kp"
        else:
            mode = "mix"
        edit_images_path = os.path.join(opts.edit_images_root, data)
        mode_map[mode].append(edit_images_path)

    for mode, edit_images_path_list in mode_map.items():
        print(mode)
        if mode == "mix":
            continue
        with open(output_path1, mode='a') as f:
            f.write(f"{mode}\n")
        with open(output_path2, mode='a') as f:
            f.write(f"{mode}\n")
        edit_metric1, condition_path1 = None, None
        if mode == "kp":
            edit_metric = KpMetric(device="cuda").eval()
            condition_path = "/home/liudongxv/workplace/GANInverter-dev/edit_data/kp_pt"
        elif mode == "seg":
            edit_metric = SegMetric(device="cuda").eval()
            condition_path = "/home/liudongxv/workplace/GANInverter-dev/edit_data/seg_bisenet_pt"
        else:
            edit_metric = KpMetric(device="cuda").eval()
            edit_metric1 = SegMetric(device="cuda").eval()
            condition_path = "/home/liudongxv/workplace/GANInverter-dev/edit_data/kp_pt"
            condition_path1 = "/home/liudongxv/workplace/GANInverter-dev/edit_data/seg_bisenet_pt"
        id_metric = IDMetric().cuda().eval()
        edit_images_path_list.sort(key=lambda a: int(a.split('_')[-1][1:]))
        for edit_images_path in edit_images_path_list:
            print(edit_images_path)
            with open(output_path1, mode='a') as f:
                f.write(f"{edit_images_path}\n")
            dataset = TestDataset(edit_root=edit_images_path, source_root=source_images_path, target_root=condition_path,
                                  target_root1=condition_path1, edit_transform=edit_transform, source_transform=source_transform)
            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    drop_last=False)

            edit_results, edit_results1, id_results = [], [], []
            for i, input_batch in enumerate(tqdm(dataloader)):
                if i == 10:
                    break
                targets1 = None
                if mode != "mix":
                    edit_images, source_images, targets, edit_paths, source_paths, target_paths = input_batch
                    edit_images, source_images, targets = edit_images.cuda(), source_images.cuda(), targets.cuda()
                else:
                    edit_images, source_images, targets, targets1, edit_paths, source_paths, target_paths, targets_path1 = input_batch
                    edit_images, source_images, targets = edit_images.cuda(), source_images.cuda(), targets.cuda()
                with torch.inference_mode():
                    edit_result = edit_metric(edit_images, targets)
                    id_result = id_metric(edit_images, source_images)
                edit_results.append(edit_result)
                id_results.append(id_result)
                if targets1 is not None:
                    targets1 = targets1.cuda()
                    with torch.inference_mode():
                        edit_result1 = edit_metric1(edit_images, targets1)
                    edit_results1.append(edit_result1)
                with open(output_path1, mode='a') as f:
                    if mode != "mix":
                        f.write(f"{edit_paths[0]}: {mode}: {edit_result} 'id': {id_result}\n")
                    else:
                        f.write(f"{edit_paths[0]}: 'kp': {edit_result}, 'seg': {edit_result1}, 'id': {id_result}\n")
            with open(output_path2, mode='a') as f:
                if mode != "mix":
                    f.write(f"{sum(edit_results) / len(edit_results):.4f}, {sum(id_results) / len(id_results):.4f}\n")
                else:
                    f.write(f"{sum(edit_results) / len(edit_results):.4f}, "
                            f"{sum(edit_results1) / len(edit_results1):.4f}, "
                            f"{sum(id_results) / len(id_results):.4f}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--edit_images_root',
                        default="/home/liudongxv/workplace/GANInverter-dev/test_edit/edit/restyle_id",
                        type=str, help='edit images.')
    parser.add_argument('--source_images_path',
                        default="/nas/Database/Public/CelebA-HQ/test100",
                        type=str, help='source images.')
    parser.add_argument('--output_dir', default="../edit_metrics", type=str)
    opts = parser.parse_args()

    model = opts.edit_images_root.split('/')[-1]

    main(model,
         opts.edit_images_root,
         opts.source_images_path,
         opts.output_dir)
