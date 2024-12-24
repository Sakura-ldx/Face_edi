import os
import sys

sys.path.append('.')
sys.path.append('..')

from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import EditDataset
from inference import TwoStageInference
from inference.edit_infer import OptimizerInference
from utils.common import tensor2im
from options.test_options import TestOptions
import torchvision.transforms as transforms


def load_config(file_path):
    opts = TestOptions().parse(file_path)
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    save_folder = opts.save_folder

    if opts.output_dir is None:
        opts.output_dir = opts.exp_dir
    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, save_folder), exist_ok=True)

    if opts.output_resolution is not None and len(opts.output_resolution) == 1:
        opts.output_resolution = (opts.output_resolution, opts.output_resolution)
    return opts

def main():
    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    save_folder = opts.save_folder
    editor = OptimizerInference(opts)

    if opts.output_dir is None:
        opts.output_dir = opts.exp_dir
    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, save_folder), exist_ok=True)

    if opts.output_resolution is not None and len(opts.output_resolution) == 1:
        opts.output_resolution = (opts.output_resolution, opts.output_resolution)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = EditDataset(root=opts.test_dataset_path, info_root=opts.test_info_path,
                          image_root=opts.test_inversion_path, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False)

    for i, input_batch in enumerate(tqdm(dataloader)):
        # if i < 30:
        #     continue
        if i == 10:
            break
        try:
            codes, info, images, code_paths, info_paths, img_paths = input_batch
            images, codes, info = images.cuda(), codes.cuda(), info.cuda()

            images_edit, codes_edit, marks, results_inter = editor.inverse(images, img_paths, codes=codes, info=info, return_marks=False, return_inter=True)
            edit_images = images_edit

            H, W = edit_images.shape[2:]
            for path, edit_img in zip(img_paths, edit_images):
                basename = os.path.basename(path).split('.')[0]
                if opts.output_resolution is not None and ((H, W) != opts.output_resolution):
                    edit_img = torch.nn.functional.resize(edit_img, opts.output_resolution)
                edit_result = tensor2im(edit_img)
                edit_result.save(os.path.join(opts.output_dir, save_folder, f'{basename}.jpg'))

                os.makedirs(os.path.join(opts.output_dir, save_folder, basename), exist_ok=True)
                for j, results_inter_batch in enumerate(results_inter):
                    if j % 100 == 0:
                        results_inter_batch.save(os.path.join(opts.output_dir, save_folder, basename, "{:04d}.png".format(j)))
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    main()
