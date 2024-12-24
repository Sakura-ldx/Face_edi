import os
import sys

sys.path.append('.')
sys.path.append('..')

from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import EditDataset, MixDataset
from inference import TwoStageInference
from inference.edit_infer import OptimizerInference
from utils.common import tensor2im
from options.test_options import TestOptions
import torchvision.transforms as transforms


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
    dataset = MixDataset(root=opts.test_dataset_path)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False)

    for i, input_batch in enumerate(tqdm(dataloader)):
        try:
            source, target, source_path, target_path = input_batch
            source, target = source.cuda(), target.cuda()

            images_mix, _ = editor.mix(source, target)

            H, W = images_mix.shape[2:]
            for path, image_mix in zip(source_path, images_mix):
                basename = os.path.basename(path).split('.')[0]
                if opts.output_resolution is not None and ((H, W) != opts.output_resolution):
                    image_mix = torch.nn.functional.resize(image_mix, opts.output_resolution)
                result = tensor2im(image_mix)
                result.save(os.path.join(opts.output_dir, save_folder, f'{basename}.jpg'))
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    main()
