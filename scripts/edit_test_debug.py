import os
import sys

sys.path.append('.')
sys.path.append('..')

from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.inference_dataset import InversionEditDataset
from inference import TwoStageInference
from inference.optim_infer import OptimizerInference
from utils.common import tensor2im
from options.test_options import TestOptions
import torchvision.transforms as transforms


def main():
    opts = TestOptions().parse()
    if opts.checkpoint_path is None:
        opts.auto_resume = True

    save_folder = opts.save_folder
    inversion = TwoStageInference(opts)
    editor = OptimizerInference(opts)

    if opts.output_dir is None:
        opts.output_dir = opts.exp_dir
    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, save_folder), exist_ok=True)

    if opts.output_resolution is not None and len(opts.output_resolution) == 1:
        opts.output_resolution = (opts.output_resolution, opts.output_resolution)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_no_resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    if os.path.isdir(opts.test_dataset_path):
        dataset = InversionEditDataset(root=opts.test_dataset_path, ld_root=opts.test_ld_path, transform=transform,
                                       transform_no_resize=transform_no_resize)
        dataloader = DataLoader(dataset,
                                batch_size=opts.test_batch_size,
                                shuffle=False,
                                num_workers=int(opts.test_workers),
                                drop_last=False)
    else:
        img = Image.open(opts.test_dataset_path)
        img = img.convert('RGB')
        img_aug = transform(img)
        img_aug_no_resize = transform_no_resize(img)
        dataloader = [(img_aug[None], [opts.test_dataset_path], img_aug_no_resize[None])]

    for i, input_batch in enumerate(tqdm(dataloader)):
        images_resize, img_paths, images, landmarks, landmark_paths = input_batch
        images_resize, images, landmarks = images_resize.cuda(), images.cuda(), landmarks.cuda()

        # with torch.no_grad():
        _, emb_codes, _, _, refine_codes, _ = inversion.inverse(images, images_resize, img_paths)
        images_edit, codes_edit, _, results_inter = editor.inverse(images, images_resize, img_paths, codes=emb_codes, ld=landmarks, return_inter=True)
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
        if i == 0:
            break
        # except:
        #     break


if __name__ == '__main__':
    main()
