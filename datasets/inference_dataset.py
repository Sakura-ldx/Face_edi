import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import glob
import os


class InversionDataset(Dataset):

	def __init__(self, root, transform=None, transform_no_resize=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.transform_no_resize = transform_no_resize

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')
		if self.transform:
			from_im_aug = self.transform(from_im)
		else:
			from_im_aug = from_im

		if self.transform_no_resize is not None:
			from_im_no_resize_aug = self.transform_no_resize(from_im)
			return from_im_aug, from_path, from_im_no_resize_aug
		else:
			return from_im_aug, from_path


class InversionCodeDataset(Dataset):

	def __init__(self, root):
		self.paths = sorted(glob.glob(os.path.join(root, '*.pt')))

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		code_path = self.paths[index]
		return torch.load(code_path, map_location='cpu'), code_path


class InversionEditDataset(Dataset):
	def __init__(self, root, ld_root, transform=None, transform_no_resize=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.ld_paths = sorted(glob.glob(os.path.join(ld_root, '*.pt')))
		self.transform = transform
		self.transform_no_resize = transform_no_resize

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')
		if self.transform:
			from_im_aug = self.transform(from_im)
		else:
			from_im_aug = from_im

		ld_path = self.ld_paths[(index + 5) % len(self.paths)]
		ld = torch.load(ld_path, map_location='cpu')

		if self.transform_no_resize is not None:
			from_im_no_resize_aug = self.transform_no_resize(from_im)
			return from_im_aug, from_path, from_im_no_resize_aug, ld, ld_path
		else:
			return from_im_aug, from_path, ld, ld_path


class EditDataset(Dataset):
	def __init__(self, root, info_root, image_root, transform=None):
		self.paths = sorted(glob.glob(os.path.join(root, '*.pt')))
		self.info_paths = sorted(glob.glob(os.path.join(info_root, '*.pt')))
		self.image_paths = sorted(data_utils.make_dataset(image_root))
		self.transform = transform

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		code_path = self.paths[index]
		code = torch.load(code_path, map_location='cpu')

		info_path = self.info_paths[(index + 5) % len(self.info_paths)]
		info = torch.load(info_path, map_location='cpu')

		image_path = self.image_paths[index]
		image = Image.open(image_path)
		image = image.convert('RGB')
		if self.transform:
			image_aug = self.transform(image)
		else:
			image_aug = image

		return code, info, image_aug, code_path, info_path, image_path


class WebDataset(Dataset):
	def __init__(self, image_root, info_root, transform=None):
		self.info_paths = sorted(glob.glob(os.path.join(info_root, '*.pt')))
		self.image_paths = sorted(data_utils.make_dataset(image_root))
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		info_path = self.info_paths[index]
		info = torch.load(info_path, map_location='cpu')

		image_path = self.image_paths[index]
		image = Image.open(image_path)
		image = image.convert('RGB')
		if self.transform:
			image_aug = self.transform(image)
		else:
			image_aug = image

		return image_aug, info, image_path, info_path


class TestDataset(Dataset):
	def __init__(self, edit_root, source_root, target_root, target_root1, edit_transform=None, source_transform=None):
		self.edit_paths = sorted(glob.glob(os.path.join(edit_root, '*.jpg')))
		self.source_root = source_root
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.source_paths_map = list2map(self.source_paths)
		self.target_paths = sorted(glob.glob(os.path.join(target_root, '*.pt')))
		self.target_paths1 = None
		if target_root1:
			self.target_paths1 = sorted(glob.glob(os.path.join(target_root1, '*.pt')))

		self.edit_transform = edit_transform
		self.source_transform = source_transform

	def __len__(self):
		return len(self.edit_paths)

	def __getitem__(self, index):
		edit_path = self.edit_paths[index]
		edit_image = Image.open(edit_path)
		edit_image = edit_image.convert('RGB')
		if self.edit_transform:
			edit_image = self.edit_transform(edit_image)

		source_path = os.path.join(self.source_root, os.path.basename(edit_path))
		source_image = Image.open(source_path)
		source_image = source_image.convert('RGB')
		if self.source_transform:
			source_image = self.source_transform(source_image)

		source_index = self.source_paths_map[source_path]
		target_path = self.target_paths[(source_index + 5) % len(self.target_paths)]
		target = torch.load(target_path, map_location='cpu')
		if self.target_paths1:
			target_path1 = self.target_paths1[(source_index + 5) % len(self.target_paths1)]
			target1 = torch.load(target_path1, map_location='cpu')
			return edit_image, source_image, target, target1, edit_path, source_path, target_path, target_path1
		return edit_image, source_image, target, edit_path, source_path, target_path


class MixDataset(Dataset):
	def __init__(self, root):
		self.source_paths = sorted(glob.glob(os.path.join(root, '*.pt')))

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		source_path = self.source_paths[index]
		source = torch.load(source_path, map_location='cpu')

		target_path = self.source_paths[(index + 5) % len(self.source_paths)]
		target = torch.load(target_path, map_location='cpu')

		return source, target, source_path, target_path


def list2map(paths):
	res = dict()
	for i, path in enumerate(paths):
		res[path] = i
	return res
