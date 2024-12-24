import os

from torchvision import transforms
from loguru import logger
from torch.utils.data import IterableDataset
from tqdm import tqdm

from criteria import id_loss
from criteria.kp_loss import KPLoss
from criteria.lpips.lpips import LPIPS
import math

from criteria.seg_loss import SegLoss
from criteria.w_norm import WNormLoss
from models.encoder import Encoder
from models.stylegan2.model import Generator
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.common import tensor2im
from utils.train_utils import load_train_checkpoint
from inference.inference import BaseInference



def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


class OptimizerInference(BaseInference):

    def __init__(self, opts, decoder=None):
        super(OptimizerInference, self).__init__()
        self.opts = opts
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opts.device = self.device
        self.opts.n_styles = int(math.log(opts.resolution, 2)) * 2 - 2

        # resume from checkpoint
        checkpoint = load_train_checkpoint(opts)

        # initialize encoder and decoder
        self.latent_avg = None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = Generator(opts.resolution, 512, 8).to(self.device)
            self.decoder.eval()
            if checkpoint is not None:
                self.decoder.load_state_dict(checkpoint['decoder'], strict=True)
                self.latent_avg = checkpoint['encoder'].get('latent_avg', None).to(self.device)
            else:
                decoder_checkpoint = torch.load(opts.stylegan_weights, map_location='cpu')
                self.decoder.load_state_dict(decoder_checkpoint['g_ema'])
                self.latent_avg = decoder_checkpoint['latent_avg'].to(self.device) if checkpoint is None else None
        self.latent_std = None

        self.encoder = Encoder(opts, checkpoint, self.latent_avg, device=self.device).to(self.device)
        self.encoder.set_progressive_stage(self.opts.n_styles)
        self.encoder.eval()

        # initial loss
        # self.lpips_loss = LPIPS(net_type='vgg').to(self.device).eval()
        if self.opts.mode == "kp":
            self.kp_loss = KPLoss(self.device).eval()
        else:
            self.seg_loss = SegLoss(self.device).eval()
        self.id_loss = id_loss.IDLoss1().to(self.device).eval()
        # self.w_norm_loss = WNormLoss(start_from_latent_avg=True).to(self.device).eval()

    def inverse(self, images, info=None, image_name=None, return_marks=False, codes=None, return_inter=False):
        with torch.inference_mode():
            resize_transform = transforms.Resize((256, 256))
            images_resize = resize_transform(images)
            codes = self.encoder(images_resize)
            # normalize with respect to the center of an average latent codes
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w or codes.dim() == 2:
                    codes = codes + self.latent_avg[0].repeat(codes.shape[0], 1)
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        codes = codes.clone().detach()
        # del self.encoder
        # torch.cuda.empty_cache()
        layer = int(self.opts.edit_layer)

        latent_optim_kp = codes[:, :layer, :].clone().detach()
        latent_optim_other = codes[:, layer:, :].clone().detach()
        latent_optim_kp.requires_grad = True
        # latent_optim_other.requires_grad = True

        optimizer = optim.Adam([latent_optim_kp], lr=self.opts.lr)
        # optimizer_other = optim.Adam([latent_optim_other], lr=self.opts.lr)

        inter = []
        marks = dict()
        for i in tqdm(range(self.opts.optim_step)):
            t = i / self.opts.optim_step
            lr = get_lr(t, self.opts.lr)
            optimizer.param_groups[0]["lr"] = lr
            # optimizer_other.param_groups[0]["lr"] = lr

            latent_n = torch.concat((latent_optim_kp, latent_optim_other), dim=1)
            img_gen, _ = self.decoder([latent_n], input_is_latent=True)     # noise=noises)
            if self.opts.mode == "kp":
                loss_extra = self.opts.optim_kp_lambda * self.kp_loss(img_gen, info)
            else:
                loss_extra = self.opts.optim_seg_lambda * self.seg_loss(img_gen, info)

            loss_id = self.opts.optim_id_lambda * self.id_loss(img_gen, images)

            loss = loss_extra + loss_id
            optimizer.zero_grad()
            # optimizer_other.zero_grad()

            loss.backward()
            optimizer.step()

            # loss_id.backward(inputs=[latent_optim_other])
            # optimizer_other.step()

            # noise_normalize_(noises)
            inter.append(tensor2im(img_gen[0]))

        latent_n = torch.concat((latent_optim_kp, latent_optim_other), dim=1)
        img_gen, result_latent = self.decoder([latent_n], input_is_latent=True,  # noise=noises,
                                              return_latents=True)
        # loss_extra = self.seg_loss(img_gen, info)
        # loss_id = self.id_loss(img_gen, images)
        # marks["name"] = [os.path.basename(name).split('.')[0] for name in image_name]
        # marks["id_loss"] = loss_id.item()
        # marks["extra_loss"] = loss_extra.item()

        if return_marks and return_inter:
            return img_gen, result_latent, marks, inter
        elif return_marks:
            return img_gen, result_latent, marks, None
        elif return_inter:
            return img_gen, result_latent, None, inter
        else:
            return img_gen, result_latent, None, None

    def mix(self, source, target):
        layer = int(self.opts.edit_layer)

        with torch.inference_mode():
            latent_n = torch.concat((target[:, :layer, :], source[:, layer:, :]), dim=1)
            img_gen, result_latent = self.decoder([latent_n], input_is_latent=True,  # noise=noises,
                                                  return_latents=True)

        return img_gen, result_latent

    def edit(self, images, images_resize, image_path, editor):
        images, codes, _ = self.inverse(images, images_resize, image_path)
        edit_codes = editor.edit_code(codes)
        edit_images = self.generate(edit_codes)
        return images, edit_images, codes, edit_codes, None
