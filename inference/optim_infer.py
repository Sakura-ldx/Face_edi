import os

from loguru import logger
from tqdm import tqdm

from criteria import id_loss
from criteria.kp_loss import KPLoss
from criteria.lpips.lpips import LPIPS
import math

from criteria.w_norm import WNormLoss
from models.stylegan2.model import Generator
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.common import tensor2im
from utils.train_utils import load_train_checkpoint, requires_grad
from inference.inference import BaseInference
import cv2
import numpy as np


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
        self.device = 'cuda'
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

        # initial loss
        # self.lpips_loss = LPIPS(net_type='vgg').to(self.device).eval()
        self.kp_loss = KPLoss(self.device).eval()
        # self.id_loss = id_loss.IDLoss1().to(self.device).eval()
        # self.w_norm_loss = WNormLoss(start_from_latent_avg=True).to(self.device).eval()

    def inverse(self, images, images_resize, image_name, return_lpips=False, codes=None, ld=None, return_inter=False):

        # if self.latent_std is None:
        #     n_mean_latent = 10000
        #     with torch.no_grad():
        #         noise_sample = torch.randn(n_mean_latent, 512, device=self.device)
        #         latent_out = self.decoder.style(noise_sample)
        #         latent_mean = latent_out.mean(0)
        #         if self.latent_avg is None:
        #             self.latent_avg = latent_mean
        #         self.latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        #
        # latent_std = self.latent_std.detach().clone()
        # latent_mean = self.latent_avg.detach().clone()

        # noises_single = self.decoder.make_noise()
        # noises = []
        # for noise in noises_single:
        #     noises.append(noise.repeat(images_resize.shape[0], 1, 1, 1).normal_())

        latent_in = None
        # if codes is None:
        #     latent_in = latent_mean.unsqueeze(0).repeat(images_resize.shape[0], 1)
        #     if self.opts.w_plus:
        #         latent_in = latent_in.unsqueeze(1).repeat(1, self.decoder.n_latent, 1)
        # else:
        latent_in = codes

        # # latent_in.requires_grad = True
        # for noise in noises:
        #     noise.requires_grad = True
        latent_optim_kp = latent_in[:, :3, :].clone().detach()
        latent_optim_other = latent_in[:, 3:, :].clone().detach()
        latent_optim_kp.requires_grad = True
        latent_optim_other.requires_grad = True

        optimizer_kp = optim.Adam([latent_optim_kp], lr=self.opts.lr)
        # optimizer_other = optim.Adam([latent_optim_other], lr=self.opts.lr)

        inter = []
        for i in range(self.opts.optim_step):
            # print(i)
            # dir_path = "/home/ssd2/ldx/workplace/GANInverter-dev/test_edit/e4e/edit1/183072lr0.001"
            # os.makedirs(dir_path, exist_ok=True)
            # save_path = os.path.join(dir_path, "{:04d}.png".format(i))
            t = i / self.opts.optim_step
            lr = get_lr(t, self.opts.lr)
            optimizer_kp.param_groups[0]["lr"] = lr
            # optimizer_other.param_groups[0]["lr"] = lr
            # noise_strength = latent_std * self.opts.noise * max(0, 1 - t / self.opts.noise_ramp) ** 2
            latent_n = torch.concat((latent_optim_kp, latent_optim_other), dim=1)
            # latent_n = latent_noise(latent_n, noise_strength.item())

            img_gen, _ = self.decoder([latent_n], input_is_latent=True)     # noise=noises)
            # img_save = tensor2im(img_gen[0])
            # img_save.save(save_path)

            # if i == (self.opts.optim_step - 1) and return_lpips:
            #     delta = self.lpips_loss(img_gen, images, keep_res=True)

            loss_kp = self.opts.optim_kp_lambda * self.kp_loss(img_gen, ld, i)
            # loss_id = self.id_loss(img_gen, images)
            # loss_w_norm = self.opts.optim_w_norm_lambda * self.w_norm_loss(latent_n, self.latent_avg)
            # loss_lpips = self.opts.optim_lpips_lambda * self.lpips_loss(img_gen, images)

            # img_gen = F.interpolate(torch.clamp(img_gen, -1., 1.), size=(images_resize.shape[2], images_resize.shape[3]), mode='bilinear')
            #
            # p_loss = self.lpips_loss(img_gen, images_resize)
            # n_loss = noise_regularize(noises)
            # mse_loss = F.mse_loss(img_gen, images_resize)
            #
            # loss = self.opts.optim_lpips_lambda * p_loss + self.opts.noise_regularize * n_loss + \
            #        self.opts.optim_l2_lambda * mse_loss + self.opts.optim_kp_lambda * kp_loss
            # loss = loss_kp + loss_id + loss_lpips
            optimizer_kp.zero_grad()
            # optimizer_other.zero_grad()
            # loss_w_norm.backward(retain_graph=True)

            loss_kp.backward(retain_graph=True, inputs=[latent_optim_kp])
            optimizer_kp.step()

            # loss_id.backward(inputs=[latent_optim_other])
            # optimizer_other.step()

            # noise_normalize_(noises)
            inter.append(tensor2im(img_gen[0]))

        latent_n = torch.concat((latent_optim_kp, latent_optim_other), dim=1)
        images, result_latent = self.decoder([latent_n], input_is_latent=True,  # noise=noises,
                                             return_latents=True)

        if return_lpips and return_inter:
            return images, result_latent, _, inter
        elif return_lpips:
            return images, result_latent, _, None
        elif return_inter:
            return images, result_latent, None, inter
        else:
            return images, result_latent, None, None

    def edit(self, images, images_resize, image_path, editor):
        images, codes, _ = self.inverse(images, images_resize, image_path)
        edit_codes = editor.edit_code(codes)
        edit_images = self.generate(edit_codes)
        return images, edit_images, codes, edit_codes, None
