# -*- coding:utf-8 -*-
import os
import argparse
import shutil
import random
import platform
from prefetch_generator import BackgroundGenerator
import numpy as np

from Dataset import *

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
from init_weights import init_weights
from loss import GANLoss
# from models.mynetworks.myunet import UNet
from basic_unet import UNet
from patchGAN_discriminator import NLayerDiscriminator
from log_function import print_options, print_network
from util import LambdaLR, set_requires_grad
import lpips
""" set flags / seeds """
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Pix2PixModel(nn.Module):
    def __init__(self, opt, device='cpu'):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.netG = UNet(n_in=7, n_out=1, first_channels=32, n_dps=5, use_pool=True, use_bilinear=True,
                         norm_type=opt.norm_G, device=device)
        init_weights(self.netG, init_type=opt.init_type)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        if self.opt.isTrain:
            self.netD = NLayerDiscriminator(input_nc=1, norm_D=opt.norm_D, n_layers_D=4)
            init_weights(self.netD, init_type=opt.init_type)
            # define loss functions
            self.criterionGAN = GANLoss(gan_mode=opt.gan_mode)
            self.criterionL1 = torch.nn.MSELoss()
            self.criterionCos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


    def criterion_D(self, fake_B, real_B):
        fake_in = fake_B
        pred_fake = self.netD(fake_in.detach())
        loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False, for_discriminator=True)
        real_in = real_B
        pred_real = self.netD(real_in)
        loss_D_real = self.criterionGAN(pred_real, target_is_real=True, for_discriminator=True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return [loss_D]


    def criterion_G(self, fake_B, real_B, enc_fea1, enc_fea2):
        fake_in = fake_B
        pred_fake = self.netD(fake_in)
        loss_G_GAN = self.criterionGAN(pred_fake, target_is_real=True, for_discriminator=False)

        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1

        CosSim1 = self.criterionCos(enc_fea1[0], enc_fea2[0])
        CosSim1 = CosSim1.squeeze()
        CosSim2 = self.criterionCos(enc_fea1[1], enc_fea2[0])
        CosSim2 = CosSim2.squeeze()
        loss_G_Cos = -torch.log(torch.exp(CosSim2) / torch.exp(CosSim1)) * self.opt.lambda_L1 * 0.1

        loss_G = loss_G_GAN + loss_G_L1 + loss_G_Cos

        return [loss_G, loss_G_GAN, loss_G_L1, loss_G_Cos]


if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="new pix2pix")
    parser.add_argument('--experiment_name', type=str, default='gen_pix2pix_[lr=100.0]_five_fold_and_one_4')
    # training option
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--num_iters', type=int, default=140000)
    parser.add_argument('--schedule_times', type=int, default=20000)
    parser.add_argument('--schedule_decay_times', type=int, default=10000)
    parser.add_argument('--lr_g', type=float, default=0.0001)
    parser.add_argument('--lr_d', type=float, default=0.0004)
    parser.add_argument('--lambda_L1', type=float, default=100.0)
    parser.add_argument('--eval_iters', type=int, default=5000)  # 1000
    parser.add_argument('--save_iters', type=int, default=10000)  # 5000
    # model option
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
    parser.add_argument('--init_type', type=str, default='normal', help='[normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--norm_G', type=str, default='instance')
    parser.add_argument('--norm_D', type=str, default='instance')

    # data option
    parser.add_argument('--data_path', type=str, default='/home/Data/Data/CNV/Model_data')
    parser.add_argument('--result_dir', type=str, default='logs')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_crop_size', type=int, default=0)
    opt = parser.parse_args()

    opt.result_dir = os.path.join(opt.result_dir, opt.experiment_name)
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
        print_options(parser, opt)
        shutil.copyfile(os.path.abspath(__file__), os.path.join(opt.result_dir, os.path.basename(__file__)))
    else:
        print("result_dir exists: ", opt.result_dir)
        "exit()"
    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """ track of experiments """
    writer = SummaryWriter(os.path.join(opt.result_dir, 'runs'))
    print("track view:", os.path.join(opt.result_dir, 'runs')[3:])

    """ datasets and dataloader """
    train_loader = get_data_loaders(opt, 'train')
    test_loader = get_data_loaders(opt, 'test')
    print('train_dataset len:', len(train_loader), 'test_dataset len:', len(test_loader))

    """ instantiate network and loss function"""
    model = Pix2PixModel(opt, device)
    print_network(model, opt)

    """ optimizer and scheduler """
    optimizer_G = torch.optim.Adam(model.netG.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model.netD.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
    if opt.schedule_times:
        scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.schedule_times, 0,
                                                                                        opt.schedule_times - opt.schedule_decay_times).step)
        scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.schedule_times, 0,
                                                                                        opt.schedule_times - opt.schedule_decay_times).step)
        schedule_iters = opt.num_iters // opt.schedule_times
        print('scheduler total %d iters from %d iters, iter_size %d' % (opt.schedule_times, opt.schedule_times - opt.schedule_decay_times, schedule_iters))

    """ training part """
    losses_name = ['loss_D', 'loss_G', 'loss_G_GAN', 'loss_l1']
    losses_cnt = 0
    losses_sum = np.zeros(len(losses_name))
    pbar = tqdm.tqdm(range(opt.num_iters))
    global_iter = 0
    model = model.to(device)
    model.train()
    while global_iter < opt.num_iters:
        for _, sampled_batch_train in enumerate(train_loader):
            real_A, real_B = sampled_batch_train['images'], sampled_batch_train['labels']
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            real_A_mid = real_A[:, 3, :, :].unsqueeze(1)
            real_B_mid = real_B[:, 3, :, :].unsqueeze(1)

            fake_B, enc_fea1 = model.netG(real_A)
            _, enc_fea2 = model.netG(real_B)

            # update D
            set_requires_grad(model.netD, True)
            losses_D = model.criterion_D(fake_B, real_B_mid)
            optimizer_D.zero_grad()
            losses_D[0].backward()
            optimizer_D.step()

            # update G
            set_requires_grad(model.netD, False)
            losses_G = model.criterion_G(fake_B, real_B_mid, enc_fea1, enc_fea2)
            optimizer_G.zero_grad()
            losses_G[0].backward()
            optimizer_G.step()

            losses_sum = list(map(lambda x, y: x.item() + y, losses_D + losses_G, losses_sum))
            losses_cnt += real_A.shape[0]

            global_iter += 1
            pbar.update(1)
            if opt.schedule_times and global_iter % schedule_iters == 0:
                scheduler_D.step()
                scheduler_G.step()

            if global_iter % opt.eval_iters == 0:

                # write training data
                losses_sum = list(map(lambda x: x / losses_cnt, losses_sum))
                ave_losses = dict(zip(losses_name, losses_sum))
                writer.add_scalars('ave_losses', ave_losses, global_iter)
                losses_sum = np.zeros(len(losses_name))

                train_lpips = model.loss_fn_vgg(fake_B, real_B_mid)

                real_B_mid = real_B_mid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
                fake_B = fake_B.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
                writer.add_scalar('train lpips', train_lpips.item(), global_iter)
                train_paras = 'lp_%.3f' % train_lpips

                # model eval
                test_lpips = 0
                p = 0
                model.eval()
                with torch.no_grad():
                    for _, sampled_batch_test in enumerate(test_loader):
                        randnum = random.randint(1, 4)
                        if randnum != 1:
                            continue
                        real_A, real_B = sampled_batch_test['images'], sampled_batch_test['labels']
                        real_A = real_A.to(device)
                        real_B = real_B.to(device)
                        real_B_mid = real_B[:, 3, :, :].unsqueeze(1)
                        fake_B, _ = model.netG(real_A)
                        test_lpips += model.loss_fn_vgg(fake_B, real_B_mid)
                        p = p+1
                model.train()

                # write testing data
                test_lpips /= p
                real_A_mid = real_A[:, 3, :, :].unsqueeze(1)
                writer.add_images('test_real_A', real_A_mid, global_iter)
                writer.add_images('test_fake_B', fake_B, global_iter)
                writer.add_images('test_real_B', real_B_mid, global_iter)
                eval_paras = 'lp_%.3f' % test_lpips
                # model saving
                save_dir = opt.result_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if global_iter % opt.save_iters == 0:
                    state = {'netG': model.netG.state_dict()}
                    torch.save(state,
                               os.path.join(opt.result_dir, 'model_G_' + str(global_iter) + eval_paras + '.ckpt'))

                # print log
                print('iter: %d' % global_iter + ', train: ' + train_paras + ', test: ' + eval_paras)

            if global_iter == opt.num_iters:
                break
