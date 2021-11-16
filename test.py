# -*- coding:utf-8 -*-

import argparse
import torch
import scipy.io as sio
from torchvision.utils import save_image
from basic_unet import UNet
from Dataset import *
import cv2

""" set flags / seeds """
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description="")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--data_path', type=str, default='/home/Data/Data/CNV/Model_data/train1')
parser.add_argument('--model_path', type=str,
                    default='logs/gen_pix2pix_[lr=150.0]_five_fold_2/model_G_140000lp_0.400.ckpt')
parser.add_argument('--results_dir', type=str, default='/home/Data/Data/CNV/Model_data/results_our_new/five_fold_2')
opt = parser.parse_args()

results_realA = os.path.join(opt.results_dir, 'realA')
if not os.path.exists(results_realA):
    os.makedirs(results_realA)
results_realB = os.path.join(opt.results_dir, 'realB')
if not os.path.exists(results_realB):
    os.makedirs(results_realB)

results_fakeB = os.path.join(opt.results_dir, 'fakeB')
if not os.path.exists(results_fakeB):
    os.makedirs(results_fakeB)

if __name__ == "__main__":

    """ datasets and dataloader """
    test_loader = get_data_loaders(opt, 'test')

    """ instantiate network and loss function"""
    # netG = networks.define_G(3, 1, 64, 'global',
    #                                   n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1,
    #                                   n_blocks_local=3, norm = 'instance', gpu_ids=0)

    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    netG = UNet(n_in=7, n_out=1, first_channels=32, n_dps=5, use_pool=True, use_bilinear=True,
                norm_type='instance', device=device)

    """models init or load checkpoint"""
    # print(torch.load(opt.model_save_path))
    netG.load_state_dict(torch.load(opt.model_path)['netG'])

    print('ok')

    netG = netG.to(device)
    netG.eval()
    ssim = 0

    with torch.no_grad():

        for _, sampled_batch_test in enumerate(test_loader):
            real_A, real_B = sampled_batch_test['images'], sampled_batch_test['labels']

            # real_A_mid = np.array(real_A_mid.cpu())
            # real_B = real_B[0, 0, :, :]
            # real_B = np.array(real_B.cpu())

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B, _ = netG(real_A)
            # fake_B = np.array(fake_B.cpu())
            # fake_B = fake_B[0, 0, :, :]
            real_A_mid = real_A[:, 3, :, :].unsqueeze(1)
            real_B_mid = real_B[:, 3, :, :].unsqueeze(1)

            filename = sampled_batch_test['filename']
            filename = filename[0]

            # sio.savemat(os.path.join(opt.results_dir, filename), {'realA': real_A_mid, 'realB': real_B, 'fakeB': fake_B})

            save_image(real_A_mid, os.path.join(results_realA, filename + '.png'))
            save_image(real_B_mid, os.path.join(results_realB, filename+'.png'))
            save_image(fake_B, os.path.join(results_fakeB, filename + '.png'))

            # gen_label = gen_label.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            # seg_out = seg_out.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            # ssim += compare_ssim(gen_label, seg_out)

    # ssim /= len(test_dataset)
    # os.rename(opt.img_save_dir, opt.img_save_dir+'ssim_%.3f'%ssim)
