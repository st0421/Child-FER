import argparse
import os
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

import torchvision
from diffusion import get_time_schedule, Diffusion_Coefficients, Posterior_Coefficients, sample_from_model, q_sample_pairs, q_sample

from score_sde.models.ncsnpp_generator_adagn import DCNCSNpp


import torchvision.transforms as transforms
from datasets_prep.lmdb_datasets import OurDataset

import itertools
import torch_dct as dct


def patchify_image(image, patch_size=8):
    batch_size, channels, height, width = image.size()

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    image = image.view(batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size)
    image = image.permute(0, 2, 4, 1, 3, 5).contiguous()
    image = image.view(batch_size, num_patches_h * num_patches_w, channels, patch_size, patch_size)

    return image

def depatchify_image(patched_image, image_size=(256, 256), patch_size=8):
    batch_size, num_patches, channels, _, _ = patched_image.size()

    num_patches_h = image_size[0] // patch_size
    num_patches_w = image_size[1] // patch_size

    patched_image = patched_image.view(batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
    patched_image = patched_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    patched_image = patched_image.view(batch_size, channels, image_size[0], image_size[1])

    return patched_image

# %%
def sample_and_test(args):
    torch.manual_seed(args.seed)
    device = 'cuda:0'

    real_img_dir = args.real_img_dir

    def to_range_0_1(x):
        return (x + 1.) / 2.

    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    print(args.image_size, args.ch_mult, args.attn_resolutions)

    G_NET_ZOO = {"DC":DCNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    print("GEN: {}".format(gen_net))

    netG = gen_net(args).to(device)

    ckpt = torch.load('./saved_info/DCD_DCAC_origin_fer/{}/{}/netG_{}.pth'.format(
        args.dataset, args.exp, args.epoch_id), map_location=device)
    # loading weights from ddp in single gpu
    #for key in list(ckpt['netG'].keys()):
    #    ckpt['netG'][key[7:]] = ckpt['netG'].pop(key)

    netG.load_state_dict(ckpt['netG'], strict=False)
    netG.eval()

    T = get_time_schedule(args, device)
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    iters_needed = 50000 // args.batch_size

    save_dir = "./generated_samples/{}".format(args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    batch_size=args.batch_size
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = OurDataset(root=args.datadir, name='RAFDB', train=False, 
                            transform=img_transform,label_txt='label/rafdb.txt',img_dir='aligned')

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            drop_last=True)


    count=0
    for x, y,name in data_loader: 
        start.record()
        x0 = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        name = name[0].split('\\')[-1]

        patch_x0 = patchify_image(x0) # b,1024,c,8,8
        pdcx = dct.dct_2d(patch_x0)  #b,1024,c,8,8
        coeffs = []
        coeffs.append(pdcx[:,:,:,0,0].clone())
        coeffs.append(pdcx[:,:,:,0,1].clone())
        coeffs.append(pdcx[:,:,:,1,0].clone())
        coeffs.append(pdcx[:,:,:,2,0].clone())
        coeffs.append(pdcx[:,:,:,1,1].clone())
        coeffs.append(pdcx[:,:,:,0,2].clone())

        channel_max=torch.zeros(len(coeffs),batch_size) 
        channel_min=torch.zeros(len(coeffs),batch_size)

        for i in range(len(coeffs)):
            coeffs[i] = coeffs[i].reshape(batch_size,1024, 3, 1, 1)
            logf = torch.log(torch.abs(coeffs[i]) + 1)
            signf = torch.sign(coeffs[i])
            coeffs[i]= logf * signf

            for b in range(batch_size): 
                channel_max[i,b]=coeffs[i][b].max()
                channel_min[i,b]=coeffs[i][b].min()
                coeffs[i][b] = (coeffs[i][b] - channel_min[i,b]) / (channel_max[i,b] - channel_min[i,b])

            coeffs[i] = depatchify_image(coeffs[i],(32,32),1)

        real_data = torch.cat(coeffs,dim=1)
        real_data = real_data *2 -1

        assert -1 <= real_data.min() <= 0
        assert 0 <= real_data.max() <= 1

        x = real_data

        # diffusion process
        diffusion_middle_feature = []
        for i in range(args.num_timesteps):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            x = q_sample(coeff, x, t)
        real_base = patchify_image(x,1).clone() #32,1024,9,1,1


        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x, T, None, args)
        pred_base = patchify_image(fake_sample,1)
        pred = (pred_base+ 1) / 2
        for b in range(batch_size):
            for j in range(int(real_base.size(2)/3)): #개수
                g = j*3
                pred[b,:,g:g+3] = pred[b,:,g:g+3] * (channel_max[j,b] - channel_min[j,b]) + channel_min[j,b]

        p_sign = torch.sign(pred_base)
        pred = p_sign * torch.exp(torch.abs(pred))-1
        pred = pred.reshape(batch_size,1024,18)

        pred_result = torch.zeros(batch_size,1024,3,8,8).to(device)
        pred_result[:,:,:,0,0] = pred[:,:,:3]
        pred_result[:,:,:,0,1] = pred[:,:,3:6]
        pred_result[:,:,:,1,0] = pred[:,:,6:9]
        pred_result[:,:,:,2,0] = pred[:,:,9:12]
        pred_result[:,:,:,1,1] = pred[:,:,12:15]
        pred_result[:,:,:,0,2] = pred[:,:,15:18]

        prediction = dct.idct_2d(pred_result)
        pred = depatchify_image(prediction).to(device)
        end.record()
        torchvision.utils.save_image(pred, 'datasets/childFER/{}'.format(name), normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--datadir', default='D:/VILAB/datasets/childFER')

    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--epoch_id', type=int, default=500)
    parser.add_argument('--num_channels', type=int, default=9,
                        help='channel of wavelet subbands')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=64,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,default=(1, 2, 2, 2, 4),
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), type=int, nargs='+',
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # generator and training
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='',
                        help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='sample generating batch size')

    parser.add_argument("--current_resolution", type=int, default=32)
    parser.add_argument("--net_type", default="DC")

    args = parser.parse_args()

    sample_and_test(args)
