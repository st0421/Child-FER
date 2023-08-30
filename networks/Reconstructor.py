import argparse
import os

import numpy as np
import torch
import torchvision
from diffusion import get_time_schedule, Diffusion_Coefficients, Posterior_Coefficients, sample_from_model, q_sample


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

    gen_net = DCNCSNpp()
    print("GEN: {}".format(gen_net))

    netG = gen_net(args).to(device)
    ckpt = torch.load('../models/netG_{}.pth'.format(args.epoch_id), map_location=device)

    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)

    netG.load_state_dict(ckpt, strict=False)
    netG.eval()

    T = get_time_schedule(args, device)
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    iters_needed = 50000 // args.batch_size

    save_dir = "./generated_samples/{}".format(args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.measure_time:
        x_t_1 = torch.randn(args.batch_size, args.num_channels,
                            args.image_size, args.image_size).to(device)
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1,None, T, args)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                fake_sample = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1,None, T, args)

                fake_sample *= 2.
                fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                    (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("Inference time: {:.2f}+/-{:.2f}ms".format(mean_syn, std_syn))
        exit(0)

    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():
                x_t_1 = torch.randn(
                    args.batch_size, args.num_channels, args.image_size, args.image_size).to(device)
                fake_sample = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1,None, T, args)

                fake_sample *= 2
                fake_sample = torch.clamp(fake_sample, -1, 1)

                fake_sample = to_range_0_1(fake_sample)  # 0-1
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j
                    torchvision.utils.save_image(
                        x, '{}/{}.jpg'.format(save_dir, index))
                print('generating batch ', i)

        paths = [save_dir, real_img_dir]
        print(paths)

        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    
    if args.inference:
        batch_size=args.batch_size
        img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])

        dataset = OurDataset(root=args.datadir, name=args.dataset, train=False, 
                             transform=img_transform,label_txt='childFER.txt',img_dir='childFER')

        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=True)

        for x, y in data_loader: 
            x0 = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)


            #diffusion input preprocessing
            patch_x0 = patchify_image(x0) # b,1024,c,8,8
            pdcx = dct.dct_2d(patch_x0)  #b,1024,c,8,8
            pdc = pdcx[:,:,:,0,0].clone().to(device) # b,1024,c
            pac1= pdcx[:,:,:,1,0].clone().to(device)
            pac2= pdcx[:,:,:,0,1].clone().to(device)
            pdc = pdc.reshape(batch_size,1024,3,1,1) # 32,1024,3,1,1
            pac1 = pac1.reshape(batch_size,1024,3,1,1) # 32,1024,3,1,1
            pac2 = pac2.reshape(batch_size,1024,3,1,1) # 32,1024,3,1,1

            dc_log = torch.log(torch.abs(pdc) + 1)
            dc_sign = torch.sign(pdc)
            dcinput = dc_log * dc_sign
            
            ac1_log = torch.log(torch.abs(pac1) + 1)
            ac1_sign = torch.sign(pac1)
            ac1input = ac1_log * ac1_sign

            ac2_log = torch.log(torch.abs(pac2) + 1)
            ac2_sign = torch.sign(pac2)
            ac2input = ac2_log * ac2_sign

            channel_max=torch.zeros(3,batch_size) 
            channel_min=torch.zeros(3,batch_size)

            for b in range(batch_size): 
                channel_min[0,b] = dcinput[b].min()
                channel_max[0,b] = dcinput[b].max()
                channel_min[1,b] = ac1input[b].min()
                channel_max[1,b] = ac1input[b].max()
                channel_min[2,b] = ac2input[b].min()
                channel_max[2,b] = ac2input[b].max()

                dcinput[b] = (dcinput[b] - channel_min[0,b]) / (channel_max[0,b] - channel_min[0,b])
                ac1input[b] = (ac1input[b] - channel_min[1,b]) / (channel_max[1,b] - channel_min[1,b])
                ac2input[b] = (ac2input[b] - channel_min[2,b]) / (channel_max[2,b] - channel_min[2,b])

            dcinput = depatchify_image(dcinput,(32,32),1) # b,3,32,32
            ac1input = depatchify_image(ac1input,(32,32),1) # b,3,32,32
            ac2input = depatchify_image(ac2input,(32,32),1) # b,3,32,32
            
            real_data = torch.cat([dcinput, ac1input, ac2input], dim=1)  # [b, 9, h, w]
            real_data = real_data *2 -1

            assert -1 <= real_data.min() < 0
            assert 0 < real_data.max() <= 1


            x = real_data
            # forward process
            diffusion_middle_feature = []
            for i in range(args.num_timesteps):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
                x = q_sample(coeff, x, t)
                x = (x - x.min()) / (x.max() - x.min())
                diffusion_middle_feature.append(x)
                #diffusion_middle_feature[i] *= 2

            # real_data = (real_data + 1) / 2
            # real_data = real_data*(dmax-dmin)+dmin
            # real_signed = torch.sign(real_data)
            # real_exp =(torch.abs(real_data)).clamp_(max=80).exp()-1
            # real_sample = real_exp*real_signed
            # real_base = torch.zeros(args.batch_size,3,args.ori_image_size,args.ori_image_size)
            # real_base[:,:,:128,:128] = real_sample
            # real_data = dct.idct_2d(real_base).cuda()
            # real_data = (torch.clamp(real_data, -1, 1) + 1) / 2  # 0-1
            # torchvision.utils.save_image(
            #     real_data,'real_data.png')


            # inverse process for visualizing
            for i in range(len(diffusion_middle_feature)):
                
                DFF = diffusion_middle_feature[i]*(dmax-dmin)+dmin

                sign_feat = torch.sign(DFF)
                DFF = (torch.abs(DFF)).clamp_(max=80).exp()-1
                DFF = DFF*sign_feat

                fake_base = torch.zeros(args.batch_size,3,args.ori_image_size,args.ori_image_size)
                fake_base[:,:,:128,:128] = DFF
                print(fake_base.shape)
                fake_sample = dct.idct_2d(fake_base).cuda()
                fake_sample = (torch.clamp(fake_sample, -1, 1) + 1) / 2  # 0-1

                torchvision.utils.save_image(
                    fake_sample, './diffusion_process_samples{}_{}.jpg'.format(i,args.dataset), nrow=8, padding=0)
            print("Diffuion process outputs are saved at samples_{}.jpg".format(args.dataset))          
            

            # denoising process          
            fake_sample,mf = sample_from_model(pos_coeff, netG, args.num_timesteps, x, T, None, args)
            
            for i in range(len(mf)):
                DDF = (mf[i] +1) /2
                DDF = DDF*(dmax-dmin)+dmin
                sign_feat = torch.sign(DDF)
                DDF = (torch.abs(DDF)).clamp_(max=80).exp()-1
                DDF = DDF*sign_feat
                fake_base = torch.zeros(args.batch_size,3,args.ori_image_size,args.ori_image_size)
                fake_base[:,:,:128,:128] = DDF
                DDF = dct.idct_2d(fake_base).cuda()
                DDF = (torch.clamp(DDF, -1, 1) + 1) / 2  # 0-1

                torchvision.utils.save_image(
                    DDF, './dinoising_process_samples{}_{}.jpg'.format(i,args.dataset), nrow=8, padding=0)
            fake_sample = (fake_sample + 1) / 2 # [-1,1] => [0,1]
            fake_sample = fake_sample*(dmax-dmin)+dmin
            fake_signed = torch.sign(fake_sample)
            fake_exp = (torch.abs(fake_sample)).clamp_(max=80).exp()-1
            fake_sample = fake_exp*fake_signed
            fake_base = torch.zeros(args.batch_size,3,args.ori_image_size,args.ori_image_size)
            fake_base[:,:,:128,:128] = fake_sample
            fake_sample = dct.idct_2d(fake_base).cuda()
            fake_sample = (torch.clamp(fake_sample, -1, 1) + 1) / 2  # 0-1

            torchvision.utils.save_image(
                fake_sample, './Fake_samples.jpg', nrow=8, padding=0)
            print("Dinoising process outputs are saved at samples_{}.jpg".format(args.dataset))    
            

            # 표정인식 네트워크 들어갈 때 이미지를 n+1장 넣는다 (한 장은 input)
            




    else:
        x_t_1 = torch.randn(args.batch_size, args.num_channels,
                            args.image_size, args.image_size).to(device)
        fake_sample,mf = sample_from_model(
            pos_coeff, netG, args.num_timesteps, x_t_1,T,None,  args)

        fake_sample *= 2
        if not args.use_pytorch_wavelet:
            fake_sample = iwt(
                fake_sample[:, :3], fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12])
        else:
            fake_sample = iwt((fake_sample[:, :3], [torch.stack(
                (fake_sample[:, 3:6], fake_sample[:, 6:9], fake_sample[:, 9:12]), dim=2)]))
        fake_sample = torch.clamp(fake_sample, -1, 1)

        fake_sample = to_range_0_1(fake_sample)  # 0-1
        torchvision.utils.save_image(
            fake_sample, './samples_{}.jpg'.format(args.dataset), nrow=8, padding=0)
        print("Results are saved at samples_{}.jpg".format(args.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--inference',action='store_true',default=True)   
    parser.add_argument('--datadir', default='../')

    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    # parser.add_argument('--compute_fid', action='store_true', default=False,
    #                     help='whether or not compute FID')
    # parser.add_argument('--measure_time', action='store_true', default=False,
    #                     help='whether or not measure time')
    parser.add_argument('--epoch_id', type=int, default=900)
    parser.add_argument('--num_channels', type=int, default=18,
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
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1, 2, 2, 2, 4),
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
    parser.add_argument('--batch_size', type=int, default=4,
                        help='sample generating batch size')

    # wavelet GAN
    #parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--current_resolution", type=int, default=32)
    parser.add_argument("--net_type", default="DC")
    #parser.add_argument("--no_use_fbn", action="store_true")
    #parser.add_argument("--no_use_freq", action="store_true")
    #parser.add_argument("--no_use_residual", action="store_true")

    args = parser.parse_args()

    sample_and_test(args)
