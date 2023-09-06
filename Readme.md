#### Train
We provide the training code.
using [train.py](train.py), You will end up running something like `python train.py --dataset childFER --image_size 256 --exp childFER 
--num_channels 18 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 1000 
--ngf 64 --embedding_type positional --r1_gamma 2. --z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 
--save_content --datadir datasets/childFER --current_resolution 64 --attn_resolution 16 --num_disc_layers 5 --rec_loss --save_content_every 50 --net_type DC`

#### Test
We provide the testing code.
using [test.py](test.py), You will end up running something like `python test.py --dataset childFER --exp childFER 
--num_channels 18 --num_channels_dae 64 --num_timesteps 4 --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 3 --ch_mult 1 2 2 2 4 --epoch_id 1000 
--image_size 256 --current_resolution 64 --attn_resolutions 16 --inference --datadir childFER`
