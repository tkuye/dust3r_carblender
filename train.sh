NUM_NODES=$1

if [ -z "$NUM_NODES" ]; then
    NUM_NODES=1
fi

# (1024, 768), (1024, 672), (1024, 576), (1024, 512), (1024, 320)
# (512, 384), (512, 336), (512, 288), (512, 256), (512, 160)
# (768, 576), (768, 5496), (768, 432), (768, 384), (768, 240)
# (960, 720), (960, 640), (960, 544), (960, 480), (960, 288)
# (800, 600), (800, 520), (800, 440), (800, 400), (800, 240)
# (640, 480), (640, 416), (640, 352), (640, 320), (640, 192)

torchrun --nproc_per_node 1 train.py \
    --train_dataset="666_760 @ Carblender(ROOT='/home/azureuser/cloudfiles/code/Users/tosin/blender_data/data', resolution=[(640, 480), (640, 416), (640, 352), (640, 320), (640, 192)], aug_crop=16, transform=ColorJitter, split='train')" \
    --test_dataset="74_085 @ Carblender(ROOT='/home/azureuser/cloudfiles/code/Users/tosin/blender_data/data', resolution=(640, 480), split='val')" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(640, 640), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=1 --epochs=10 --batch_size=2 --accum_iter=4 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=10 --amp 1 \
    --output_dir="checkpoints/dust3r_carblender640dpt" 