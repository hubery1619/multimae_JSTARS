export WANDB_API_KEY=086abda60b36b2a79c0fee558059ed21d96ced9b  # 替换为你刚刚复制的 API Key
wandb login $WANDB_API_KEY


torchrun --nproc_per_node=1 main_pretrain_diffusion_integrate.py \
--batch_size 128 --accum_iter 4 --blr 0.0001 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--model mae_diffusion_integrate_vit_large_patch16 \
--input_size 96 --patch_size 8 \
--mask_ratio 0.75 \
--wandb MAE_diffusion_integrate \
--model_type MAE_diffusion_integrate \
--dataset_type sentinel --dropped_bands 0 9 10 \
--grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
--train_path ./dataset/fmow-sentinel/train_updated.csv \
--output_dir ./experiments/MAE_diffusion_integrate \
--log_dir ./experiments/MAE_diffusion_integrate \
--norm_pix_loss