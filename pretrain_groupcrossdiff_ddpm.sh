# export WANDB_API_KEY=086abda60b36b2a79c0fee558059ed21d96ced9b  # 替换为你刚刚复制的 API Key
# wandb login $WANDB_API_KEY


CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 main_pretrain_groupmaediff_ddpm.py \
--batch_size 2 --accum_iter 4 --blr 0.0001 \
--epochs 200 --warmup_epochs 20 --num_workers 16 \
--model crossmaediffddpm_vit_large_patch16 \
--input_size 96 --patch_size 8 \
--mask_ratio 0.75 \
--wandb group_crossMAE_diff_ddpm \
--model_type groupmaediff_ddpm \
--dataset_type sentinel --dropped_bands 0 9 10 \
--grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 \
--train_path ./dataset/fmow-sentinel/train_updated.csv \
--output_dir ./experiments/group_crossMAE_diffusion_ddpm \
--log_dir ./experiments/group_crossMAE_diffusion_ddpm \
--norm_pix_loss