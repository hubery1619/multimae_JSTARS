export WANDB_API_KEY=086abda60b36b2a79c0fee558059ed21d96ced9b  # 替换为你刚刚复制的 API Key
wandb login $WANDB_API_KEY

CUDA_VISIBLE_DEVICES=1 python main_finetune.py \
--batch_size 32 --accum_iter 16 --blr 0.0002 \
--epochs 30 --num_workers 2 \
--input_size 96 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--dataset_type sentinel --dropped_bands 0 9 10 \
--train_path ./dataset/fmow-sentinel/train_updated.csv \
--test_path ./dataset/fmow-sentinel/val_updated.csv \
--output_dir ./experiments/mae_diffusion_finetune \
--log_dir ./experiments/mae_diffusion_finetune \
--model_type vanilla \
--finetune experiments/pretrain_diffusion/checkpoint-140.pth 
