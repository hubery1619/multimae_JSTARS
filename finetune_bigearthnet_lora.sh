# export WANDB_API_KEY=086abda60b36b2a79c0fee558059ed21d96ced9b  # 替换为你刚刚复制的 API Key
# wandb login $WANDB_API_KEY

python -m torch.distributed.launch --nproc_per_node=2 --use-env main_finetune_bigearthnet_lora.py \
--wandb satmae_bigearthnet_finetune_lora \
--batch_size 6 --accum_iter 16 --blr 0.0002 \
--epochs 1 --num_workers 2 \
--input_size 128 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--model_type group_c  \
--dataset_type bigearthnet_finetune --dropped_bands 0 9 \
--train_path dataset/bigearthnet_csv/bigearthnet_train_filtered.txt \
--test_path dataset/bigearthnet_csv/bigearthnet_val_filtered.txt \
--output_dir ./experiments/bigearthnet/finetune_lora \
--log_dir ./experiments/bigearthnet/finetune_lora \
--finetune checkpoint/crossmaeinputfm/checkpoint-49.pth \
--nb_classes 19 \
--epochs 150