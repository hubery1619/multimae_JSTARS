# export WANDB_API_KEY=086abda60b36b2a79c0fee558059ed21d96ced9b  # 替换为你刚刚复制的 API Key
# wandb login $WANDB_API_KEY

# --wandb satmae_eurosat_finetune \


python -m torch.distributed.launch --nproc_per_node=2 --use-env main_finetune.py \
--batch_size 12 --accum_iter 16 --blr 0.0002 \
--epochs 1 --num_workers 2 \
--input_size 96 --patch_size 8  \
--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--model_type group_c  \
--dataset_type euro_sat --dropped_bands 0 9 10 \
--train_path ./dataset/eurosat/eurosat_train_dataset.csv \
--test_path ./dataset/eurosat/eurosat_val_dataset.csv \
--output_dir ./experiments/eurosat/finetune \
--log_dir ./experiments/eurosat/finetune \
--finetune ./checkpoint/crossmaenofm/checkpoint-49.pth \
--nb_classes 10 \
--epochs 150
