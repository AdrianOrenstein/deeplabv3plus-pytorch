#!bin/bash

# Script for running network training
python train_segmentation.py \
	--dataset_name fruit \
	--dataset_dir ../../Downloads/ \
	--decoder refinenet \
	--encoder resnet_18 \
	--imagenet \
	--batch_size 64 \
	--val_batch_size 32 \
	--val_interval 10 \
	--lr 0.03 \
	--decay 1e-5 \
	--momentum 0.9 \
	--num_epochs 300 \
	--gradient_ckpt \
	--amp_level "02" \
	--gpus "0"
