export CUDA_VISIBLE_DEVICES=3,4
echo $CUDA_VISIBLE_DEVICES
export MASTER_PORT=23477
torchrun --nproc_per_node=2 \
--master_port=${MASTER_PORT} \
train_DDP_qwen_sc.py --opt config/tf-locoformer/train.yml
