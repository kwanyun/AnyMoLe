#Single GPU is enough for ICAdapt.
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12355 --node_rank=0 \
./main/trainer.py \
--base configs/training_512_v1.0/config_interp.yaml \
--train \
--name ${name} \
--logdir checkpoint_models \
--devices 1 \
lightning.trainer.num_nodes=1
