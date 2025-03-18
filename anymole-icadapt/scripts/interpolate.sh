#!/bin/bash

# Assigning the arguments to variables
cond=$1
repstart=$2
repduration=$3
model_name=$4

ckpt=checkpoints/dynamicrafter_512_interp_v1/${model_name}.ckpt
config=configs/inference_512_v1.0.yaml

prompt_dir=prompts/512_interp/
res_dir="results"

FS=5 ## This model adopts FPS=5, range recommended: 5-30 (smaller value -> larger motion)

seed=12306
name=dynamicrafter_512_seed${seed}
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae --interp \
--cond ${cond} \
--repstart ${repstart} \
--repduration ${repduration}