#!/bin/bash
### gpu 分区
#SBATCH -p gpu
### 该作业的作业名
#SBATCH --job-name=ucf_DM_Batch54_lr2e-4_c4p4_region64_resume_lr_edited
### 该作业需要1个节点
#SBATCH --nodes=1
### 该作业需要CPU（注意！默认只允许1块GPU最多匹配3个CPU核心）
#SBATCH --ntasks=4
### 申请GPU卡跑程序
#SBATCH --gres=gpu:1
### 待运行程序的代码目录 (uxxxxxxxx为个人用户名)
#SBATCH -D /home/u1120230288/zzc/code/video_prediction/EDM

### !!!!!! 非常重要，激活conda命令，否则系统无法找到
source ~/.bashrc

### 举例命令，请修改为自己的：
conda activate EDM
sh ./scripts/diffusion/train_diffusion_ucf.sh
export WANDB_API_KEY=770170a626cba9cffc79070d08907e5ccd0c256a
### 启动集群 sbatch -s ./scripts/diffusion/slurm_ucf.sh