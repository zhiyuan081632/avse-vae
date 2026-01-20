#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 NTCD-TIMIT 数据集训练 VAE 模型的示例脚本
演示如何使用 TCD_TIMIT.py 加载数据
"""

import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
import os
from pathlib import Path

# 导入项目模块
from TCD_TIMIT import TIMIT
from AV_VAE import myVAE
from pytorchtools import EarlyStopping

def get_file_list(data_dir):
    """
    获取指定目录下所有 WAV 文件的路径列表
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        WAV 文件路径列表
    """
    file_list = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 递归查找所有 .wav 文件
    for wav_file in data_path.rglob('*.wav'):
        file_list.append(str(wav_file.absolute()))
    
    file_list.sort()
    
    print(f"在 {data_dir} 中找到 {len(file_list)} 个音频文件")
    
    return file_list


def train_vae_with_ntcd_timit(quick_test=False):
    """
    使用 NTCD-TIMIT 数据集训练 VAE 模型
    
    Args:
        quick_test: 快速测试模式，使用较小的 batch_size 和 epochs
    """
    
    print("=" * 80)
    if quick_test:
        print("🚀 快速测试模式 - 使用 NTCD-TIMIT 数据集训练 VAE 模型")
    else:
        print("使用 NTCD-TIMIT 数据集训练 VAE 模型")
    print("=" * 80)
    
    # ========== 1. 设置路径 ==========
    
    # 数据根目录 (通过 prepare_ntcd_timit_data.py 准备)
    base_dir = Path('/mnt/d/data/NTCD-TIMIT/avse')
    
    # 训练和验证数据目录
    data_dir_tr = base_dir / 'training' / 'speech'
    data_dir_val = base_dir / 'validation' / 'speech'
    
    print(f"\n训练数据目录: {data_dir_tr}")
    print(f"验证数据目录: {data_dir_val}")
    
    # ========== 2. 获取文件列表 ==========
    
    print("\n" + "=" * 80)
    print("加载数据文件列表...")
    print("=" * 80)
    
    try:
        file_list_tr = get_file_list(data_dir_tr)
        file_list_val = get_file_list(data_dir_val)
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请先运行数据准备脚本:")
        print("  python prepare_ntcd_timit_data.py --noise_type Clean --speaker_type volunteers")
        return
    
    if len(file_list_tr) == 0:
        print("\n错误: 训练数据为空，请先准备数据")
        return
    
    if len(file_list_val) == 0:
        print("\n警告: 验证数据为空")
    
    print(f"\n训练集文件数: {len(file_list_tr)}")
    print(f"验证集文件数: {len(file_list_val)}")
    print(f"\n第一个训练文件: {file_list_tr[0]}")
    if file_list_val:
        print(f"第一个验证文件: {file_list_val[0]}")
    
    # ========== 3. STFT 参数 ==========
    
    wlen_sec = 64e-3      # STFT 窗口长度 64ms
    hop_percent = 0.521   # 跳跃比例 52.1%
    fs = 16000           # 采样率 16kHz
    zp_percent = 0       # 零填充比例
    trim = False         # 是否修剪静音
    verbose = False
    
    # ========== 4. 网络参数 ==========
    
    input_dim = 513              # STFT 频点数 (nfft/2 + 1)
    latent_dim = 32             # 潜在空间维度
    hidden_dim_encoder = [128]  # 编码器隐藏层维度
    activation = torch.tanh     # 音频层激活函数
    activationv = nn.ReLU()     # 视频层激活函数
    
    # ========== 5. 训练参数 ==========
    
    if quick_test:
        # 快速测试模式：小批次、少轮数
        batch_size = 32
        epochs = 5
        print(f"\n🚀 快速测试模式：")
        print(f"   batch_size = {batch_size} (减小以加快训练)")
        print(f"   epochs = {epochs} (减少以快速验证)")
    else:
        # 正常训练模式
        batch_size = 128
        epochs = 50  # 演示用，实际训练可设置为 200
    
    lr = 1e-4
    num_workers = 0
    shuffle_file_list = True
    shuffle_samples_in_batch = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # ========== 6. 创建数据加载器 ==========
    
    print("\n" + "=" * 80)
    print("创建数据加载器...")
    print("=" * 80)
    
    # 创建训练数据集
    # video_part=False 表示不使用视频数据（纯音频训练）
    train_dataset = TIMIT(
        data_mode='training',
        file_list=file_list_tr,
        wlen_sec=wlen_sec,
        hop_percent=hop_percent,
        fs=fs,
        zp_percent=zp_percent,
        trim=trim,
        verbose=verbose,
        batch_size=batch_size,
        shuffle_file_list=shuffle_file_list,
        video_part=True  # 设置为 True 表示使用音频和视频
    )
    
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_samples_in_batch,
        num_workers=num_workers
    )
    
    print(f"训练数据集大小: {len(train_dataset)} 帧")
    print(f"注意: video_part=False，视频输入将使用音频数据副本")
    
    # 创建验证数据集
    if file_list_val:
        val_dataset = TIMIT(
            data_mode='validation',
            file_list=file_list_val,
            wlen_sec=wlen_sec,
            hop_percent=hop_percent,
            fs=fs,
            zp_percent=zp_percent,
            trim=trim,
            verbose=verbose,
            batch_size=batch_size,
            shuffle_file_list=shuffle_file_list,
            video_part=False
        )
        
        val_dataloader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_samples_in_batch,
            num_workers=num_workers
        )
        
        print(f"验证数据集大小: {len(val_dataset)} 帧")
    else:
        val_dataloader = None
    
    print(f"\n⚠️  重要: 当前为音视频联合模式 (video_part=True)")
    print(f"   模型将使用视频信息进行联合训练 (blockVenc=0.0, blockVdec=0.0)")
    
    # ========== 7. 创建模型 ==========
    
    print("\n" + "=" * 80)
    print("创建 VAE 模型...")
    print("=" * 80)
    
    # blockVenc=0.0, blockVdec=0.0 表示启用视频路径，音视频联合训练
    vae = myVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder,
        activation=activation,
        activationv=activationv,
        blockZ=0.0,      # 0=使用潜在变量 z
        blockVenc=0.0,   # 0=启用编码器视频路径
        blockVdec=0.0,   # 0=启用解码器视频路径
        x_block=0.0,
        landmarks_dim=4489  # 视频特征维度 (67x67=4489)
    ).to(device)
    
    print(f"模型参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  潜在维度: {latent_dim}")
    print(f"  隐藏层维度: {hidden_dim_encoder}")
    print(f"  视频特征维度: 4489 (67x67)")
    print(f"  音视频联合训练: blockVenc=0.0, blockVdec=0.0")
    
    # 优化器
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    # 损失函数
    def loss_function(recon_xi, xi, mui, logvari):
        # 重构损失（负对数似然）
        recon = torch.sum(torch.log(recon_xi) + xi / recon_xi)
        # KL 散度
        KLD = -0.5 * torch.sum(logvari - mui.pow(2) - logvari.exp())
        return recon + KLD
    
    # ========== 8. 训练循环 ==========
    
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    # Early Stopping
    save_dir = base_dir / 'saved_model'
    save_dir.mkdir(exist_ok=True)
    
    if quick_test:
        checkpoint_path = save_dir / 'ntcd_timit_quicktest_checkpoint.pt'
        early_stopping = EarlyStopping(save_dir=str(checkpoint_path), patience=3)  # 减小 patience
    else:
        checkpoint_path = save_dir / 'ntcd_timit_checkpoint.pt'
        early_stopping = EarlyStopping(save_dir=str(checkpoint_path), patience=10)
    
    for epoch in range(epochs):
        # 训练模式
        vae.train()
        train_losses = []
        
        for batch_idx, (batch_audio, batch_video) in enumerate(train_dataloader):
            batch_audio = batch_audio.to(device)
            batch_video = batch_video.to(device)
            
            # 前向传播
            recon_batch, mu, logvar = vae(batch_audio, batch_video)
            loss = loss_function(recon_batch, batch_audio, mu, logvar)
            
            train_losses.append(loss.item())
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 每 10 个 batch 打印一次
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        train_loss = sum(train_losses) / len(train_dataset)
        
        # 验证
        if val_dataloader is not None:
            vae.eval()
            valid_losses = []
            
            with torch.no_grad():
                for batch_audio, batch_video in val_dataloader:
                    batch_audio = batch_audio.to(device)
                    batch_video = batch_video.to(device)
                    
                    recon_batch, mu, logvar = vae(batch_audio, batch_video)
                    loss = loss_function(recon_batch, batch_audio, mu, logvar)
                    valid_losses.append(loss.item())
            
            valid_loss = sum(valid_losses) / len(val_dataset)
        else:
            valid_loss = train_loss
        
        # 打印统计
        print(f"\n====> Epoch: [{epoch+1}/{epochs}] "
              f"train_loss: {train_loss:.5f} "
              f"valid_loss: {valid_loss:.5f}\n")
        
        # Early stopping
        early_stopping(train_loss, valid_loss, vae, epoch, optimizer)
        
        if early_stopping.early_stop:
            print("Early stopping 触发，停止训练")
            break
    
    # ========== 9. 保存最终模型 ==========
    
    if quick_test:
        final_model_path = save_dir / 'final_model_ntcd_timit_quicktest.pt'
    else:
        final_model_path = save_dir / 'final_model_ntcd_timit.pt'
    
    torch.save(vae.state_dict(), final_model_path)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"最终模型已保存到: {final_model_path}")
    print(f"最佳检查点: {checkpoint_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用 NTCD-TIMIT 数据集训练 VAE 模型')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式：使用小批次(32)和少轮数(5)快速验证训练流程')
    
    args = parser.parse_args()
    
    train_vae_with_ntcd_timit(quick_test=args.quick_test)
