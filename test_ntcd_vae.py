#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的 VAE 模型在 NTCD-TIMIT 测试集上进行语音增强测试
"""

import torch
import numpy as np
import soundfile as sf
import librosa
import torch.nn as nn
from pathlib import Path
import argparse
import os

from MCEM_algo import MCEM_algo
from AV_VAE import myVAE, myDecoder
from sisnr import si_snr, si_snri


def test_ntcd_vae(
    test_audio_dir='/mnt/d/data/NTCD-TIMIT/avse/validation/speech',
    test_video_dir='/mnt/d/data/NTCD-TIMIT/avse/validation/video',
    model_path='/mnt/d/data/NTCD-TIMIT/avse/saved_model/ntcd_timit_quicktest_checkpoint.pt',
    result_dir='./results',
    device='cpu',
    max_files=None  # 限制测试文件数（用于快速测试）
):
    """
    测试训练好的 VAE 模型
    
    Args:
        test_audio_dir: 测试音频目录
        test_video_dir: 测试视频特征目录
        model_path: 训练好的模型路径
        result_dir: 结果保存目录
        device: 运行设备 ('cpu' 或 'cuda')
        max_files: 最多测试的文件数（None 表示全部）
    """
    
    print("=" * 80)
    print("使用 NTCD-TIMIT 测试集评估 VAE 模型")
    print("=" * 80)
    
    # ========== 1. 网络参数 ==========
    input_dim = 513
    latent_dim = 32
    hidden_dim_encoder = [128]
    activation = torch.tanh
    activationv = nn.ReLU()
    landmarks_dim = 67 * 67  # 原始视频特征维度 4489
    
    # ========== 2. STFT 参数 ==========
    wlen_sec = 64e-3
    hop_percent = 0.521
    fs = 16000
    wlen = int(wlen_sec * fs)
    wlen = int(np.power(2, np.ceil(np.log2(wlen))))
    nfft = wlen
    hop = int(hop_percent * wlen)
    win = np.sin(np.arange(.5, wlen - .5 + 1) / wlen * np.pi)
    
    # ========== 3. MCEM 参数 ==========
    niter_MCEM = 50  # 测试时可以减少迭代次数以加快速度
    niter_MH = 30
    burnin = 20
    var_MH = 0.01
    tol = 1e-5
    K_b = 10  # NMF rank for noise model
    
    # ========== 4. 加载模型 ==========
    print(f"\n加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型（音视频联合 VAE）
    vae = myVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder,
        activation=activation,
        activationv=activationv,
        blockZ=0.,
        blockVenc=0.,  # 允许视频信息通过编码器
        blockVdec=0.,  # 允许视频信息通过解码器
        x_block=0.,
        landmarks_dim=landmarks_dim
    ).to(device)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    decoder = myDecoder(vae)
    
    # 设置为评估模式
    vae.eval()
    decoder.eval()
    
    # 冻结参数
    for param in decoder.parameters():
        param.requires_grad = False
    
    print(f"✓ 模型加载成功")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    
    # ========== 5. 获取测试文件列表 ==========
    print(f"\n扫描测试数据...")
    print(f"  音频目录: {test_audio_dir}")
    print(f"  视频目录: {test_video_dir}")
    
    test_audio_path = Path(test_audio_dir)
    test_video_path = Path(test_video_dir)
    
    if not test_audio_path.exists():
        raise FileNotFoundError(f"测试音频目录不存在: {test_audio_dir}")
    
    if not test_video_path.exists():
        raise FileNotFoundError(f"测试视频目录不存在: {test_video_dir}")
    
    # 查找所有测试音频文件
    audio_files = sorted(list(test_audio_path.rglob('*.wav')))
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"\n找到 {len(audio_files)} 个测试文件")
    if max_files:
        print(f"  (限制为前 {max_files} 个文件)")
    
    # ========== 6. 创建结果目录 ==========
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    
    # ========== 7. 逐文件测试 ==========
    print("\n" + "=" * 80)
    print("开始测试...")
    print("=" * 80)
    
    all_si_snri = []
    failed_files = []
    
    for idx, audio_file in enumerate(audio_files):
        try:
            print(f"\n[{idx+1}/{len(audio_files)}] 处理: {audio_file.name}")
            
            # 构建视频特征文件路径
            # 音频: .../validation/speech/08F/sa1.wav
            # 视频: .../validation/video/08F/sa1Raw.npy
            speaker = audio_file.parent.name
            file_id = audio_file.stem  # sa1, si1004, etc.
            
            video_file = test_video_path / speaker / f'{file_id}Raw.npy'
            
            if not video_file.exists():
                print(f"  ⚠️  跳过: 视频特征文件不存在 {video_file}")
                failed_files.append((audio_file.name, "视频特征缺失"))
                continue
            
            # 读取音频和视频
            x, fs_read = librosa.load(str(audio_file), sr=fs)
            x = x / np.max(np.abs(x))  # 归一化
            T_orig = len(x)
            
            v = np.load(str(video_file))
            
            # STFT
            X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
            X_abs_2 = np.abs(X) ** 2
            
            F, N = X.shape
            
            # 调整视频帧数
            Nl = max(N, v.shape[1])
            if v.shape[1] < Nl:
                v = np.hstack((v, np.tile(v[:, [-1]], Nl - v.shape[1])))
            
            v = v.T
            v = torch.from_numpy(v.astype(np.float32))
            v.requires_grad = False
            
            # 初始化 NMF 参数
            eps = np.finfo(float).eps
            np.random.seed(idx)  # 使用索引作为种子保证可重复性
            W0 = np.maximum(np.random.rand(F, K_b), eps)
            H0 = np.maximum(np.random.rand(K_b, N), eps)
            
            # 使用 VAE 编码器初始化潜在变量
            with torch.no_grad():
                data = X_abs_2.T
                data = torch.from_numpy(data.astype(np.float32)).to(device)
                z, _ = vae.encode(data, v.to(device))
                z = torch.t(z)
            
            Z_init = z.cpu().numpy()
            
            # 实例化 MCEM 算法
            mcem_algo = MCEM_algo(
                X=X,
                W=W0,
                H=H0,
                Z=Z_init,
                v=v,
                decoder=decoder,
                niter_MCEM=niter_MCEM,
                niter_MH=niter_MH,
                burnin=burnin,
                var_MH=var_MH
            )
            
            # 运行 MCEM
            print(f"  运行 MCEM 算法...")
            cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)
            print(f"    完成，迭代次数: {niter_final}")
            
            # 分离语音和噪声
            mcem_algo.separate(niter_MH=50, burnin=30)
            
            # ISTFT 重建时域信号
            s_hat = librosa.istft(
                stft_matrix=mcem_algo.S_hat,
                hop_length=hop,
                win_length=wlen,
                window=win,
                length=T_orig
            )
            
            b_hat = librosa.istft(
                stft_matrix=mcem_algo.N_hat,
                hop_length=hop,
                win_length=wlen,
                window=win,
                length=T_orig
            )
            
            # 保存结果
            output_dir = result_path / speaker
            output_dir.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(output_dir / f'{file_id}_enhanced.wav'), s_hat, fs)
            sf.write(str(output_dir / f'{file_id}_noise.wav'), b_hat, fs)
            
            # 计算 SI-SNRi（假设原始音频是干净的）
            si_snri_val, si_snr_noisy, si_snr_enhanced = si_snri(x, x, s_hat)
            all_si_snri.append(si_snri_val)
            
            print(f"  ✓ 完成")
            print(f"    SI-SNR (输入): {si_snr_noisy:.2f} dB")
            print(f"    SI-SNR (增强): {si_snr_enhanced:.2f} dB")
            print(f"    SI-SNRi: {si_snri_val:.2f} dB")
            
            # 保存指标
            with open(output_dir / f'{file_id}_metrics.txt', 'w') as f:
                f.write(f'File: {audio_file.name}\n')
                f.write(f'SI-SNR (input): {si_snr_noisy:.2f} dB\n')
                f.write(f'SI-SNR (enhanced): {si_snr_enhanced:.2f} dB\n')
                f.write(f'SI-SNRi: {si_snri_val:.2f} dB\n')
                f.write(f'MCEM iterations: {niter_final}\n')
        
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)[:100]}")
            failed_files.append((audio_file.name, str(e)[:100]))
            continue
    
    # ========== 8. 汇总结果 ==========
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    if all_si_snri:
        mean_si_snri = np.mean(all_si_snri)
        std_si_snri = np.std(all_si_snri)
        print(f"\n总体结果 ({len(all_si_snri)} 个文件):")
        print(f"  平均 SI-SNRi: {mean_si_snri:.2f} ± {std_si_snri:.2f} dB")
        
        # 保存总体统计
        with open(result_path / 'summary.txt', 'w') as f:
            f.write(f'Total files processed: {len(all_si_snri)}\n')
            f.write(f'Mean SI-SNRi: {mean_si_snri:.2f} dB\n')
            f.write(f'Std SI-SNRi: {std_si_snri:.2f} dB\n')
            f.write(f'\nPer-file SI-SNRi:\n')
            for idx, val in enumerate(all_si_snri):
                f.write(f'  File {idx+1}: {val:.2f} dB\n')
    
    if failed_files:
        print(f"\n失败文件 ({len(failed_files)} 个):")
        for fname, reason in failed_files:
            print(f"  {fname}: {reason}")
        
        with open(result_path / 'failed_files.txt', 'w') as f:
            for fname, reason in failed_files:
                f.write(f'{fname}: {reason}\n')
    
    print(f"\n结果保存在: {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试 NTCD-TIMIT VAE 模型')
    parser.add_argument('--test_audio', type=str,
                        default='/mnt/d/data/NTCD-TIMIT/avse/validation/speech',
                        help='测试音频目录')
    parser.add_argument('--test_video', type=str,
                        default='/mnt/d/data/NTCD-TIMIT/avse/validation/video',
                        help='测试视频特征目录')
    parser.add_argument('--model', type=str,
                        default='/mnt/d/data/NTCD-TIMIT/avse/saved_model/ntcd_timit_quicktest_checkpoint.pt',
                        help='训练好的模型路径')
    parser.add_argument('--result_dir', type=str,
                        default='./results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='运行设备')
    parser.add_argument('--max_files', type=int, default=None,
                        help='最多测试的文件数（用于快速测试）')
    
    args = parser.parse_args()
    
    test_ntcd_vae(
        test_audio_dir=args.test_audio,
        test_video_dir=args.test_video,
        model_path=args.model,
        result_dir=args.result_dir,
        device=args.device,
        max_files=args.max_files
    )
