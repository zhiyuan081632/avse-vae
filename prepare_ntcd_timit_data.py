#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTCD-TIMIT 数据准备脚本
用于将原始 NTCD-TIMIT 数据集整理成 TCD_TIMIT.py 可以使用的格式
"""

import os
import shutil
from pathlib import Path
import numpy as np
import scipy.io as sio
import h5py

def convert_speaker_video_features(
    speaker_id,
    matlab_video_root,
    output_dir,
    is_training=True,
    noise_type='Clean'
):
    """
    转换单个说话人的视频特征从 MATLAB .mat 到 .npy
    
    Args:
        speaker_id: 说话人ID (如 '01M', '08F')
        matlab_video_root: MATLAB视频特征根目录 (包含 train/dev/test)
        output_dir: 输出目录
        is_training: 是否为训练集
        noise_type: 噪声类型 (用于训练集路径)
    
    Returns:
        转换成功的文件数
    """
    # 在 train/dev/test 中查找该说话人的特征
    matlab_root = Path(matlab_video_root)
    converted_count = 0
    
    for split in ['train', 'dev', 'test']:
        split_dir = matlab_root / split / speaker_id
        if not split_dir.exists():
            continue
        
        # 找到该说话人目录，转换所有 .mat 文件
        mat_files = list(split_dir.glob('*.mat'))
        if not mat_files:
            continue
        
        print(f"  找到 {speaker_id} 在 {split} 中有 {len(mat_files)} 个视频特征文件")
        
        for mat_file in mat_files:
            try:
                # 读取 MATLAB 特征
                features = None
                file_id = mat_file.stem  # 'sa1', 'si1004', etc.
                
                try:
                    # 尝试 scipy.io (MATLAB v7)
                    mat_data = sio.loadmat(str(mat_file))
                    var_names = [k for k in mat_data.keys() if not k.startswith('__')]
                    if var_names:
                        features = mat_data[var_names[0]]
                except NotImplementedError:
                    # MATLAB v7.3，使用 h5py
                    with h5py.File(str(mat_file), 'r') as f:
                        var_names = [k for k in f.keys() if not k.startswith('#')]
                        if var_names:
                            features = f[var_names[0]][()]
                
                if features is None:
                    continue
                
                # 转换为 float32 并确保是 (D, T) 格式
                features = features.astype(np.float32)
                if features.ndim == 2 and features.shape[1] > features.shape[0]:
                    features = features.T
                
                # 保存为 .npy
                if is_training:
                    # 训练集: output_dir/speech/Clean/01M/sa1Raw.npy
                    output_subdir = output_dir / 'video' / noise_type / speaker_id
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    output_file = output_subdir / f'{file_id}Raw.npy'
                else:
                    # 验证集: output_dir/video/08F/sa1Raw.npy
                    output_subdir = output_dir / 'video' / speaker_id
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    output_file = output_subdir / f'{file_id}Raw.npy'
                
                np.save(output_file, features)
                converted_count += 1
                
            except Exception as e:
                print(f"    ⚠️  转换失败 {mat_file.name}: {str(e)[:80]}")
        
        # 找到就不再继续搜索其它split
        if mat_files:
            break
    
    return converted_count


def organize_ntcd_timit_data(
    source_dir='/mnt/d/data/NTCD-TIMIT/TCDTIMIT',
    target_base_dir='/mnt/d/data/NTCD-TIMIT/avse',
    matlab_video_root='/mnt/d/data/NTCD-TIMIT/matlab_raw',  # MATLAB视频特征根目录
    noise_type='Clean',  # 'Clean', 'Babble', 'Cafe', 'Car', 'LR', 'Street', 'White'
    snr_level=None,      # None for Clean, or -5, 0, 5, 10, 15, 20 for noisy
    speaker_type='volunteers',  # 'volunteers' or 'lipspeakers'
    train_speakers=None,
    val_speakers=None,
    quick_test=False,    # 快速测试模式：只使用少量数据
    max_files_per_speaker=None,  # 每个说话人最多复制的文件数（用于快速测试）
    convert_video=True   # 是否同时转换视频特征
):
    """
    整理 NTCD-TIMIT 数据集为训练所需的目录结构
    
    预期的源数据结构:
    source_dir/
    ├── Clean/
    │   ├── volunteers/
    │   │   ├── 01M/straightcam/
    │   │   │   ├── sa1.wav
    │   │   │   ├── sa2.wav
    │   │   │   └── ...
    │   └── lipspeakers/
    └── Babble/
        ├── -5/
        │   ├── volunteers/
        │   └── lipspeakers/
        └── 0/
    
    目标数据结构 (符合 TCD_TIMIT.py 期望):
    target_base_dir/
    ├── training/
    │   ├── speech/
    │   │   └── Clean/
    │   │       ├── 01M/
    │   │       │   ├── sa1.wav
    │   │       │   └── ...
    │   └── video/
    │       └── Clean/
    │           ├── 01M/
    │           │   ├── sa1Raw.npy
    │           │   └── ...
    └── validation/
        ├── speech/
        │   └── 08F/
        │       ├── sa1.wav
        │       └── ...
        └── video/
            └── 08F/
                ├── sa1Raw.npy
                └── ...
    """
    
    # 快速测试模式：只使用2个训练说话人和1个验证说话人
    if quick_test:
        if train_speakers is None:
            train_speakers = ['01M', '02M']  # 只用2个说话人
        if val_speakers is None:
            # 使用 dev 目录中的说话人（对应 matlab_raw/dev/）
            val_speakers = ['08F']  # dev 目录中的说话人
        if max_files_per_speaker is None:
            max_files_per_speaker = 5  # 每个说话人只用5个文件
        print(f"\n🚀 快速测试模式已启用！")
        print(f"   训练说话人: {train_speakers}")
        print(f"   验证说话人: {val_speakers}")
        print(f"   每人最多 {max_files_per_speaker} 个文件\n")
    else:
        # 默认训练/验证说话人划分 (前80%训练，后20%验证)
        if train_speakers is None:
            train_speakers = [f'{i:02d}M' for i in range(1, 17)] + \
                            [f'{i:02d}F' for i in range(3, 16, 2)]
        
        if val_speakers is None:
            val_speakers = [f'{i:02d}M' for i in range(17, 21)] + \
                          [f'{i:02d}F' for i in range(17, 20)]
    
    print(f"训练说话人: {train_speakers}")
    print(f"验证说话人: {val_speakers}")
    
    # 构建源数据路径
    source_path = Path(source_dir)
    
    # 如果 source_dir 不直接包含说话人目录，则需要添加噪声类型和说话人类型
    # 检查是否直接包含说话人目录（如 01M, 02M 等）
    test_speakers = ['01M', '02M', '03F']
    has_speaker_dirs = any((source_path / s).exists() for s in test_speakers)
    
    if not has_speaker_dirs:
        # 需要添加噪声类型和说话人类型路径
        if noise_type == 'Clean':
            source_path = source_path / 'Clean' / speaker_type
        else:
            if snr_level is None:
                raise ValueError(f"噪声类型 '{noise_type}' 需要指定 SNR 等级")
            source_path = source_path / noise_type / str(snr_level) / speaker_type
    
    if not source_path.exists():
        raise FileNotFoundError(f"源数据路径不存在: {source_path}")
    
    print(f"\n源数据路径: {source_path}")
    
    # 创建目标目录结构
    target_base = Path(target_base_dir)
    train_speech_dir = target_base / 'training' / 'speech' / noise_type
    val_speech_dir = target_base / 'validation' / 'speech'
    
    train_speech_dir.mkdir(parents=True, exist_ok=True)
    val_speech_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"训练音频目录: {train_speech_dir}")
    print(f"验证音频目录: {val_speech_dir}")
    
    # 统计信息
    train_count = 0
    val_count = 0
    video_train_count = 0
    video_val_count = 0
    
    # 视频特征输出目录
    train_video_dir = target_base / 'training'
    val_video_dir = target_base / 'validation'
    
    if convert_video:
        matlab_video_path = Path(matlab_video_root)
        if not matlab_video_path.exists():
            print(f"\n⚠️  警告: MATLAB视频特征目录不存在: {matlab_video_path}")
            print(f"   将跳过视频特征转换")
            convert_video = False
        else:
            print(f"\nMAT视频特征目录: {matlab_video_path}")
            print(f"训练视频输出: {train_video_dir / 'video'}")
            print(f"验证视频输出: {val_video_dir / 'video'}")
    
    # 遍历所有说话人
    for speaker_dir in sorted(source_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        speaker_id = speaker_dir.name
        
        # 判断是训练集还是验证集
        if speaker_id in train_speakers:
            target_speaker_dir = train_speech_dir / speaker_id
            is_training = True
        elif speaker_id in val_speakers:
            target_speaker_dir = val_speech_dir / speaker_id
            is_training = False
        else:
            print(f"跳过说话人: {speaker_id}")
            continue
        
        target_speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找 straightcam 目录下的音频文件
        straightcam_dir = speaker_dir / 'straightcam'
        if not straightcam_dir.exists():
            print(f"警告: {speaker_id} 没有 straightcam 目录")
            continue
        
        # 复制所有 wav 文件
        wav_files = list(straightcam_dir.glob('*.wav'))
        
        # 如果设置了最大文件数限制，只复制前N个
        if max_files_per_speaker is not None:
            wav_files = wav_files[:max_files_per_speaker]
        
        for wav_file in wav_files:
            target_file = target_speaker_dir / wav_file.name
            
            # 使用符号链接节省空间（如果支持的话），否则复制
            try:
                if not target_file.exists():
                    # 在 Windows 上可能需要管理员权限创建符号链接
                    # 所以这里直接复制文件
                    shutil.copy2(wav_file, target_file)
                    if is_training:
                        train_count += 1
                    else:
                        val_count += 1
            except Exception as e:
                print(f"处理文件失败 {wav_file}: {e}")
        
        # 转换该说话人的视频特征
        if convert_video:
            print(f"  转换 {speaker_id} 的视频特征...")
            if is_training:
                video_count = convert_speaker_video_features(
                    speaker_id,
                    matlab_video_root,
                    train_video_dir,
                    is_training=True,
                    noise_type=noise_type
                )
                video_train_count += video_count
                print(f"    ✓ 已转换 {video_count} 个视频特征文件")
            else:
                video_count = convert_speaker_video_features(
                    speaker_id,
                    matlab_video_root,
                    val_video_dir,
                    is_training=False
                )
                video_val_count += video_count
                print(f"    ✓ 已转换 {video_count} 个视频特征文件")
    
    print(f"\n数据整理完成!")
    print(f"训练集音频文件数: {train_count}")
    print(f"验证集音频文件数: {val_count}")
    if convert_video:
        print(f"训练集视频特征数: {video_train_count}")
        print(f"验证集视频特征数: {video_val_count}")
    print(f"\n现在可以使用 TCD_TIMIT.py 加载数据:")
    print(f"  训练集路径: {train_speech_dir}")
    print(f"  验证集路径: {val_speech_dir}")
    if convert_video:
        print(f"  训练视频路径: {train_video_dir / 'video'}")
        print(f"  验证视频路径: {val_video_dir / 'video'}")


def create_file_list(data_dir, output_file):
    """
    创建音频文件列表，供训练脚本使用
    
    Args:
        data_dir: 数据目录路径
        output_file: 输出文件列表路径
    """
    data_path = Path(data_dir)
    wav_files = []
    
    for wav_file in data_path.rglob('*.wav'):
        wav_files.append(str(wav_file.absolute()))
    
    wav_files.sort()
    
    # 保存文件列表
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for wav_file in wav_files:
            f.write(wav_file + '\n')
    
    print(f"文件列表已保存到: {output_path}")
    print(f"文件总数: {len(wav_files)}")
    
    return wav_files


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='准备 NTCD-TIMIT 数据集')
    parser.add_argument('--source_dir', type=str, 
                        default='/mnt/d/data/NTCD-TIMIT/TCDTIMIT',
                        help='NTCD-TIMIT 源数据目录')
    parser.add_argument('--target_dir', type=str,
                        default='/mnt/d/data/NTCD-TIMIT/avse',
                        help='目标数据目录')
    parser.add_argument('--noise_type', type=str, default='Clean',
                        choices=['Clean', 'Babble', 'Cafe', 'Car', 'LR', 'Street', 'White'],
                        help='噪声类型')
    parser.add_argument('--snr_level', type=int, default=None,
                        choices=[-5, 0, 5, 10, 15, 20],
                        help='SNR等级 (仅噪声数据需要)')
    parser.add_argument('--speaker_type', type=str, default='volunteers',
                        choices=['volunteers', 'lipspeakers'],
                        help='说话人类型')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式：只使用少量数据（2个训练说话人，1个验证说话人，每人5个文件）')
    parser.add_argument('--max_files', type=int, default=None,
                        help='每个说话人最多复制的文件数（用于自定义快速测试）')
    parser.add_argument('--matlab_video_root', type=str,
                        default='/mnt/d/data/NTCD-TIMIT/matlab_raw',
                        help='MATLAB 视频特征根目录')
    parser.add_argument('--no_video', action='store_true',
                        help='不转换视频特征')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NTCD-TIMIT 数据准备工具")
    print("=" * 60)
    print(f"噪声类型: {args.noise_type}")
    if args.snr_level is not None:
        print(f"SNR等级: {args.snr_level} dB")
    print(f"说话人类型: {args.speaker_type}")
    print("=" * 60)
    
    # 整理数据
    organize_ntcd_timit_data(
        source_dir=args.source_dir,
        target_base_dir=args.target_dir,
        matlab_video_root=args.matlab_video_root,
        noise_type=args.noise_type,
        snr_level=args.snr_level,
        speaker_type=args.speaker_type,
        quick_test=args.quick_test,
        max_files_per_speaker=args.max_files,
        convert_video=not args.no_video
    )
    
    # 创建文件列表
    train_dir = Path(args.target_dir) / 'training' / 'speech' / args.noise_type
    val_dir = Path(args.target_dir) / 'validation' / 'speech'
    
    print("\n" + "=" * 60)
    print("创建文件列表...")
    print("=" * 60)
    
    if train_dir.exists():
        train_files = create_file_list(
            train_dir,
            Path(args.target_dir) / 'file_lists' / f'train_{args.noise_type}.txt'
        )
    
    if val_dir.exists():
        val_files = create_file_list(
            val_dir,
            Path(args.target_dir) / 'file_lists' / f'val_{args.noise_type}.txt'
        )
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
