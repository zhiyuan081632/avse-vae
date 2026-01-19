#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从输入视频 input.mp4 一键生成：
  1. ./data/mix.wav      (16kHz 单声道音频)
  2. ./data/video.npy    (67x67 嘴部特征，形状 (4489, T))
"""

import cv2
import numpy as np
import os
import librosa
import soundfile as sf
import sys
import argparse


def extract_audio_to_wav(video_file, audio_file, target_sr=16000):
    """
    从视频中提取音频，重采样到 target_sr，保存为单声道 .wav 文件。
    
    :param video_file: 输入视频路径，例如 'input.mp4'
    :param audio_file: 输出音频路径，例如 './data/mix.wav'
    :param target_sr: 目标采样率，默认 16000 Hz
    """
    print(f'[Audio] Extracting audio from {video_file} ...')
    
    # OpenCV 打开视频，获取音频轨（需注意 OpenCV 不直接支持音频提取）
    # 这里采用临时提取 + librosa 重采样方案
    import tempfile
    import subprocess
    import shutil
    
    # 检查是否有 ffmpeg
    if shutil.which('ffmpeg') is None:
        raise RuntimeError(
            'ffmpeg not found. Please install ffmpeg or use moviepy method.'
        )
    
    # 用 ffmpeg 提取音频到临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_audio_path = tmp.name
    
    cmd = [
        'ffmpeg', '-y', '-i', video_file,
        '-vn',  # 不要视频流
        '-acodec', 'pcm_s16le',
        '-ar', str(target_sr),
        '-ac', '1',  # 单声道
        tmp_audio_path
    ]
    
    ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret.returncode != 0:
        raise RuntimeError(f'ffmpeg failed to extract audio from {video_file}')
    
    # 读取临时音频，确保采样率和格式正确
    audio, sr = librosa.load(tmp_audio_path, sr=target_sr, mono=True)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(audio_file)), exist_ok=True)
    
    # 保存为 wav
    sf.write(audio_file, audio, sr)
    
    # 删除临时文件
    os.remove(tmp_audio_path)
    
    print(f'[Audio] Saved audio to {audio_file}, sr={sr}, length={len(audio)/sr:.2f}s')


def mix_audio_with_noise(clean_audio, noise_audio, snr_db=0):
    """
    将干净音频和噪声按指定信噪比(SNR)混合
    
    :param clean_audio: 干净音频信号 (numpy array)
    :param noise_audio: 噪声信号 (numpy array)
    :param snr_db: 信噪比(dB)，默认0dB。值越大，噪声越小；值越小，噪声越大
    :return: 混合后的音频信号
    """
    # 如果噪声长度不够，循环填充
    if len(noise_audio) < len(clean_audio):
        repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)[:len(clean_audio)]
    else:
        # 如果噪声太长，从头开始截取对应长度
        noise_audio = noise_audio[:len(clean_audio)]
    
    # 计算干净信号和噪声的能量
    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    # 根据SNR计算噪声缩放因子
    # SNR(dB) = 10*log10(clean_power/noise_power_target)
    # Therefore: noise_power_target = clean_power / (10^(snr_db/10))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power_target = clean_power / snr_linear
    scale_factor = np.sqrt(noise_power_target / noise_power)
    
    # 缩放噪声
    scaled_noise = noise_audio * scale_factor
    
    # 预测混合后的最大幅度，提前缩放以避免削波
    # 使用保守估计：假设信号和噪声峰值可能叠加
    max_clean = np.max(np.abs(clean_audio))
    max_noise = np.max(np.abs(scaled_noise))
    predicted_max = max_clean + max_noise
    
    # 如果预测会超过0.95，提前按比例缩放两者，保持SNR不变
    if predicted_max > 0.95:
        pre_scale = 0.95 / predicted_max
        clean_audio = clean_audio * pre_scale
        scaled_noise = scaled_noise * pre_scale
    
    # 混合音频
    mixed_audio = clean_audio + scaled_noise
    
    return mixed_audio


def extract_mouth_patches_to_npy(
    video_file,
    save_path='video.npy',
    output_size=(67, 67),
    max_frames=None
):
    """
    从输入视频中提取嘴部区域（粗略用下半张脸代替），
    每帧生成一个 67x67 的灰度图，展平后组成 (D, T) 的矩阵，保存为 .npy

    :param video_file: 输入视频路径，例如 'input.mp4'
    :param save_path: 输出 .npy 文件路径，例如 './data/video.npy'
    :param output_size: 工程里用 67x67，这里默认 (67, 67)
    :param max_frames: 最多处理多少帧，为 None 表示处理整个视频
    """
    print(f'[Video] Extracting mouth features from {video_file} ...')

    # OpenCV 自带的人脸检测器（Haar Cascade）
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f'Failed to load face cascade from {face_cascade_path}')

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_file}')

    mouth_vecs = []  # 每帧一个 (67*67,) 向量

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        frame_count += 1
        if max_frames is not None and frame_count > max_frames:
            break

        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            # 没检测到人脸时，简单起见：用整幅图的下半部分
            h, w = gray.shape
            mouth_region = gray[h//2:h, :]
        else:
            # 取检测到的第一张脸
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]

            # 简单取人脸下半部分作为"嘴部区域"
            mouth_region = face_roi[h//2:h, :]

        # 调整嘴部区域大小到 67x67
        mouth_resized = cv2.resize(mouth_region, output_size, interpolation=cv2.INTER_AREA)

        # 归一化到 [0, 1]，再展平
        mouth_norm = mouth_resized.astype(np.float32) / 255.0
        mouth_vec = mouth_norm.reshape(-1)  # (67*67,)

        mouth_vecs.append(mouth_vec)

    cap.release()

    if len(mouth_vecs) == 0:
        raise RuntimeError('No frames were processed from the video.')

    # 现在 mouth_vecs 是一个长度为 T 的 list，每个元素是 (D,)
    # 先堆叠成 (T, D)
    V = np.stack(mouth_vecs, axis=0)  # (T, D)

    # 再转置为 (D, T)，与工程中 v 的格式一致
    V = V.T  # (D, T)

    # 确保类型为 float32
    V = V.astype(np.float32)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    np.save(save_path, V)
    print(f'[Video] Saved video features to {save_path}, shape = {V.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='Input video file, mp4 format') # 输入视频
    parser.add_argument('--noise_file', type=str, default='', help='Input noise file (optional), wav format')
    parser.add_argument('--audio_file', type=str, help='Output audio file, wav format')
    parser.add_argument('--mix_file', type=str, help='Audio mix file, wav format')
    parser.add_argument('--video_feat', type=str, help='Video feature file, npy format')
    parser.add_argument('--snr_db', type=float, default=0.0, help='Signal-to-Noise Ratio in dB (default: 0.0).')

    args = parser.parse_args()
    video_file = args.video_file
    noise_file = args.noise_file
    audio_file = args.audio_file    
    mix_file = args.mix_file
    video_feat = args.video_feat
    snr_db = args.snr_db

    
    print(f'\n\nProcessing video: {video_file}')
    
    # 1. 提取音频
    if noise_file and os.path.exists(noise_file):
        # 如果提供了噪声文件，则提取干净音频到audio_file
        extract_audio_to_wav(
            video_file=video_file,
            audio_file=audio_file,
            target_sr=16000
        )
        
        # 加载干净音频和噪声
        print(f'[Audio] Loading clean audio from {audio_file} ...')
        print(f'[Audio] Loading noise from {noise_file} ...')
        clean_audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        noise_audio, _ = librosa.load(noise_file, sr=16000, mono=True)
        
        # 混合音频
        print(f'[Audio] Mixing audio with noise at SNR = {snr_db} dB ...')
        mixed_audio = mix_audio_with_noise(clean_audio, noise_audio, snr_db=snr_db)
        
        # 保存混合后的音频
        os.makedirs(os.path.dirname(os.path.abspath(mix_file)), exist_ok=True)
        sf.write(mix_file, mixed_audio, sr)
        print(f'[Audio] Saved mixed audio to {mix_file}, sr={sr}, length={len(mixed_audio)/sr:.2f}s')
    else:
        # 没有噪声文件，直接提取音频到audio_file
        if noise_file:
            print(f'[Warning] Noise file not found: {noise_file}, skipping noise mixing.')
        extract_audio_to_wav(
            video_file=video_file,
            audio_file=audio_file,
            target_sr=16000
        )
    
    # 2. 提取视频嘴部特征
    extract_mouth_patches_to_npy(
        video_file=video_file,
        save_path=video_feat,
        output_size=(67, 67),
        max_frames=None  # 可选：限制帧数方便调试，None 表示处理全部
    )
    
    print(f'\n\nProcessing completed! You can now run speech_enhance_VAE.py')
