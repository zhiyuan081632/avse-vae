#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-experiment wrapper for AV-VAE speech enhancement.

This script runs ONE selected enhancement configuration (A-VAE, AV-VAE, AV-CVAE,
A-VAE-VI, AV-CVAE-VI) based on command-line arguments, instead of running all
of them in one shot like speech_enhance_VAE.py.
"""

import os
import numpy as np
import torch
import soundfile as sf
import librosa
import torch.nn as nn
import argparse

from MCEM_algo import MCEM_algo, MCEM_algo_cvae, VI_MCEM_algo
from AV_VAE import myVAE, CVAERTied, myDecoder, CDecoderRTied
from sisnr import si_snr, si_snri

# ================== Network parameters ==================

input_dim = 513
latent_dim = 32
device = 'cpu'  # keep CPU as in original script
hidden_dim_encoder = [128]
activation = torch.tanh
activationv = nn.ReLU()
landmarks_dim = 67 * 67  # raw visual data: 67x67=4489

# ================== MCEM / VI-MCEM parameters ==================

niter_MCEM = 100       # MCEM iterations
niter_MH = 40          # MH samples
burnin = 30            # MH burn-in
var_MH = 0.01          # MH proposal variance
tol = 1e-5             # MCEM stopping tolerance

# ================== STFT parameters ==================

wlen_sec = 64e-3
hop_percent = 0.521
fs_default = 16000

wlen = int(wlen_sec * fs_default)
wlen = int(np.power(2, np.ceil(np.log2(wlen))))  # next power of 2
nfft = wlen
hop = int(hop_percent * wlen)
win = np.sin(np.arange(.5, wlen - .5 + 1) / wlen * np.pi)  # sine window


def prepare_data(mix_file, clean_file, video_feat):
    """Load mixture, optional clean audio, video features and init NMF.

    Returns a dict with all shared data used by different enhancement modes.
    """
    # Read input audio and video observations
    x, fs = librosa.load(mix_file, sr=None)
    x = x / np.max(np.abs(x))
    v = np.load(video_feat)
    T_orig = len(x)

    # Load clean audio if provided (for SI-SNRi calculation)
    clean_audio = None
    if clean_file and os.path.exists(clean_file):
        clean_audio, _ = librosa.load(clean_file, sr=fs)
        # Ensure same length as mixture
        if len(clean_audio) > T_orig:
            clean_audio = clean_audio[:T_orig]
        elif len(clean_audio) < T_orig:
            clean_audio = np.pad(clean_audio, (0, T_orig - len(clean_audio)))
        print(f'[Info] Loaded clean audio from {clean_file} for SI-SNRi calculation')
    else:
        print('[Info] No clean audio file provided, SI-SNRi will not be calculated')

    # STFT of mixture
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
    X_abs_2 = np.abs(X) ** 2

    F, N = X.shape

    # Align video frames to spectrogram frames
    Nl = np.maximum(N, v.shape[1])
    if v.shape[1] < Nl:
        v = np.hstack((v, np.tile(v[:, [-1]], Nl - v.shape[1])))

    v = v.T
    v = torch.from_numpy(v.astype(np.float32))
    v.requires_grad = False

    # Random initialization of NMF parameters
    eps = np.finfo(float).eps
    np.random.seed(0)
    K_b = 10
    W0 = np.maximum(np.random.rand(F, K_b), eps)
    H0 = np.maximum(np.random.rand(K_b, N), eps)

    data = {
        'x': x,
        'fs': fs,
        'clean_audio': clean_audio,
        'X': X,
        'X_abs_2': X_abs_2,
        'v': v,
        'Nl': Nl,
        'W0': W0,
        'H0': H0,
        'T_orig': T_orig,
    }
    return data


def run_a_vae(models_dir, result_dir, data):
    """Audio-only VAE + MCEM (A-VAE)."""
    x = data['x']
    fs = data['fs']
    clean_audio = data['clean_audio']
    X = data['X']
    X_abs_2 = data['X_abs_2']
    Nl = data['Nl']
    W0 = data['W0']
    H0 = data['H0']
    T_orig = data['T_orig']

    saved_model_a_vae = os.path.join(models_dir, 'A_VAE_checkpoint.pt')
    vae = myVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder,
        activation=activation,
        activationv=activationv,
        blockZ=0.0,
        blockVenc=1.0,
        blockVdec=1.0,
        x_block=0.0,
        landmarks_dim=1280,
    ).to(device)

    checkpoint = torch.load(saved_model_a_vae, map_location='cpu', weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder = myDecoder(vae)

    vae.eval()
    decoder.eval()

    # do not update network parameters
    for param in decoder.parameters():
        param.requires_grad = False

    # zero video features for audio-only case
    v0 = np.zeros((Nl, 1280), dtype=np.float32)
    v0 = torch.from_numpy(v0)
    v0.requires_grad = False

    with torch.no_grad():
        data_orig = X_abs_2
        data_tensor = torch.from_numpy(data_orig.T.astype(np.float32)).to(device)
        vae.eval()
        z, _ = vae.encode(data_tensor, v0)
        z = torch.t(z)

    Z_init = z.numpy()

    mcem_algo = MCEM_algo(
        X=X,
        W=W0,
        H=H0,
        Z=Z_init,
        v=v0,
        decoder=decoder,
        niter_MCEM=niter_MCEM,
        niter_MH=niter_MH,
        burnin=burnin,
        var_MH=var_MH,
    )

    cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)

    mcem_algo.separate(niter_MH=100, burnin=75)
    s_hat = librosa.istft(mcem_algo.S_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)
    b_hat = librosa.istft(mcem_algo.N_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)

    save_vae = os.path.join(result_dir, 'A-VAE')
    os.makedirs(save_vae, exist_ok=True)
    sf.write(os.path.join(save_vae, 'est_speech.wav'), s_hat, fs)
    sf.write(os.path.join(save_vae, 'est_noise.wav'), b_hat, fs)

    if clean_audio is not None:
        si_snri_val, si_snr_noisy, si_snr_enhanced = si_snri(clean_audio, x, s_hat)
        print('A-VAE Results:')
        print(f'  SI-SNR (noisy): {si_snr_noisy:.2f} dB')
        print(f'  SI-SNR (enhanced): {si_snr_enhanced:.2f} dB')
        print(f'  SI-SNRi: {si_snri_val:.2f} dB')
        with open(os.path.join(save_vae, 'metrics.txt'), 'w') as f:
            f.write(f'SI-SNR (noisy): {si_snr_noisy:.2f} dB\n')
            f.write(f'SI-SNR (enhanced): {si_snr_enhanced:.2f} dB\n')
            f.write(f'SI-SNRi: {si_snri_val:.2f} dB\n')

    print('A-VAE finished ...')


def run_av_vae(models_dir, result_dir, data):
    """Audio-visual VAE + MCEM (AV-VAE)."""
    x = data['x']
    fs = data['fs']
    clean_audio = data['clean_audio']
    X = data['X']
    X_abs_2 = data['X_abs_2']
    v = data['v']
    W0 = data['W0']
    H0 = data['H0']
    T_orig = data['T_orig']

    saved_model_av_vae = os.path.join(models_dir, 'AV_VAE_checkpoint.pt')
    vae = myVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder,
        activation=activation,
        activationv=activationv,
        blockZ=0.0,
        blockVenc=0.0,
        blockVdec=0.0,
        x_block=0.0,
        landmarks_dim=landmarks_dim,
    ).to(device)

    checkpoint = torch.load(saved_model_av_vae, map_location='cpu', weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder = myDecoder(vae)

    vae.eval()
    decoder.eval()

    for param in decoder.parameters():
        param.requires_grad = False

    with torch.no_grad():
        data_orig = X_abs_2
        data_tensor = torch.from_numpy(data_orig.T.astype(np.float32)).to(device)
        vae.eval()
        z, _ = vae.encode(data_tensor, v)
        z = torch.t(z)

    Z_init = z.numpy()

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
        var_MH=var_MH,
    )

    cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)
    mcem_algo.separate(niter_MH=100, burnin=75)

    s_hat = librosa.istft(mcem_algo.S_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)
    b_hat = librosa.istft(mcem_algo.N_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)

    save_vae = os.path.join(result_dir, 'AV-VAE')
    os.makedirs(save_vae, exist_ok=True)
    sf.write(os.path.join(save_vae, 'est_speech.wav'), s_hat, fs)
    sf.write(os.path.join(save_vae, 'est_noise.wav'), b_hat, fs)

    if clean_audio is not None:
        si_snri_val, si_snr_noisy, si_snr_enhanced = si_snri(clean_audio, x, s_hat)
        print('AV-VAE Results:')
        print(f'  SI-SNR (noisy): {si_snr_noisy:.2f} dB')
        print(f'  SI-SNR (enhanced): {si_snr_enhanced:.2f} dB')
        print(f'  SI-SNRi: {si_snri_val:.2f} dB')
        with open(os.path.join(save_vae, 'metrics.txt'), 'w') as f:
            f.write(f'SI-SNR (noisy): {si_snr_noisy:.2f} dB\n')
            f.write(f'SI-SNR (enhanced): {si_snr_enhanced:.2f} dB\n')
            f.write(f'SI-SNRi: {si_snri_val:.2f} dB\n')

    print('AV-VAE finished ...')


def run_av_cvae(models_dir, result_dir, data):
    """Conditional AV-CVAE + MCEM (original MCEM_algo_cvae)."""
    x = data['x']
    fs = data['fs']
    clean_audio = data['clean_audio']
    X = data['X']
    X_abs_2 = data['X_abs_2']
    v = data['v']
    W0 = data['W0']
    H0 = data['H0']
    T_orig = data['T_orig']

    saved_model_av_cvae = os.path.join(models_dir, 'AV_CVAE_checkpoint.pt')
    vae = CVAERTied(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder,
        activation=activation,
        activationV=activationv,
    ).to(device)

    checkpoint = torch.load(saved_model_av_cvae, map_location='cpu', weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder = CDecoderRTied(vae)

    vae.eval()
    decoder.eval()

    for param in decoder.parameters():
        param.requires_grad = False

    with torch.no_grad():
        data_orig = X_abs_2
        data_tensor = torch.from_numpy(data_orig.T.astype(np.float32)).to(device)
        vae.eval()
        z, _ = vae.encode(data_tensor, v)
        z = torch.t(z)
        mu_z, logvar_z = vae.zprior(v)

    Z_init = z.numpy()
    mu_z = mu_z.numpy()
    logvar_z = logvar_z.numpy()

    mcem_algo = MCEM_algo_cvae(
        mu_z,
        logvar_z,
        X=X,
        W=W0,
        H=H0,
        Z=Z_init,
        v=v,
        decoder=decoder,
        niter_MCEM=niter_MCEM,
        niter_MH=niter_MH,
        burnin=burnin,
        var_MH=var_MH,
    )

    cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)
    mcem_algo.separate(niter_MH=100, burnin=75)

    s_hat = librosa.istft(mcem_algo.S_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)
    b_hat = librosa.istft(mcem_algo.N_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)

    save_vae = os.path.join(result_dir, 'AV-CVAE')
    os.makedirs(save_vae, exist_ok=True)
    sf.write(os.path.join(save_vae, 'est_speech.wav'), s_hat, fs)
    sf.write(os.path.join(save_vae, 'est_noise.wav'), b_hat, fs)

    if clean_audio is not None:
        si_snri_val, si_snr_noisy, si_snr_enhanced = si_snri(clean_audio, x, s_hat)
        print('AV-CVAE Results:')
        print(f'  SI-SNR (noisy): {si_snr_noisy:.2f} dB')
        print(f'  SI-SNR (enhanced): {si_snr_enhanced:.2f} dB')
        print(f'  SI-SNRi: {si_snri_val:.2f} dB')
        with open(os.path.join(save_vae, 'metrics.txt'), 'w') as f:
            f.write(f'SI-SNR (noisy): {si_snr_noisy:.2f} dB\n')
            f.write(f'SI-SNR (enhanced): {si_snr_enhanced:.2f} dB\n')
            f.write(f'SI-SNRi: {si_snri_val:.2f} dB\n')

    print('AV-CVAE finished ...')


def run_a_vae_vi(models_dir, result_dir, data):
    """Audio-only VAE + VI-MCEM (A-VAE-VI)."""
    x = data['x']
    fs = data['fs']
    clean_audio = data['clean_audio']
    X = data['X']
    X_abs_2 = data['X_abs_2']
    Nl = data['Nl']
    W0 = data['W0']
    H0 = data['H0']
    T_orig = data['T_orig']

    saved_model_a_vae = os.path.join(models_dir, 'A_VAE_checkpoint.pt')
    vae = myVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder,
        activation=activation,
        activationv=activationv,
        blockZ=0.0,
        blockVenc=1.0,
        blockVdec=1.0,
        x_block=0.0,
        landmarks_dim=1280,
    ).to(device)

    checkpoint = torch.load(saved_model_a_vae, map_location='cpu', weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder = myDecoder(vae)

    vae.eval()
    decoder.eval()

    for param in decoder.parameters():
        param.requires_grad = False

    v0 = np.zeros((Nl, 1280), dtype=np.float32)
    v0 = torch.from_numpy(v0)
    v0.requires_grad = False

    with torch.no_grad():
        data_orig = X_abs_2
        data_tensor = torch.from_numpy(data_orig.T.astype(np.float32)).to(device)
        vae.eval()
        z, _ = vae.encode(data_tensor, v0)
        z = torch.t(z)

    Z_init = z.numpy()
    mu_z = np.zeros_like(Z_init.T)
    logvar_z = np.ones_like(Z_init.T)

    vi_mcem_algo = VI_MCEM_algo(
        mu_z,
        logvar_z,
        X=X,
        W=W0,
        H=H0,
        Z=Z_init,
        v=v0,
        decoder=decoder,
        niter_VI_MCEM=niter_MCEM,
        niter_MH=niter_MH,
        burnin=burnin,
        var_MH=var_MH,
    )

    S_hat, N_hat = vi_mcem_algo.run()

    s_hat = librosa.istft(S_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)
    b_hat = librosa.istft(N_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)

    save_vae = os.path.join(result_dir, 'A-VAE-VI')
    os.makedirs(save_vae, exist_ok=True)
    sf.write(os.path.join(save_vae, 'est_speech.wav'), s_hat, fs)
    sf.write(os.path.join(save_vae, 'est_noise.wav'), b_hat, fs)

    if clean_audio is not None:
        si_snri_val, si_snr_noisy, si_snr_enhanced = si_snri(clean_audio, x, s_hat)
        print('A-VAE-VI Results:')
        print(f'  SI-SNR (noisy): {si_snr_noisy:.2f} dB')
        print(f'  SI-SNR (enhanced): {si_snr_enhanced:.2f} dB')
        print(f'  SI-SNRi: {si_snri_val:.2f} dB')
        with open(os.path.join(save_vae, 'metrics.txt'), 'w') as f:
            f.write(f'SI-SNR (noisy): {si_snr_noisy:.2f} dB\n')
            f.write(f'SI-SNR (enhanced): {si_snr_enhanced:.2f} dB\n')
            f.write(f'SI-SNRi: {si_snri_val:.2f} dB\n')

    print('A-VAE-VI finished ...')


def run_av_cvae_vi(models_dir, result_dir, data):
    """Conditional AV-CVAE + VI-MCEM (AV-CVAE-VI)."""
    x = data['x']
    fs = data['fs']
    clean_audio = data['clean_audio']
    X = data['X']
    X_abs_2 = data['X_abs_2']
    v = data['v']
    W0 = data['W0']
    H0 = data['H0']
    T_orig = data['T_orig']

    saved_model_av_cvae = os.path.join(models_dir, 'AV_CVAE_checkpoint.pt')
    vae = CVAERTied(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder,
        activation=activation,
        activationV=activationv,
    ).to(device)

    checkpoint = torch.load(saved_model_av_cvae, map_location='cpu', weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    decoder = CDecoderRTied(vae)

    vae.eval()
    decoder.eval()

    for param in decoder.parameters():
        param.requires_grad = False

    with torch.no_grad():
        data_orig = X_abs_2
        data_tensor = torch.from_numpy(data_orig.T.astype(np.float32)).to(device)
        vae.eval()
        z, _ = vae.encode(data_tensor, v)
        z = torch.t(z)
        mu_z, logvar_z = vae.zprior(v)

    Z_init = z.numpy()
    mu_z = mu_z.numpy()
    logvar_z = logvar_z.numpy()

    vi_mcem_algo = VI_MCEM_algo(
        mu_z,
        logvar_z,
        X=X,
        W=W0,
        H=H0,
        Z=Z_init,
        v=v,
        decoder=decoder,
        niter_VI_MCEM=niter_MCEM,
        niter_MH=niter_MH,
        burnin=burnin,
        var_MH=var_MH,
    )

    S_hat, N_hat = vi_mcem_algo.run()

    s_hat = librosa.istft(S_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)
    b_hat = librosa.istft(N_hat, hop_length=hop, win_length=wlen, window=win, length=T_orig)

    save_vae = os.path.join(result_dir, 'AV-CVAE-VI')
    os.makedirs(save_vae, exist_ok=True)
    sf.write(os.path.join(save_vae, 'est_speech.wav'), s_hat, fs)
    sf.write(os.path.join(save_vae, 'est_noise.wav'), b_hat, fs)

    if clean_audio is not None:
        si_snri_val, si_snr_noisy, si_snr_enhanced = si_snri(clean_audio, x, s_hat)
        print('AV-CVAE-VI Results:')
        print(f'  SI-SNR (noisy): {si_snr_noisy:.2f} dB')
        print(f'  SI-SNR (enhanced): {si_snr_enhanced:.2f} dB')
        print(f'  SI-SNRi: {si_snri_val:.2f} dB')
        with open(os.path.join(save_vae, 'metrics.txt'), 'w') as f:
            f.write(f'SI-SNR (noisy): {si_snr_noisy:.2f} dB\n')
            f.write(f'SI-SNR (enhanced): {si_snr_enhanced:.2f} dB\n')
            f.write(f'SI-SNRi: {si_snri_val:.2f} dB\n')

    print('AV-CVAE-VI finished ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run selected AV-VAE speech enhancement experiment')
    parser.add_argument('--mix_file', type=str, required=True, help='Audio mix file')
    parser.add_argument('--clean_file', type=str, default='', help='Clean audio file (optional, for SI-SNRi calculation)')
    parser.add_argument('--video_feat', type=str, required=True, help='Video feature file')
    parser.add_argument('--models_dir', type=str, default='./models/', help='Directory containing pretrained VAE checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results_avse', help='Directory to save enhancement results')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['A_VAE', 'AV_VAE', 'AV_CVAE', 'A_VAE_VI', 'AV_CVAE_VI'],
        required=True,
        help='Which model/inference configuration to run',
    )

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    data = prepare_data(args.mix_file, args.clean_file, args.video_feat)

    if args.mode == 'A_VAE':
        run_a_vae(args.models_dir, args.result_dir, data)
    elif args.mode == 'AV_VAE':
        run_av_vae(args.models_dir, args.result_dir, data)
    elif args.mode == 'AV_CVAE':
        run_av_cvae(args.models_dir, args.result_dir, data)
    elif args.mode == 'A_VAE_VI':
        run_a_vae_vi(args.models_dir, args.result_dir, data)
    elif args.mode == 'AV_CVAE_VI':
        run_av_cvae_vi(args.models_dir, args.result_dir, data)
