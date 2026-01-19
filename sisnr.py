import torch
import numpy as np
import argparse
import librosa


def si_snr(reference, estimate, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
    
    Args:
        reference: clean reference signal
        estimate: estimated signal
        eps: small constant to avoid numerical issues
        
    Returns:
        SI-SNR value in dB
    """
    # Remove mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Calculate the scaling factor
    reference_energy = np.sum(reference ** 2) + eps
    scale = np.dot(estimate, reference) / reference_energy
    
    # Project estimate onto reference
    projection = scale * reference
    
    # Calculate noise
    noise = estimate - projection
    
    # Calculate SI-SNR
    si_snr_value = 10 * np.log10(np.sum(projection ** 2) / (np.sum(noise ** 2) + eps) + eps)
    
    return si_snr_value


def si_snri(clean, noisy, enhanced, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio Improvement (SI-SNRi)
    
    Args:
        clean: clean reference signal
        noisy: noisy input signal
        enhanced: enhanced/separated signal
        eps: small constant to avoid numerical issues
        
    Returns:
        SI-SNRi value in dB (improvement)
    """
    # Ensure all signals have the same length
    min_len = min(len(clean), len(noisy), len(enhanced))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]
    
    # Calculate SI-SNR for noisy and enhanced signals
    si_snr_noisy = si_snr(clean, noisy, eps)
    si_snr_enhanced = si_snr(clean, enhanced, eps)
    
    # SI-SNRi is the improvement
    si_snri_value = si_snr_enhanced - si_snr_noisy
    
    return si_snri_value, si_snr_noisy, si_snr_enhanced



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_file', type=str, help='Input clean file, wav format')
    parser.add_argument('--mix_file', type=str, help='Audio mix file, wav format')
    parser.add_argument('--sep_file', type=str, help='Separated speech file, wav format')

    args = parser.parse_args()
    clean_file = args.clean_file
    mix_file = args.mix_file
    sep_file = args.sep_file

    # Load audio files using librosa
    print(f'Loading audio files...')
    clean, sr_clean = librosa.load(clean_file, sr=None)
    mixture, sr_mix = librosa.load(mix_file, sr=None)
    separated, sr_sep = librosa.load(sep_file, sr=None)
    
    # Ensure all have the same sample rate
    if not (sr_clean == sr_mix == sr_sep):
        print(f'Warning: Sample rates differ (clean: {sr_clean}, mix: {sr_mix}, sep: {sr_sep})')
    
    # Ensure all signals have the same length
    min_len = min(len(clean), len(mixture), len(separated))
    clean = clean[:min_len]
    mixture = mixture[:min_len]
    separated = separated[:min_len]

    # Calculate SI-SNRi (using numpy arrays directly)
    si_snri_value, si_snr_mix, si_snr_sep = si_snri(clean, mixture, separated)
    
    print('\n' + '='*60)
    print('SI-SNR Metrics')
    print('='*60)
    print(f'SI-SNR (mixture):   {si_snr_mix:.2f} dB')
    print(f'SI-SNR (separated): {si_snr_sep:.2f} dB')
    print(f'SI-SNRi:            {si_snri_value:.2f} dB')
    print('='*60)