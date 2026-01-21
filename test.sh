

input_video=../../data/lrs2_v1-sample/mvlrs_v1/main/5535415699068794046/00001.mp4
input_noise=./data/noise.wav

basename=$(basename ${input_video} .mp4)
timestamp=$(date +%Y%m%d%H%M%S)
result_dir=./results/${basename}_${timestamp}

# Extract audio and video features from a video file and mix audio file (with noise)
python extract_audio_video.py \
    --video_file ${input_video} \
    --noise_file ${input_noise} \
    --audio_file ${result_dir}/clean.wav \
    --mix_file ${result_dir}/mix.wav \
    --video_feat ${result_dir}/video.npy \
    --snr_db 0

# Run all models to separate the audio (with noise)
python speech_enhance_VAE.py \
    --clean_file ${result_dir}/clean.wav \
    --mix_file ${result_dir}/mix.wav \
    --video_feat ${result_dir}/video.npy \
    --result_dir ${result_dir}


# # Run the different model to separate the audio (with noise)
# python test_avse.py \
#     --mode AV_VAE \
#     --models_dir ./saved_model/ \
#     --clean_file ${result_dir}/clean.wav \
#     --mix_file ${result_dir}/mix.wav \
#     --video_feat ${result_dir}/video.npy \
#     --result_dir ${result_dir}
